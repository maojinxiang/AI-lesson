import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    SpeechEmotionDataset,
    build_label2id_from_samples,
    scan_emotion_samples,
)
from model import SpeechEmotionTransformer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(
    samples: List[Tuple[str, int]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    by_label: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for item in samples:
        by_label[item[1]].append(item)

    rng = random.Random(seed)
    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []
    test_samples: List[Tuple[str, int]] = []

    for _, cls_samples in by_label.items():
        cls_samples = cls_samples.copy()
        rng.shuffle(cls_samples)
        n = len(cls_samples)

        train_n = int(round(n * train_ratio))
        val_n = int(round(n * val_ratio))

        # Keep a strict held-out test set and avoid empty splits when possible.
        if n >= 3:
            train_n = min(max(1, train_n), n - 2)
            val_n = min(max(1, val_n), n - train_n - 1)
        elif n == 2:
            train_n, val_n = 1, 0
        else:
            train_n, val_n = 1, 0

        test_n = n - train_n - val_n
        if test_n <= 0 and n >= 2:
            if val_n > 0:
                val_n -= 1
            else:
                train_n -= 1
            test_n = 1

        train_samples.extend(cls_samples[:train_n])
        val_samples.extend(cls_samples[train_n : train_n + val_n])
        test_samples.extend(cls_samples[train_n + val_n :])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return train_samples, val_samples, test_samples


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_count += y.size(0)

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc


def evaluate_with_confusion(model, loader, device, num_classes: int):
    model.eval()
    total = 0
    correct = 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)

            total += y.size(0)
            correct += (pred == y).sum().item()
            for gt, pd in zip(y.cpu().tolist(), pred.cpu().tolist()):
                conf[gt, pd] += 1

    return correct / max(1, total), conf


def apply_specaugment(
    x: torch.Tensor,
    time_mask_param: int,
    freq_mask_param: int,
    num_masks: int = 2,
) -> torch.Tensor:
    """Apply simple SpecAugment on [B, T, F] features."""
    b, t, f = x.shape
    out = x.clone()
    for i in range(b):
        for _ in range(num_masks):
            if time_mask_param > 0 and t > 1:
                w = random.randint(0, min(time_mask_param, t - 1))
                if w > 0:
                    t0 = random.randint(0, t - w)
                    out[i, t0 : t0 + w, :] = 0.0
            if freq_mask_param > 0 and f > 1:
                w = random.randint(0, min(freq_mask_param, f - 1))
                if w > 0:
                    f0 = random.randint(0, f - w)
                    out[i, :, f0 : f0 + w] = 0.0
    return out


def compute_class_weights(samples: List[Tuple[str, int]], num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, y in samples:
        counts[y] += 1.0
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def save_history_plots(history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(9, 4))
    plt.plot(epochs, history["train_loss"], marker="o", label="train_loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(epochs, history["val_acc"], marker="o", color="tab:green", label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "val_acc_curve.png", dpi=160)
    plt.close()

    with open(output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "train_history.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_acc\n")
        for i in range(len(epochs)):
            f.write(
                f"{i+1},{history['train_loss'][i]:.6f},{history['val_loss'][i]:.6f},{history['val_acc'][i]:.6f}\n"
            )


def save_split_manifest(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    test_samples: List[Tuple[str, int]],
    id2label: Dict[int, str],
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _pack(items: List[Tuple[str, int]]):
        packed = []
        for wav_path, label_id in items:
            packed.append({
                "path": wav_path,
                "label": id2label[label_id],
                "label_id": int(label_id),
            })
        return packed

    payload = {
        "train": _pack(train_samples),
        "val": _pack(val_samples),
        "test": _pack(test_samples),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_split_stats(name: str, items: List[Tuple[str, int]], id2label: Dict[int, str]):
    counts: Dict[int, int] = defaultdict(int)
    for _, label_id in items:
        counts[label_id] += 1
    stat_str = ", ".join(f"{id2label[k]}={counts[k]}" for k in sorted(counts.keys()))
    print(f"{name}: {len(items)} samples | {stat_str}")


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_samples = scan_emotion_samples(args.data_dir)
    label2id = build_label2id_from_samples(raw_samples)
    print(f"Detected classes ({len(label2id)}): {list(label2id.keys())}")
    id2label = {v: k for k, v in label2id.items()}

    all_samples = [(path, label2id[label_name]) for path, label_name in raw_samples]
    train_samples, val_samples, test_samples = stratified_split(
        all_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print_split_stats("Train", train_samples, id2label)
    print_split_stats("Val", val_samples, id2label)
    print_split_stats("Test", test_samples, id2label)

    train_ds = SpeechEmotionDataset(
        data_root=args.data_dir,
        label2id=label2id,
        samples=train_samples,
        seq_len=args.seq_len,
        sr=args.sr,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        feature_type=args.feature_type,
    )
    val_ds = SpeechEmotionDataset(
        data_root=args.data_dir,
        label2id=label2id,
        samples=val_samples,
        seq_len=args.seq_len,
        sr=args.sr,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        feature_type=args.feature_type,
    )
    test_ds = SpeechEmotionDataset(
        data_root=args.data_dir,
        label2id=label2id,
        samples=test_samples,
        seq_len=args.seq_len,
        sr=args.sr,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        feature_type=args.feature_type,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    input_dim = args.n_mels if args.feature_type == "mel" else args.n_mfcc
    model = SpeechEmotionTransformer(
        input_dim=input_dim,
        num_classes=len(label2id),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.seq_len,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    if args.use_class_weight:
        cls_weights = compute_class_weights(train_samples, num_classes=len(label2id)).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=cls_weights,
            label_smoothing=args.label_smoothing,
        )
        print(f"Using class-weighted loss: {cls_weights.cpu().numpy().round(4).tolist()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.1,
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    no_improve_epochs = 0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(args.vis_dir)
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    split_path = Path(args.split_path)
    save_split_manifest(train_samples, val_samples, test_samples, id2label, split_path)
    print(f"Saved split manifest to: {split_path}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            if args.use_specaug and random.random() < args.specaug_prob:
                x = apply_specaugment(
                    x,
                    time_mask_param=args.specaug_time_mask,
                    freq_mask_param=args.specaug_freq_mask,
                    num_masks=args.specaug_num_masks,
                )

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            total += y.size(0)
            pbar.set_postfix(train_loss=running_loss / max(1, total))

        train_loss = running_loss / max(1, total)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6e}"
        )
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        improved = (val_acc > best_val_acc) or (
            np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "label2id": label2id,
                },
                save_path,
            )
            print(f"Saved best model to: {save_path}")

            save_history_plots(history, vis_dir)
            print(f"Saved visualizations to: {vis_dir}")
        else:
            no_improve_epochs += 1

        scheduler.step()

        # if args.patience > 0 and no_improve_epochs >= args.patience:
        #     print(f"Early stopping triggered at epoch {epoch}.")
        #     break

    # Evaluate the best checkpoint on strict held-out test set.
    best_ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_acc, test_conf = evaluate_with_confusion(
        model, test_loader, device, num_classes=len(label2id)
    )
    print(f"Training finished. Best val acc: {best_val_acc:.4f}")
    print(f"Held-out test acc: {test_acc:.4f}")
    print("Held-out test confusion matrix (rows=true, cols=pred):")
    print(test_conf.numpy())

    with open(vis_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_acc": float(best_val_acc),
                "test_acc": float(test_acc),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="CASIA database")
    parser.add_argument("--feature_type", type=str, default="mel", choices=["mel", "mfcc"])
    parser.add_argument("--seq_len", type=int, default=300)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--n_mfcc", type=int, default=40)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--use_specaug", action="store_true")
    parser.add_argument("--specaug_prob", type=float, default=0.7)
    parser.add_argument("--specaug_time_mask", type=int, default=24)
    parser.add_argument("--specaug_freq_mask", type=int, default=8)
    parser.add_argument("--specaug_num_masks", type=int, default=2)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--vis_dir", type=str, default="outputs/train")
    parser.add_argument("--split_path", type=str, default="outputs/train/data_split.json")

    args = parser.parse_args()
    main(args)
