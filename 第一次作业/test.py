import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    SpeechEmotionDataset,
    extract_feature,
    normalize_label,
)
from model import SpeechEmotionTransformer


def build_model_from_checkpoint(checkpoint, device):
    train_args = checkpoint["args"]
    label2id = checkpoint["label2id"]

    input_dim = train_args["n_mels"] if train_args["feature_type"] == "mel" else train_args["n_mfcc"]
    model = SpeechEmotionTransformer(
        input_dim=input_dim,
        num_classes=len(label2id),
        d_model=train_args["d_model"],
        nhead=train_args["nhead"],
        num_layers=train_args["num_layers"],
        dim_feedforward=train_args["dim_feedforward"],
        dropout=train_args["dropout"],
        max_len=train_args["seq_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, train_args, label2id


def evaluate_dataset(model, loader, device, num_classes):
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


def preprocess_single_wav(wav_path, train_args):
    feat = extract_feature(
        wav_path,
        sr=train_args["sr"],
        n_mels=train_args["n_mels"],
        n_mfcc=train_args["n_mfcc"],
        feature_type=train_args["feature_type"],
    )
    t = feat.shape[1]
    seq_len = train_args["seq_len"]
    if t >= seq_len:
        feat = feat[:, :seq_len]
    else:
        feat = np.pad(feat, ((0, 0), (0, seq_len - t)), mode="constant")

    # [freq, time] -> [1, time, freq]
    x = torch.tensor(feat.T, dtype=torch.float32).unsqueeze(0)
    return x


def save_test_visualizations(conf, id2label, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    conf_np = conf.numpy()
    labels = [id2label[i] for i in range(len(id2label))]

    plt.figure(figsize=(7, 6))
    plt.imshow(conf_np, cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    for i in range(conf_np.shape[0]):
        for j in range(conf_np.shape[1]):
            v = int(conf_np[i, j])
            plt.text(j, i, str(v), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    per_class_total = conf.sum(dim=1).clamp_min(1)
    per_class_acc = conf.diag().float() / per_class_total.float()
    per_class_acc_np = per_class_acc.numpy()

    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, per_class_acc_np, color="tab:orange")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, per_class_acc_np):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_accuracy.png", dpi=160)
    plt.close()


def load_samples_from_split(split_json: str, split_name: str, label2id: dict):
    with open(split_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if split_name not in payload:
        raise RuntimeError(f"Split '{split_name}' not found in {split_json}")

    samples = []
    for item in payload[split_name]:
        label_name = normalize_label(item["label"])
        if label_name not in label2id:
            raise RuntimeError(
                f"Label '{label_name}' in split file does not exist in checkpoint labels."
            )
        samples.append((item["path"], label2id[label_name]))
    return samples


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt, map_location=device)
    model, train_args, label2id = build_model_from_checkpoint(checkpoint, device)
    id2label = {v: k for k, v in label2id.items()}

    if args.wav_path:
        x = preprocess_single_wav(args.wav_path, train_args).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
        print(f"Predicted emotion: {id2label[pred]}")
        return

    samples = None
    if args.split_json:
        samples = load_samples_from_split(args.split_json, args.split_name, label2id)
        print(f"Evaluating split '{args.split_name}' from: {args.split_json}")

    dataset = SpeechEmotionDataset(
        data_root=args.data_dir,
        label2id=label2id,
        samples=samples,
        seq_len=train_args["seq_len"],
        sr=train_args["sr"],
        n_mels=train_args["n_mels"],
        n_mfcc=train_args["n_mfcc"],
        feature_type=train_args["feature_type"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    acc, conf = evaluate_dataset(model, loader, device, num_classes=len(label2id))
    print(f"Test accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(conf.numpy())

    per_class_total = conf.sum(dim=1)
    per_class_correct = conf.diag()
    for i in range(len(label2id)):
        cls_acc = float(per_class_correct[i].item()) / max(1, int(per_class_total[i].item()))
        print(f"{id2label[i]}: {cls_acc:.4f} ({int(per_class_correct[i])}/{int(per_class_total[i])})")

    out_dir = Path(args.vis_dir)
    save_test_visualizations(conf, id2label, out_dir)
    print(f"Saved test visualizations to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="CASIA database")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--wav_path", type=str, default="", help="Optional single wav for inference")
    parser.add_argument("--vis_dir", type=str, default="outputs/test")
    parser.add_argument("--split_json", type=str, default="outputs/train/data_split.json")
    parser.add_argument("--split_name", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    main(args)
