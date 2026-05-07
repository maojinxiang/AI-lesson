import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


VALID_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def normalize_label(raw_label: str) -> str:
    """Normalize label aliases to canonical names."""
    label = raw_label.strip().lower()
    alias_map = {
        "surprised": "surprise",
        "calm": "neutral",
    }
    return alias_map.get(label, label)


def scan_emotion_samples(data_root: str) -> List[Tuple[str, str]]:
    """Recursively scan audio files and infer label from parent folder name.

    Returns:
        List of (wav_path, label_name)
    """
    root = Path(data_root)
    if not root.exists():
        raise RuntimeError(f"Data directory not found: {data_root}")

    samples: List[Tuple[str, str]] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VALID_AUDIO_EXT:
            continue
        label_name = normalize_label(p.parent.name)
        samples.append((str(p), label_name))

    return samples


def build_label2id_from_samples(samples: List[Tuple[str, str]]) -> Dict[str, int]:
    labels = sorted({label for _, label in samples})
    if len(labels) < 2:
        raise RuntimeError(
            "Need at least 2 classes with audio files. "
            "Please check your dataset folders and labels."
        )
    return {name: i for i, name in enumerate(labels)}


def extract_feature(
    wav_path: str,
    sr: int = 16000,
    n_mels: int = 64,
    n_mfcc: int = 40,
    feature_type: str = "mel",
) -> np.ndarray:
    """Load audio and extract time-frequency feature.

    Returns shape: [freq_bins, time_steps]
    """
    y, _ = librosa.load(wav_path, sr=sr)

    if feature_type == "mfcc":
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    elif feature_type == "mel":
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        feat = librosa.power_to_db(mel, ref=np.max)
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    return feat.astype(np.float32)


class SpeechEmotionDataset(Dataset):
    """Dataset for speech emotion classification.

    Directory structure expected:
        data_root/
            class_a/
                a.wav
            class_b/
                b.wav
    """

    def __init__(
        self,
        data_root: str,
        label2id: dict,
        samples: Optional[List[Tuple[str, int]]] = None,
        seq_len: int = 300,
        sr: int = 16000,
        n_mels: int = 64,
        n_mfcc: int = 40,
        feature_type: str = "mel",
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.label2id = label2id
        self.seq_len = seq_len
        self.sr = sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.feature_type = feature_type

        self.samples: List[Tuple[str, int]] = samples if samples is not None else self._scan_files()
        if len(self.samples) == 0:
            raise RuntimeError(
                "No audio files found. Please check data folder and labels. "
                "Expected folders like data/happy, data/sad or any class subfolders. "
                "If you do not have real data yet, run: python generate_dummy_data.py"
            )

    def _scan_files(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        valid_ext = VALID_AUDIO_EXT

        for label_name, label_id in self.label2id.items():
            label_dir = self.data_root / label_name
            if not label_dir.exists():
                continue

            for root, _, files in os.walk(label_dir):
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in valid_ext:
                        samples.append((str(Path(root) / f), label_id))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_or_truncate(self, x: np.ndarray) -> np.ndarray:
        # x: [freq_bins, time_steps]
        t = x.shape[1]
        if t >= self.seq_len:
            x = x[:, : self.seq_len]
        else:
            pad_width = self.seq_len - t
            x = np.pad(x, ((0, 0), (0, pad_width)), mode="constant")
        return x

    def __getitem__(self, idx: int):
        wav_path, label = self.samples[idx]
        feat = extract_feature(
            wav_path,
            sr=self.sr,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc,
            feature_type=self.feature_type,
        )
        feat = self._pad_or_truncate(feat)

        # Transformer expects sequence first, so transpose to [T, F]
        feat = feat.T
        x = torch.tensor(feat, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
