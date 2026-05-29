"""
services.action_recognition.dataset
=====================================
Dataset loader and training script for the CNN+LSTM action recognition model.

Dataset structure expected:
    data/action_clips/
        walking/        ← video clips (.mp4, .avi, etc.)
        running/
        fighting/
        falling/
        loitering/
        suspicious_stationary/

Usage (Kaggle / local):
    python -m services.action_recognition.dataset \\
        --data_dir data/action_clips \\
        --output   weights/action_model.pt \\
        --epochs   30

See scripts/kaggle_train_action_model.py for a self-contained Kaggle notebook.
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ACTION_CLASSES = [
    "walking", "running", "fighting",
    "loitering", "falling", "suspicious_stationary", "unknown",
]
LABEL_TO_IDX = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
SEQ_LEN   = 16
CROP_SIZE = (112, 112)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Video loading ─────────────────────────────────────────────────────────────

def load_clip(video_path: str, seq_len: int = SEQ_LEN) -> list[np.ndarray] | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None

    indices = (
        np.linspace(0, total - 1, seq_len, dtype=int).tolist()
        if total >= seq_len
        else list(range(total))
    )
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.resize(frame, CROP_SIZE))
    cap.release()

    if not frames:
        return None
    while len(frames) < seq_len:
        frames.append(frames[-1].copy())
    return frames[:seq_len]


def augment_clip(frames: list[np.ndarray]) -> list[np.ndarray]:
    if random.random() > 0.5:
        frames = [f[:, ::-1, :].copy() for f in frames]
    factor = 1.0 + random.uniform(-0.15, 0.15)
    return [np.clip(f.astype(np.float32) * factor, 0, 255).astype(np.uint8) for f in frames]


def frame_to_tensor(frame: np.ndarray):
    import torch
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(norm.transpose(2, 0, 1))


# ── Dataset ───────────────────────────────────────────────────────────────────

try:
    from torch.utils.data import Dataset

    class ActionDataset(Dataset):
        def __init__(self, samples: list[tuple], augment: bool = True):
            self.samples = samples
            self.augment = augment

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            import torch
            path, label = self.samples[idx]
            frames = load_clip(path)
            if frames is None:
                frames = [np.zeros((CROP_SIZE[1], CROP_SIZE[0], 3), np.uint8)] * SEQ_LEN
            if self.augment:
                frames = augment_clip(frames)
            tensors = torch.stack([frame_to_tensor(f) for f in frames])
            return tensors, label

except ImportError:
    pass


def scan_dataset(data_dir: str) -> list[tuple]:
    root = Path(data_dir)
    video_exts = {".mp4", ".avi", ".mkv", ".mov"}
    samples = []
    for cls_name in ACTION_CLASSES:
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            continue
        idx = LABEL_TO_IDX[cls_name]
        for p in cls_dir.iterdir():
            if p.suffix.lower() in video_exts:
                samples.append((p, idx))
    logger.info("Found %d clips in %s", len(samples), data_dir)
    return samples


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_dir: str, output: str, epochs: int = 30, batch_size: int = 8, lr: float = 1e-3):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from services.action_recognition.model import ActionRecognitionModel
    except ImportError:
        raise RuntimeError("PyTorch is required for training. Install via pip install torch torchvision.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    samples = scan_dataset(data_dir)
    if not samples:
        raise FileNotFoundError(f"No video clips found in {data_dir}. "
                                "Organise clips as data_dir/class_name/*.mp4")

    random.shuffle(samples)
    val_n = max(1, int(len(samples) * 0.2))
    train_ds = ActionDataset(samples[val_n:], augment=True)
    val_ds   = ActionDataset(samples[:val_n], augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = ActionRecognitionModel(freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_n = 0.0, 0, 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * clips.size(0)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_n += clips.size(0)

        model.eval()
        v_loss, v_correct, v_n = 0.0, 0, 0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                logits = model(clips)
                v_loss += criterion(logits, labels).item() * clips.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_n += clips.size(0)

        t_loss /= max(t_n, 1);  t_acc = t_correct / max(t_n, 1) * 100
        v_loss /= max(v_n, 1);  v_acc = v_correct / max(v_n, 1) * 100
        scheduler.step(v_loss)

        marker = ""
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), output)
            marker = " ★ saved"

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train {t_acc:5.1f}% loss={t_loss:.4f} | "
              f"Val {v_acc:5.1f}% loss={v_loss:.4f}{marker}")

    print(f"\nDone! Best model saved to: {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train action recognition model")
    parser.add_argument("--data_dir", required=True, help="Root directory with class subfolders of video clips")
    parser.add_argument("--output",   default="weights/action_model.pt")
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--batch",    type=int,   default=8)
    parser.add_argument("--lr",       type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data_dir, args.output, args.epochs, args.batch, args.lr)
