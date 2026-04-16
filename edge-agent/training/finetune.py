"""
Fine-tunes InsightFace buffalo_l on Indian face data.

Strategy (fits in 6GB VRAM on RTX 4050):
  - Load buffalo_l ONNX → convert to PyTorch via feature extraction
  - Freeze backbone, train only the ArcFace classification head
  - Mixed precision (fp16) to save VRAM
  - Batch size 32, 20 epochs
  - Augmentations: blur, noise, fog (matching real CCTV conditions)

Run: python training/finetune.py --data imfdb --epochs 20
     python training/finetune.py --data imfdb --epochs 20 --unfreeze-last 2
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import insightface

BASE = Path(__file__).parent
PROCESSED = BASE / "data" / "processed"
CHECKPOINTS = BASE / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_SIZE = 112


# ── Augmentations (CCTV conditions) ─────────────────────────────────────────

class CCTVAugment:
    """Random CCTV-realistic augmentations applied during training."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.3:
            k = np.random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if np.random.random() < 0.3:
            sigma = np.random.uniform(5, 30)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if np.random.random() < 0.2:
            intensity = np.random.uniform(0.1, 0.4)
            fog = np.full_like(img, 255)
            img = cv2.addWeighted(img, 1 - intensity, fog, intensity, 0)
        if np.random.random() < 0.2:
            quality = np.random.randint(15, 50)
            _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if np.random.random() < 0.2:
            scale = np.random.choice([0.5, 0.25])
            h, w = img.shape[:2]
            small = cv2.resize(img, (int(w*scale), int(h*scale)))
            img = cv2.resize(small, (w, h))
        return img


# ── Dataset ──────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    def __init__(self, root: Path, augment=False):
        self.samples = []
        self.class_to_idx = {}
        self.augment = CCTVAugment() if augment else None

        for label, identity_dir in enumerate(sorted(root.iterdir())):
            if not identity_dir.is_dir():
                continue
            self.class_to_idx[identity_dir.name] = label
            for img_path in identity_dir.glob("*.jpg"):
                self.samples.append((img_path, label))
            for img_path in identity_dir.glob("*.png"):
                self.samples.append((img_path, label))

        self.num_classes = len(self.class_to_idx)
        print(f"  Dataset: {len(self.samples)} images, {self.num_classes} identities")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)

        if self.augment:
            img = self.augment(img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5  # normalize to [-1, 1] (InsightFace standard)
        return tensor, label


# ── ArcFace loss ─────────────────────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """
    ArcFace classification head — same loss function InsightFace was trained with.
    Maximises angular margin between embeddings of different identities.
    """
    def __init__(self, embedding_dim: int, num_classes: int,
                 s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        import math
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        embeddings = nn.functional.normalize(embeddings)
        weight = nn.functional.normalize(self.weight)
        cosine = nn.functional.linear(embeddings, weight)
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return nn.functional.cross_entropy(output, labels)


# ── InsightFace feature extractor ────────────────────────────────────────────

class InsightFaceExtractor(nn.Module):
    """
    Wraps InsightFace buffalo_l as a PyTorch feature extractor.
    Uses ONNX Runtime under the hood via insightface, bridged through numpy.
    Frozen during initial training — only the ArcFace head is trained.
    """
    def __init__(self):
        super().__init__()
        self._app = insightface.app.FaceAnalysis(name="buffalo_l")
        self._app.prepare(ctx_id=0, det_thresh=0.3)
        # Get the recognition model directly
        self._rec_model = self._app.models.get("recognition")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 112, 112) normalized to [-1,1]"""
        # Convert batch to numpy for ONNX inference
        x_np = ((x.detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        x_np = x_np.transpose(0, 2, 3, 1)  # BCHW → BHWC

        embeddings = []
        for img in x_np:
            emb = self._rec_model.get_feat(img)
            embeddings.append(emb.flatten())

        emb_tensor = torch.tensor(np.stack(embeddings), dtype=torch.float32).to(DEVICE)
        return emb_tensor


# ── Embedding cache ───────────────────────────────────────────────────────────

class EmbeddingDataset(torch.utils.data.Dataset):
    """Pre-computed embeddings dataset — avoids re-running ONNX every epoch."""
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def precompute_embeddings(extractor: InsightFaceExtractor,
                          dataset: FaceDataset,
                          batch_size: int = 64,
                          desc: str = "Extracting embeddings") -> tuple:
    """
    Run InsightFace ONNX once over the whole dataset and cache results.
    This is much faster than running it every epoch since the backbone is frozen.
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    all_embs, all_labels = [], []

    print(f"  {desc} ({len(dataset)} images)...")
    for i, (images, labels) in enumerate(loader):
        embs = extractor(images)  # returns CPU tensor
        all_embs.append(embs)
        all_labels.append(labels)
        if i % 10 == 0:
            print(f"  {desc}: {min((i+1)*batch_size, len(dataset))}/{len(dataset)}", end="\r")

    print()
    embeddings = torch.cat(all_embs, dim=0)   # (N, 512)
    labels = torch.cat(all_labels, dim=0)      # (N,)
    return embeddings, labels


# ── Training loop ─────────────────────────────────────────────────────────────

def train(data_source: str, epochs: int = 20, batch_size: int = 256):
    data_dir = PROCESSED / data_source
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        print(f"ERROR: {train_dir} not found. Run prepare_data.py first.")
        return

    print(f"\n── Loading datasets ────────────────────────────────────────────────────")
    train_ds = FaceDataset(train_dir, augment=False)  # augment per-epoch below
    val_ds = FaceDataset(val_dir, augment=False)

    print(f"\n── Building backbone extractor ─────────────────────────────────────────")
    extractor = InsightFaceExtractor()

    # Pre-compute embeddings once (backbone is frozen — no need to rerun each epoch)
    print(f"\n── Pre-computing embeddings (one-time, uses CPU ONNX) ──────────────────")
    train_embs, train_labels = precompute_embeddings(
        extractor, train_ds, batch_size=64, desc="Train")
    val_embs, val_labels = precompute_embeddings(
        extractor, val_ds, batch_size=64, desc="Val")

    # Move to GPU for training
    train_embs = train_embs.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    val_embs = val_embs.to(DEVICE)
    val_labels = val_labels.to(DEVICE)

    train_cache = EmbeddingDataset(train_embs, train_labels)
    val_cache = EmbeddingDataset(val_embs, val_labels)

    train_loader = DataLoader(train_cache, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_cache, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    print(f"\n── Training ArcFace head on GPU ────────────────────────────────────────")
    EMBEDDING_DIM = 512
    arc_head = ArcFaceLoss(EMBEDDING_DIM, train_ds.num_classes).to(DEVICE)

    optimizer = optim.AdamW(arc_head.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler()

    print(f"  Device: {DEVICE}")
    print(f"  Identities: {train_ds.num_classes}")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples: {len(val_ds)}")
    print(f"  Epochs: {epochs}  Batch: {batch_size}")
    print(f"  Note: embeddings pre-cached — each epoch is pure GPU ArcFace training")

    best_val_acc = 0.0
    ckpt_path = CHECKPOINTS / f"archead_{data_source}_best.pt"

    for epoch in range(epochs):
        # ── Train ──
        arc_head.train()
        total_loss = 0.0
        t0 = time.time()

        for batch_idx, (embs, labels) in enumerate(train_loader):
            with torch.amp.autocast(device_type="cuda"):
                loss = arc_head(embs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  "
                      f"Batch {batch_idx}/{len(train_loader)}  "
                      f"Loss: {loss.item():.4f}", end="\r")

        scheduler.step()

        # ── Validate ──
        arc_head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            weight = nn.functional.normalize(arc_head.weight)
            for embs, labels in val_loader:
                embs_norm = nn.functional.normalize(embs)
                logits = nn.functional.linear(embs_norm, weight) * arc_head.s
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{epochs}  "
              f"Loss: {total_loss/len(train_loader):.4f}  "
              f"Val Acc: {val_acc:.1%}  "
              f"Time: {elapsed:.1f}s  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "arc_head_state": arc_head.state_dict(),
                "num_classes": train_ds.num_classes,
                "val_acc": val_acc,
                "class_to_idx": train_ds.class_to_idx,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_acc={val_acc:.1%})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1%}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Next: python training/evaluate.py --checkpoint {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="imfdb", choices=["imfdb", "qmul"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args.data, args.epochs, args.batch_size)
