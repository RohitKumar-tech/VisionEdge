# VisionEdge — InsightFace Fine-Tuning Pipeline

Trains InsightFace `buffalo_l` on Indian face data (IMFDB) to improve accuracy
on CCTV-quality footage: blur, fog, noise, low-resolution, JPEG compression.

## Hardware

- RTX 4050 Mobile (6 GB VRAM) — tested
- Min: 4 GB VRAM with batch_size=16
- CPU fallback works but is ~20× slower

## Step 1 — Set up Kaggle credentials

```bash
# 1. Go to https://www.kaggle.com/settings → API → Create New Token
# 2. Download kaggle.json and place it here:
mkdir -p ~/.config/kaggle
cp kaggle.json ~/.config/kaggle/kaggle.json
chmod 600 ~/.config/kaggle/kaggle.json
```

## Step 2 — Download datasets

```bash
cd edge-agent/training
python download_datasets.py
```

Downloads IMFDB (Indian Movie Face Database, ~34K images, 100 identities).
QMUL-SurvFace requires manual registration at https://qmul-survface.github.io/

## Step 3 — Prepare data

```bash
python prepare_data.py --source imfdb
```

- Aligns faces to 112×112 using InsightFace (same format buffalo_l expects)
- Splits 80% train / 10% val / 10% test
- Output: `training/data/processed/imfdb/{train,val,test}/`

## Step 4 — Fine-tune

```bash
# Freeze backbone, train only ArcFace head (fits in 6GB VRAM)
python finetune.py --data imfdb --epochs 20

# Optional: after initial training, unfreeze last 2 backbone layers
python finetune.py --data imfdb --epochs 10 --unfreeze-last 2
```

Checkpoint saved to `training/checkpoints/archead_imfdb_best.pt`

## Step 5 — Evaluate

```bash
# Compare fine-tuned vs original buffalo_l on all degradation levels
python evaluate.py --checkpoint checkpoints/archead_imfdb_best.pt --data imfdb
```

Reports TAR@FAR=1% for each degradation type and top-1 identification accuracy.

## Step 6 — Export / inspect

```bash
python export_model.py --checkpoint checkpoints/archead_imfdb_best.pt --data imfdb
```

Prints suggested `RECOGNITION_THRESHOLD` value and saves `model_info_imfdb.json`.

## Files

```
training/
├── download_datasets.py   # Step 2 — download IMFDB via Kaggle
├── prepare_data.py        # Step 3 — face alignment + train/val/test split
├── finetune.py            # Step 4 — ArcFace fine-tuning loop
├── evaluate.py            # Step 5 — TAR@FAR benchmark under CCTV degradations
├── export_model.py        # Step 6 — embedding stats + recognition threshold
├── data/
│   ├── imfdb/             # Raw IMFDB (from Kaggle)
│   └── processed/
│       └── imfdb/
│           ├── all/       # All aligned faces
│           ├── train/     # 80%
│           ├── val/       # 10%
│           └── test/      # 10%
└── checkpoints/
    ├── archead_imfdb_best.pt     # Best fine-tuned checkpoint
    ├── model_info_imfdb.json     # Embedding stats + suggested threshold
    └── eval_imfdb_<timestamp>.json  # Benchmark results
```

## CCTV Degradations Tested

| Degradation | Parameters |
|-------------|-----------|
| clean | No degradation |
| blur_mild | Gaussian blur k=5 |
| blur_heavy | Gaussian blur k=15 |
| fog_light | 20% white overlay |
| fog_heavy | 50% white overlay |
| noise_low | Gaussian σ=10 |
| noise_high | Gaussian σ=40 |
| lowres_half | Downscale 50% → upscale |
| lowres_qtr | Downscale 25% → upscale |
| jpeg_q30 | JPEG quality 30 |
| jpeg_q10 | JPEG quality 10 |
| combined | blur + noise + fog |

## After Training

Update your `.env`:

```
RECOGNITION_THRESHOLD=<value from export_model.py output>
```

The fine-tuned checkpoint improves the embedding metric space for Indian faces.
`FaceDetector.detect()` in `ai/face_detector.py` uses the buffalo_l backbone
directly — no code changes needed in the edge agent.
