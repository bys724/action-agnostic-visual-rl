# Data Directory

이 디렉토리는 학습 데이터, 체크포인트, 결과를 저장합니다.
모든 하위 디렉토리는 `.gitignore`에 포함되어 있습니다.

## Directory Structure

```
data/
├── checkpoints/           # Model checkpoints
│   ├── two_stream/        # Two-Stream model (ours)
│   ├── single_stream/     # Single-Stream baseline
│   └── videomae/          # VideoMAE baseline
│       └── YYYYMMDD_HHMMSS/  # Training run timestamp
│           ├── config.json        # Training configuration
│           ├── history.json       # Training history (loss, metrics)
│           ├── best_model.pt      # Best checkpoint (lowest loss)
│           ├── latest.pt          # Latest checkpoint (for resume)
│           └── checkpoint_epoch*.pt  # Periodic checkpoints
│
├── datasets/              # Robot datasets
│   ├── bridge_v2/         # Bridge V2 (robot manipulation)
│   └── libero_rlds/       # LIBERO in RLDS format
│
└── egodex/                # EgoDex (human manipulation videos)
    ├── part1/             # Training data
    └── test/              # Test set (17GB)
```

## Usage

### Training Checkpoints

Checkpoints are automatically saved during training:

```bash
# Train with auto-checkpoint
python scripts/train_long.py --model two-stream --epochs 100

# Resume from checkpoint
python scripts/train_long.py --model two-stream --resume data/checkpoints/two_stream/20260304_120000/latest.pt
```

Checkpoint structure:
- `config.json`: Training hyperparameters
- `history.json`: Loss curves and metrics
- `best_model.pt`: Best model (for evaluation)
- `latest.pt`: Latest model (for resuming)
- `checkpoint_epoch*.pt`: Periodic snapshots (every N epochs)

### Datasets

**EgoDex** (Self-supervised pretraining):
- Source: Human manipulation videos
- Format: Pre-extracted JPEG frames
- Location (AWS): `/workspace/data/egodex/part1/`
- Location (S3): `s3://bys724-research-2026/egodex_frames_part1/`

**Bridge V2** (Robot demonstrations):
- Source: Robot manipulation trajectories
- Format: RLDS (TensorFlow datasets)
- Location (AWS): `/workspace/data/datasets/bridge_v2/`

### AWS vs Local Paths

**AWS EC2** (`/workspace/data/`):
```bash
# Training
/workspace/data/egodex/part1/
/workspace/data/checkpoints/two_stream/

# Datasets
/workspace/data/datasets/bridge_v2/
```

**Local Development** (`./data/`):
```bash
# Relative to project root
./data/checkpoints/
./data/egodex/test/
```

Scripts automatically detect the environment and use appropriate paths.

## S3 Sync

Upload checkpoints to S3 after training:

```bash
# Automatic (via train_long.py)
python scripts/train_long.py --s3-bucket bys724-research-2026 --s3-prefix checkpoints

# Manual
aws s3 sync data/checkpoints/two_stream/ s3://bys724-research-2026/checkpoints/two_stream/
```

Download pretrained checkpoints:

```bash
# From S3
aws s3 sync s3://bys724-research-2026/checkpoints/two_stream/ data/checkpoints/two_stream/

# From local
cp -r /path/to/checkpoints data/checkpoints/
```

## Disk Usage Estimates

| Item | Size | Notes |
|------|------|-------|
| EgoDex part1 | 336GB | Pre-extracted frames (JPEG) |
| EgoDex test | 17GB | Test set only |
| Bridge V2 | ~50GB | Robot demonstrations |
| Checkpoint | ~200MB | Per model (ViT-Base) |
| Training run | ~2GB | 12 checkpoints per run |

## .gitignore

All data files are excluded from git:

```gitignore
data/*
!data/README.md
!data/checkpoints/.gitkeep
!data/datasets/.gitkeep
```

This ensures large datasets and checkpoints are not committed to the repository.
