"""
Two-Stream 모델 추론 시각화.

단일/다중 모델 비교 모두 지원. 시드 고정 + gap 고정으로 공정 비교 가능.

Usage (단일 모델):
    python scripts/eval/visualize_inference.py \
        --checkpoint /path/to/model.pt \
        --frames-dir /mnt/data/droid_frames/ext1 \
        --gap 30 --seed 42 --num-samples 4

Usage (여러 모델 비교):
    python scripts/eval/visualize_inference.py \
        --checkpoint sg.pt grad.pt \
        --label sg grad \
        --frames-dir /mnt/data/droid_frames/ext1 \
        --gap 30 --seed 42 --num-samples 4
"""

import argparse
import random
import sys
import os

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import TwoStreamModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(path, size=256):
    img = Image.open(path).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, 3]


def random_crop_pair(img_t, img_tk, crop_size=224):
    """학습과 동일하게 두 프레임에 독립적 RandomCrop 적용."""
    from torchvision import transforms
    # [H,W,3] → [3,H,W]
    t = img_t.permute(2, 0, 1)
    tk = img_tk.permute(2, 0, 1)
    params1 = transforms.RandomCrop.get_params(t, (crop_size, crop_size))
    params2 = transforms.RandomCrop.get_params(tk, (crop_size, crop_size))
    t = transforms.functional.crop(t, *params1).permute(1, 2, 0)
    tk = transforms.functional.crop(tk, *params2).permute(1, 2, 0)
    return t, tk


def find_frame_pairs(frames_dir, num_pairs=4, gap=None, max_gap=10, seed=None):
    """프레임 디렉토리에서 에피소드의 프레임 쌍 추출.

    Args:
        gap: 고정 gap (None이면 1~max_gap에서 랜덤)
        seed: 랜덤 시드 (지정 시 동일 샘플 보장)
    """
    rng = random.Random(seed) if seed is not None else random

    episode_dirs = []
    for root, dirs, files in os.walk(frames_dir):
        if any(f.endswith('.jpg') for f in files):
            episode_dirs.append(root)

    if not episode_dirs:
        raise ValueError(f"No frame directories found in {frames_dir}")

    episode_dirs.sort()  # 정렬 후 셔플하여 시드 효과 보장
    rng.shuffle(episode_dirs)

    pairs = []
    min_len = (gap if gap else max_gap) + 1
    for ep_dir in episode_dirs:
        frames = sorted([f for f in os.listdir(ep_dir) if f.endswith('.jpg')])
        if len(frames) < min_len:
            continue
        actual_gap = gap if gap else rng.randint(1, max_gap)
        idx = rng.randint(0, len(frames) - actual_gap - 1)
        ep_label = os.path.relpath(ep_dir, frames_dir)
        pairs.append((
            os.path.join(ep_dir, frames[idx]),
            os.path.join(ep_dir, frames[idx + actual_gap]),
            ep_label, actual_gap
        ))
        if len(pairs) >= num_pairs:
            break

    return pairs


def run_inference(model, img_t, img_tk):
    """[H,W,3] 텐서 2장 → (pred_m, pred_p) [H,W,3] numpy."""
    x = img_t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)   # [1,3,H,W]
    y = img_tk.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_m, pred_p, _ = model(x, y)
    pred_m = pred_m.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
    pred_p = pred_p.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
    return pred_m, pred_p


def visualize_multi(rows_data, model_labels, output_path, gap):
    """여러 모델의 동일 입력 비교 시각화.

    rows_data: list of dicts, each {
        'img_t': [H,W,3],
        'img_tk': [H,W,3],
        'predictions': [(pred_m, pred_p, mse_m, mse_p), ...] per model
        'ep': str
    }
    model_labels: list of str
    """
    n_rows = len(rows_data)
    n_models = len(model_labels)
    n_cols = 2 + n_models * 2  # frame_t + frame_tk + (pred_m, pred_p) per model

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]

    col_titles = ["Frame t", f"Frame t+{gap}"]
    for label in model_labels:
        col_titles += [f"{label}\nPred M", f"{label}\nPred P"]

    for row, data in enumerate(rows_data):
        axes[row][0].imshow(data['img_t'])
        axes[row][1].imshow(data['img_tk'])

        for i, (pred_m, pred_p, mse_m, mse_p) in enumerate(data['predictions']):
            col_m = 2 + i * 2
            col_p = 3 + i * 2
            axes[row][col_m].imshow(pred_m)
            axes[row][col_m].set_xlabel(f"MSE={mse_m:.4f}", fontsize=7)
            axes[row][col_p].imshow(pred_p)
            axes[row][col_p].set_xlabel(f"MSE={mse_p:.4f}", fontsize=7)

        axes[row][0].set_ylabel(data['ep'][:20], fontsize=8, rotation=0, labelpad=50, va='center')
        for col in range(n_cols):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    for col in range(n_cols):
        axes[0][col].set_title(col_titles[col], fontsize=10)

    plt.suptitle(f"Two-Stream Inference Compare — gap={gap}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs='+', required=True,
                        help="하나 이상의 체크포인트 경로")
    parser.add_argument("--label", nargs='+', default=None,
                        help="각 체크포인트의 라벨 (없으면 파일명 사용)")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output", default="/tmp/inference_result.png")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--gap", type=int, default=None,
                        help="고정 gap (없으면 1~max_gap에서 랜덤)")
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None,
                        help="시드 (여러 모델 비교 시 동일 샘플 보장)")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.0)
    parser.add_argument("--mask-ratio-p", type=float, default=None)
    args = parser.parse_args()

    # 라벨
    if args.label is None:
        labels = [os.path.basename(os.path.dirname(c)) or f"model{i}" for i, c in enumerate(args.checkpoint)]
    else:
        assert len(args.label) == len(args.checkpoint)
        labels = args.label

    # 모델들 로드
    models = []
    for ckpt_path, label in zip(args.checkpoint, labels):
        print(f"Loading: {label} ← {ckpt_path}")
        model = TwoStreamModel(depth=args.depth, num_stages=args.num_stages,
                               mask_ratio=args.mask_ratio, mask_ratio_p=args.mask_ratio_p).to(DEVICE)
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.eval()
        models.append(model)
        print(f"  epoch {ck.get('epoch','?')}")

    # 동일 시드로 프레임 쌍 선정
    print(f"\nSampling from {args.frames_dir} (gap={args.gap}, seed={args.seed})")
    pairs = find_frame_pairs(args.frames_dir, args.num_samples, gap=args.gap,
                             max_gap=args.max_gap, seed=args.seed)
    print(f"  Found {len(pairs)} pairs")

    # 추론
    rows_data = []
    for path_t, path_tk, ep, gap in pairs:
        img_t = load_image(path_t)
        img_tk = load_image(path_tk)
        # 동일 비교를 위해 center crop (랜덤 crop은 모델마다 결과 달라짐)
        from torchvision import transforms
        t = img_t.permute(2, 0, 1)
        tk = img_tk.permute(2, 0, 1)
        t = transforms.functional.center_crop(t, 224).permute(1, 2, 0)
        tk = transforms.functional.center_crop(tk, 224).permute(1, 2, 0)

        predictions = []
        for label, model in zip(labels, models):
            pred_m, pred_p = run_inference(model, t, tk)
            mse_m = float(((pred_m - tk.numpy()) ** 2).mean())
            mse_p = float(((pred_p - tk.numpy()) ** 2).mean())
            predictions.append((pred_m, pred_p, mse_m, mse_p))
            print(f"  [{label}] {ep[:30]} gap={gap} M={mse_m:.4f} P={mse_p:.4f}")

        rows_data.append({
            'img_t': t.numpy(),
            'img_tk': tk.numpy(),
            'predictions': predictions,
            'ep': ep,
        })

    # 사용할 gap 표기 (모두 같으면 그것, 아니면 'mixed')
    gap_label = args.gap if args.gap else 'mixed'
    visualize_multi(rows_data, labels, args.output, gap_label)


if __name__ == "__main__":
    main()
