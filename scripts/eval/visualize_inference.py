"""
Two-Stream 모델 추론 시각화.

임의의 프레임 쌍을 입력으로 예측 이미지 생성.
Usage:
    python scripts/eval/visualize_inference.py \
        --checkpoint /tmp/ts_latest.pt \
        --frames-dir /mnt/data/droid_frames/ext1 \
        --output /tmp/inference_result.png \
        --num-samples 4
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


def find_frame_pairs(frames_dir, num_pairs=4, max_gap=10):
    """프레임 디렉토리에서 랜덤 에피소드의 프레임 쌍 추출.

    Supports both flat (episode_dir/frame.jpg) and nested (task/episode/frame.jpg) layouts.
    """
    # Collect all episode directories (dirs that contain .jpg files)
    episode_dirs = []
    for root, dirs, files in os.walk(frames_dir):
        if any(f.endswith('.jpg') for f in files):
            episode_dirs.append(root)

    if not episode_dirs:
        raise ValueError(f"No frame directories found in {frames_dir}")

    pairs = []
    random.shuffle(episode_dirs)
    for ep_dir in episode_dirs:
        frames = sorted([f for f in os.listdir(ep_dir) if f.endswith('.jpg')])
        if len(frames) < max_gap + 1:
            continue
        idx = random.randint(0, len(frames) - max_gap - 1)
        gap = random.randint(1, max_gap)
        # Use relative path from frames_dir as label
        ep_label = os.path.relpath(ep_dir, frames_dir)
        pairs.append((
            os.path.join(ep_dir, frames[idx]),
            os.path.join(ep_dir, frames[idx + gap]),
            ep_label, gap
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


def visualize(pairs_data, output_path):
    """pairs_data: list of (img_t, img_tk, pred_m, pred_p, title)"""
    n = len(pairs_data)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    titles = ["Frame t (input)", "Frame t+k (target)", "Pred M (change)", "Pred P (appear)"]
    for row, (img_t, img_tk, pred_m, pred_p, title) in enumerate(pairs_data):
        for col, (img, label) in enumerate(zip([img_t, img_tk, pred_m, pred_p], titles)):
            axes[row][col].imshow(img)
            axes[row][col].set_title(label if row == 0 else "")
            axes[row][col].axis("off")
        axes[row][0].set_ylabel(title, fontsize=9, rotation=0, labelpad=80, va='center')

    plt.suptitle("Two-Stream: Per-stream Future Prediction", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output", default="/tmp/inference_result.png")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.0)
    parser.add_argument("--mask-ratio-p", type=float, default=None)
    args = parser.parse_args()

    # 모델 로드
    print(f"Loading checkpoint: {args.checkpoint}")
    model = TwoStreamModel(depth=args.depth, num_stages=args.num_stages,
                           mask_ratio=args.mask_ratio, mask_ratio_p=args.mask_ratio_p).to(DEVICE)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"  Loaded (epoch {ck.get('epoch','?')}, loss {ck.get('train_loss', '?'):.4f})")

    # 프레임 쌍 탐색
    print(f"Sampling frames from: {args.frames_dir}")
    pairs = find_frame_pairs(args.frames_dir, args.num_samples, args.max_gap)
    print(f"  Found {len(pairs)} pairs")

    # 추론
    pairs_data = []
    for path_t, path_tk, ep, gap in pairs:
        img_t = load_image(path_t)   # 256x256
        img_tk = load_image(path_tk)
        img_t, img_tk = random_crop_pair(img_t, img_tk)  # 독립적 224x224 crop
        pred_m, pred_p = run_inference(model, img_t, img_tk)

        mse_m = float(((pred_m - img_tk.numpy()) ** 2).mean())
        mse_p = float(((pred_p - img_tk.numpy()) ** 2).mean())
        print(f"  [{ep}] gap={gap}, MSE_M={mse_m:.4f}, MSE_P={mse_p:.4f}")

        pairs_data.append((
            img_t.numpy(),
            img_tk.numpy(),
            pred_m,
            pred_p,
            f"ep: {ep[:20]}  gap={gap}  M={mse_m:.4f} P={mse_p:.4f}"
        ))

    visualize(pairs_data, args.output)


if __name__ == "__main__":
    main()
