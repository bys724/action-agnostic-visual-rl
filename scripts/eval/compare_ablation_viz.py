"""Ablation A/B/C 시각화 비교: 동일 입력에 대해 3모델의 예측을 나란히 비교."""
import sys, os, random, torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace")
from src.models import TwoStreamModel

DEVICE = "cuda"
FRAMES_DIR = "/mnt/data/egodex_frames/part4"
OUTPUT_DIR = "/workspace/docs/architecture/ablation_compare"
SEED = 42

CONFIGS = {
    "A (d=6,s=3)": {"ckpt": "/mnt/data/checkpoints/ablation_A/20260401_001722/best_model.pt", "depth": 6, "num_stages": 3},
    "B (d=6,s=2)": {"ckpt": "/mnt/data/checkpoints/ablation_B/20260401_001722/best_model.pt", "depth": 6, "num_stages": 2},
    "C (d=4,s=2)": {"ckpt": "/mnt/data/checkpoints/ablation_C/20260402_010408/best_model.pt", "depth": 4, "num_stages": 2},
}

GAPS = [1, 5, 10, 30]
NUM_SAMPLES = 3  # 샘플 수

def load_image(path, size=256):
    img = Image.open(path).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(img)).float() / 255.0

def center_crop(img, crop_size=224):
    h, w = img.shape[:2]
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return img[top:top+crop_size, left:left+crop_size]

def find_fixed_pairs(frames_dir, num_pairs, gaps, seed=42):
    """시드 고정으로 동일 에피소드/프레임에서 gap별 쌍 추출."""
    rng = random.Random(seed)
    episode_dirs = []
    for root, dirs, files in os.walk(frames_dir):
        if any(f.endswith('.jpg') for f in files):
            episode_dirs.append(root)

    rng.shuffle(episode_dirs)
    max_gap = max(gaps)

    results = []  # (ep_dir, frame_idx, ep_label)
    for ep_dir in episode_dirs:
        frames = sorted([f for f in os.listdir(ep_dir) if f.endswith('.jpg')])
        if len(frames) < max_gap + 1:
            continue
        idx = rng.randint(0, len(frames) - max_gap - 1)
        ep_label = os.path.relpath(ep_dir, frames_dir)
        results.append((ep_dir, idx, frames, ep_label))
        if len(results) >= num_pairs:
            break
    return results

def run_inference(model, img_t, img_tk):
    x = img_t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    y = img_tk.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_m, pred_p, _ = model(x, y)
    pred_m = pred_m.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
    pred_p = pred_p.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
    return pred_m, pred_p

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 모델 로드
    models = {}
    for name, cfg in CONFIGS.items():
        print(f"Loading {name}...")
        model = TwoStreamModel(depth=cfg["depth"], num_stages=cfg["num_stages"]).to(DEVICE)
        ck = torch.load(cfg["ckpt"], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        models[name] = model

    # 고정 샘플
    samples = find_fixed_pairs(FRAMES_DIR, NUM_SAMPLES, GAPS, SEED)
    print(f"Found {len(samples)} episodes")

    for gap in GAPS:
        print(f"\n=== Gap {gap} ===")
        # rows: samples, cols: frame_t | frame_t+k | A_pred_m | A_pred_p | B_pred_m | B_pred_p | C_pred_m | C_pred_p
        n_rows = len(samples)
        n_cols = 2 + len(CONFIGS) * 2  # input(2) + models * 2(pred_m, pred_p)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1:
            axes = [axes]

        col_titles = ["Frame t", f"Frame t+{gap}"]
        for name in CONFIGS:
            col_titles += [f"{name}\nPred M", f"{name}\nPred P"]

        for row, (ep_dir, idx, frames, ep_label) in enumerate(samples):
            img_t = load_image(os.path.join(ep_dir, frames[idx]))
            img_tk = load_image(os.path.join(ep_dir, frames[idx + gap]))
            img_t = center_crop(img_t)
            img_tk = center_crop(img_tk)

            # Column 0, 1: input
            axes[row][0].imshow(img_t.numpy())
            axes[row][1].imshow(img_tk.numpy())

            col = 2
            for name, model in models.items():
                pred_m, pred_p = run_inference(model, img_t, img_tk)
                mse_m = float(((pred_m - img_tk.numpy()) ** 2).mean())
                mse_p = float(((pred_p - img_tk.numpy()) ** 2).mean())
                axes[row][col].imshow(pred_m)
                axes[row][col].set_xlabel(f"MSE={mse_m:.4f}", fontsize=7)
                axes[row][col + 1].imshow(pred_p)
                axes[row][col + 1].set_xlabel(f"MSE={mse_p:.4f}", fontsize=7)
                col += 2

            axes[row][0].set_ylabel(f"{ep_label[:25]}", fontsize=7, rotation=0, labelpad=60, va='center')

        for col in range(n_cols):
            axes[0][col].set_title(col_titles[col], fontsize=9)

        for row in axes:
            for ax in row:
                ax.axis("off")

        plt.suptitle(f"Ablation Compare — Gap={gap}", fontsize=14, y=1.02)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"ablation_gap{gap}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
