"""하나의 에피소드에서 gap을 변화시키며 Two-Stream 예측 비교."""
import sys, os, random
sys.path.insert(0, '/workspace')

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

from src.models import TwoStreamModel

DEVICE = "cuda:0"
CHECKPOINT = sys.argv[1]
FRAMES_DIR = sys.argv[2]
OUTPUT = sys.argv[3]
GAPS = [1, 3, 5, 10, 15, 30, 45, 60]

# 모델 로드
model = TwoStreamModel().to(DEVICE)
ck = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ck["model_state_dict"])
model.eval()
print(f"Loaded epoch {ck.get('epoch')}, loss {ck.get('train_loss', '?'):.4f}")

# 에피소드 선택 (프레임 수가 충분한 것)
episodes = []
for root, dirs, files in os.walk(FRAMES_DIR):
    jpgs = sorted([f for f in files if f.endswith('.jpg')])
    if len(jpgs) > max(GAPS) + 5:
        episodes.append((root, jpgs))

if len(sys.argv) > 4:
    target = sys.argv[4]
    episodes = [(d, f) for d, f in episodes if target in d]
random.shuffle(episodes)
ep_dir, frames = episodes[0]
ep_name = os.path.relpath(ep_dir, FRAMES_DIR)
print(f"Episode: {ep_name} ({len(frames)} frames)")

# 시작 프레임 고정
start_idx = len(frames) // 4  # 1/4 지점에서 시작

def load_and_crop(path, crop_params):
    img = Image.open(path).convert("RGB").resize((256, 256))
    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    t = transforms.functional.crop(t, *crop_params)
    return t

# 시작 프레임의 crop 파라미터 고정
dummy = Image.open(os.path.join(ep_dir, frames[start_idx])).convert("RGB").resize((256, 256))
dummy_t = torch.from_numpy(np.array(dummy)).float().permute(2, 0, 1) / 255.0
crop_params_t = transforms.RandomCrop.get_params(dummy_t, (224, 224))

rows = []
for gap in GAPS:
    future_idx = start_idx + gap
    if future_idx >= len(frames):
        continue

    path_t = os.path.join(ep_dir, frames[start_idx])
    path_tk = os.path.join(ep_dir, frames[future_idx])

    img_t = load_and_crop(path_t, crop_params_t)
    # future는 독립 crop
    crop_params_tk = transforms.RandomCrop.get_params(dummy_t, (224, 224))
    img_tk = load_and_crop(path_tk, crop_params_tk)

    with torch.no_grad():
        x = img_t.unsqueeze(0).to(DEVICE)
        y = img_tk.unsqueeze(0).to(DEVICE)
        pred_m, pred_p, _ = model(x, y)

    pred_m = pred_m.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
    pred_p = pred_p.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
    img_t_np = img_t.permute(1, 2, 0).numpy()
    img_tk_np = img_tk.permute(1, 2, 0).numpy()

    rows.append((gap, img_t_np, img_tk_np, pred_m, pred_p))

# Plot
nrows = len(rows)
fig, axes = plt.subplots(nrows, 4, figsize=(16, 4 * nrows))
if nrows == 1:
    axes = [axes]

col_titles = ['Frame t', 'Frame t+k (target)',
              'Pred M (ΔL + Sobel(ΔL))', 'Pred P (Sobel + RGB)']
for row_idx, (gap, img_t, img_tk, pred_m, pred_p) in enumerate(rows):
    for col_idx, img in enumerate([img_t, img_tk, pred_m, pred_p]):
        axes[row_idx][col_idx].imshow(img)
        axes[row_idx][col_idx].axis('off')
        if row_idx == 0:
            axes[row_idx][col_idx].set_title(col_titles[col_idx], fontsize=11)
    axes[row_idx][0].set_ylabel(f'gap={gap}', fontsize=11, fontweight='bold',
                                 rotation=0, labelpad=50, va='center')

ckpt_name = os.path.basename(CHECKPOINT)
fig.suptitle(f'Two-Stream Gap Sweep — {ckpt_name}\n{ep_name}', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT}")
