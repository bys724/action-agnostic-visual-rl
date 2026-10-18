"""E1 잡음 묶음 스모크 — 기본값 경로가 수정 전 코드와 같은 손실인지 + 4종 묶음 forward/backward.

python smoke_e1.py <repo_root> <mode: off|dn|e1>  → 'mode loss <float>' 출력.
"""
import sys, torch
sys.path.insert(0, sys.argv[1])
from src.models.two_stream_v15 import TwoStreamV15Model
mode = sys.argv[2]
dev = "cuda" if torch.cuda.is_available() else "cpu"
kw = dict(embed_dim=384, num_heads=6, p_depth=12, m_depth=6, use_sobel=False, pair_mode=True, use_compose=False,
          comp_mae=True, routing_source="m", rotation_aug=True, m_recon_weight_floor=0.02)
if mode != "off":   # C1-DN 설정
    kw.update(bright_aug=True, m_noise_max=0.01, bright_scene_gain_range=(0.73, 1.37), bright_target="aug")
if mode == "e1":
    kw["m_noise_kinds"] = ("gaussian", "shot", "read", "jpeg")
torch.manual_seed(0)
m = TwoStreamV15Model(**kw).to(dev).train()
torch.manual_seed(1); torch.cuda.manual_seed_all(1)
a = torch.rand(16, 3, 224, 224, device=dev); b = (a + 0.05 * torch.rand_like(a)).clamp(0, 1)
out = m(a, b)
loss = out["loss"]
print(mode, "loss %.10f" % loss.item(), {k: round(float(v), 5) for k, v in out.items() if "bright" in k})
loss.backward(); print(mode, "backward ok")
if mode == "e1":   # 종류별 잡음 세기 (휘도 RMS) — 묶음 각 종류가 실제로 작동하는지
    x = a[:4]; sig = torch.full((4, 1, 1, 1), 0.01, device=dev)
    L = lambda im: 0.299 * im[:, 0] + 0.587 * im[:, 1] + 0.114 * im[:, 2]
    for i, k in enumerate(m.m_noise_kinds):
        y = m._sensor_noise(x, sig, torch.full((4,), i, device=dev))
        print("kind", k, "L rms %.5f" % (L(y) - L(x)).pow(2).mean().sqrt().item())
