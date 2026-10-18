"""E1 잡음 종류별 소요 시간 (배치 128, 학습 1스텝당 nz 6회 호출 기준) + nvjpeg GPU 인코딩 가능 여부."""
import sys, time, torch, torchvision
sys.path.insert(0, sys.argv[1])
from src.models.two_stream_v15 import TwoStreamV15Model
dev = "cuda"
m = TwoStreamV15Model(embed_dim=96, num_heads=3, p_depth=1, m_depth=1, use_sobel=False, pair_mode=True,
                      use_compose=False, comp_mae=True, routing_source="m", bright_aug=True, m_noise_max=0.01,
                      m_noise_kinds=("gaussian", "shot", "read", "jpeg")).to(dev)
x = torch.rand(128, 3, 224, 224, device=dev); sig = torch.full((128, 1, 1, 1), 0.01, device=dev)
m._sensor_noise(x[:8], sig[:8], torch.full((8,), 3, device=dev))  # nvjpeg 예열
for i, k in enumerate(m.m_noise_kinds):
    kind = torch.full((128,), i, device=dev)
    m._sensor_noise(x[:8], sig[:8], kind[:8]); torch.cuda.synchronize(); t = time.time()
    for _ in range(5): m._sensor_noise(x, sig, kind)
    torch.cuda.synchronize(); print(f"kind {k}: {(time.time()-t)/5*1000:.1f} ms / 128 samples", flush=True)
kind = torch.randint(0, 4, (128,), device=dev); torch.cuda.synchronize(); t = time.time()
for _ in range(5): m._sensor_noise(x, sig, kind)
torch.cuda.synchronize(); per = (time.time()-t)/5; print(f"mixed: {per*1000:.1f} ms/call → step(6 calls) {per*6*1000:.0f} ms", flush=True)
try:
    u8 = (x[:32] * 255).to(torch.uint8)
    torch.cuda.synchronize(); t = time.time()
    enc = torchvision.io.encode_jpeg(list(u8), quality=90)
    dec = torchvision.io.decode_jpeg([e.cpu() for e in enc], device=dev); torch.cuda.synchronize()
    print(f"nvjpeg GPU encode OK: {(time.time()-t)*1000:.1f} ms / 32 imgs")
except Exception as e:
    print("nvjpeg GPU encode FAIL:", e)
# 예열 후 nvjpeg 반복 측정 (32장 × 5회)
try:
    u8 = (x[:32] * 255).to(torch.uint8)
    for _ in range(2):
        torchvision.io.decode_jpeg([e.cpu() for e in torchvision.io.encode_jpeg(list(u8), quality=90)], device=dev)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(5):
        enc = torchvision.io.encode_jpeg(list(u8), quality=90)
        dec = torchvision.io.decode_jpeg([e.cpu() for e in enc], device=dev)
    torch.cuda.synchronize(); print(f"nvjpeg warm: {(time.time()-t)/5*1000:.1f} ms / 32 imgs")
except Exception as e:
    print("nvjpeg warm FAIL:", e)
