"""alpha_plan §4-c 작은 M 스모크 — python smoke_small_m.py <repo_root> <off|small> [out.pt].

off = C1-DN 설정 손실 (수정 전/후 코드 비교) · small = 192×2 M + P 동결 (C1-DN 로드 · grad 경로 점검 · ckpt 저장)."""
import sys, torch
root, mode = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
from src.models.two_stream_v15 import TwoStreamV15Model
kw = dict(embed_dim=384, num_heads=6, p_depth=12, m_depth=6, use_sobel=False, pair_mode=True, use_compose=False,
          comp_mae=True, routing_source="m", rotation_aug=True, m_recon_weight_floor=0.02,
          bright_aug=True, m_noise_max=0.01, bright_scene_gain_range=(0.73, 1.37), bright_target="aug")
if mode == "small":
    kw.update(m_depth=2, m_embed_dim=192, m_only=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); m = TwoStreamV15Model(**kw).to(dev).train()
if mode == "small":   # init-from 과 같은 규칙 (모양 다른 키 건너뜀)
    ck = torch.load("/proj/external_group/mrg/checkpoints/two_stream_v15b_refine_comp_s_denoise_augtgt/20260927_235906/checkpoint_epoch0010.pt", map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck["model_state_dict"].items()}; del ck
    own = m.state_dict(); skip = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
    miss, unexp = m.load_state_dict({k: v for k, v in sd.items() if k not in skip}, strict=False)
    print("skipped", len(skip), skip[:4]); print("missing", miss); print("unexpected (first)", unexp[:4], len(unexp))
    p0 = m.blocks_p[0].attn.qkv.weight.detach().clone() if hasattr(m.blocks_p[0], "attn") else None
torch.manual_seed(1); torch.cuda.manual_seed_all(1)
a = torch.rand(16, 3, 224, 224, device=dev); b = (a + 0.05 * torch.rand_like(a)).clamp(0, 1)
out = m(a, b); loss = out["loss"]
print(mode, "loss %.10f" % loss.item(), "loss_t", float(out["loss_t"]), "L_mB", float(out["loss_m_caseB"]), "L_mA", float(out["loss_m_caseA"]))
loss.backward()
if mode == "small":
    tr = [n for n, p in m.named_parameters() if p.requires_grad]
    nograd = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    leak = [n for n, p in m.named_parameters() if not p.requires_grad and p.grad is not None]
    print("trainable tensors", len(tr), "params %.2fM" % (sum(p.numel() for n, p in m.named_parameters() if p.requires_grad) / 1e6))
    print("trainable roots", sorted({n.split('.')[0] for n in tr}))
    print("trainable but no grad", nograd); print("frozen with grad", leak[:3])
    print("cls_m shape", tuple(out["cls_m"].shape), "m_features", tuple(out["m_features"].shape))
    torch.save({"model_state_dict": m.state_dict()}, sys.argv[3])
