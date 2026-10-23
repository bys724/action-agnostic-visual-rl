# §4-c 사다리 후보 M 인코더 forward FLOPs (배치 1 · FlopCounterMode · observer_fuse.py flops 와 같은 조건)
import sys, json, gc, torch
sys.path.insert(0, "/proj/home/mrg/bys724/action-agnostic-visual-rl")
from torch.utils.flop_counter import FlopCounterMode
from src.models.two_stream_v15 import TwoStreamV15Model
def fl(fn):
    with FlopCounterMode(display=False) as fc: fn()
    return fc.get_total_flops() / 1e9
x = torch.rand(1, 3, 224, 224); out = {}
for d, L in [(384, 6), (384, 3), (384, 2), (384, 1), (256, 3), (256, 2), (192, 6), (192, 4), (192, 3), (192, 2), (192, 1), (128, 3), (128, 2)]:
    m = TwoStreamV15Model(embed_dim=d, num_heads=max(d // 64, 1), m_depth=L, comp_mae=True, pair_mode=True,
                          use_sobel=False, masked_anchor=True).eval()
    r = {"M_enc": fl(lambda: m._encode_m_unmasked(m.preprocessing.compute_m_channel(x, x))),
         "M_params_M": sum(p.numel() for n, p in m.named_parameters() if n.startswith(("blocks_m", "patch_embed_m", "norm_m", "pos_embed_m", "cls_token_m"))) / 1e6}
    if (d, L) == (384, 6): r["P_enc"] = fl(lambda: m._encode_p_unmasked(m.preprocessing.compute_p_channel(x)))
    out[f"{d}x{L}"] = r; print(d, L, {k: round(v, 3) for k, v in r.items()}, flush=True)
    del m; gc.collect()
json.dump(out, open(sys.argv[1], "w"), indent=1)
