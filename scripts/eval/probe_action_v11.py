#!/usr/bin/env python
"""
Action Probing for Two-Stream v11 (Motion-Guided Routing + Dual-Target).

v11 모델 (`TwoStreamV11Model`)은 v6/v10용 `probe_action.py`의
`TwoStreamEncoder` 가정과 호환되지 않아 별도 스크립트가 필요.

Representation 옵션 (사용자 명세 ablation):
    [Encoder CLS]
    - cls_m_enc                       : M encoder CLS                       [B,  768]
    - cls_p_enc                       : P encoder CLS                       [B,  768]
    - cls_concat_enc                  : M CLS ⊕ P CLS                       [B, 1536]

    [Encoder patch (decoder 일체 X — 표준 MAE baseline)]
    - patch_mean_m_enc                : M encoder patches mean              [B,  768]
    - patch_mean_p_enc                : P encoder patches mean              [B,  768]
    - patch_mean_concat_enc_only      : M enc patches ⊕ P enc patches       [B, 1536]

    [P decoder를 거친 representation (M encoder와 concat)]
    - patch_mean_p_state_after_routing: D' (motion-routing × 2 후, interp_2 전) [B, 768]
    - patch_mean_p_features_tk        : D  (interp_2 후, Phase 3 final)      [B, 768]
    - patch_mean_concat_enc_d_prime   : M enc patches ⊕ D'                   [B, 1536]
    - patch_mean_concat_enc_phase3    : M enc patches ⊕ D                    [B, 1536]

Deterministic probing 보장:
    학습용 forward는 random mask가 들어가므로, **mask_ratio=0**으로 v11 모델
    instance를 새로 만든 뒤 ckpt를 로드. mask 0이면 모든 patches가 visible →
    `_random_mask` 결과가 항상 all-False → eval 결과 결정론적.
    rotation_aug는 model.eval()로 자동 비활성.

D' 추출:
    `TwoStreamV11Model.forward()`는 D' (motion-routing 후 state)를 dict로
    노출하지 않음. 모델 수정 금지(RUNNING 잡 영향 회피)이므로 v11 forward의
    P decoder 단계를 step-by-step 재현 (`_full_forward_with_d_prime`).

Usage:
    python scripts/eval/probe_action_v11.py \\
        --encoder two-stream-v11 \\
        --checkpoint /proj/.../two_stream_v11/<ts>/checkpoint_epoch0004.pt \\
        --egodex-root /proj/external_group/mrg/datasets/egodex/raw \\
        --frames-root /proj/external_group/mrg/datasets/egodex/frames \\
        --egodex-split test \\
        --cls-mode patch_mean_p_state_after_routing \\
        --gap 10
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Reuse EgoDex dataset, probes, and metrics from probe_action.py
_project_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _project_root)
sys.path.insert(0, "/workspace")

from scripts.eval.probe_action import (  # noqa: E402
    LinearProbe,
    MLPProbe,
    build_datasets,
    train_probe,
)


# ============================================================================
# v11 representation modes
# ============================================================================

CLS_MODES_SINGLE = {
    # single-image (encode_single 또는 P encoder만) 가능 모드
    "cls_p_enc",
    "patch_mean_p_enc",
    "patch_mean_p_enc_tk",  # P encoder를 img_tk(next)에 통과
}
CLS_MODES_PAIRED = {
    # M encoder는 ΔL = pixel diff(img_t, img_tk)이 필요 → paired
    "cls_m_enc",
    "patch_mean_m_enc",
    "cls_concat_enc",
    "patch_mean_concat_enc_only",
    # P decoder phase는 mask token inject + motion routing(M dependency) 필요
    "patch_mean_p_state_after_routing",
    "patch_mean_p_features_tk",
    "patch_mean_concat_enc_d_prime",
    "patch_mean_concat_enc_phase3",
    # B+D', A+B+D' (representation 조합 ablation)
    "patch_mean_concat_p_enc_d_prime",
    "patch_mean_concat_all",
    # D' 위치 CLS (motion-routing 후 CLS) — v13 분석용
    "cls_p_state_after_routing",
    # D 위치 CLS (interpreter_2 후 CLS = predicted_cls_tk) — v13 디자인 핵심 위치
    "cls_p_features_tk",
    # B+D (P enc + interpreter_2 후) — v13 디자인 충실 concat
    "patch_mean_concat_p_enc_phase3",
    # P_t + P_tk concat (P encoder 두 frame 비교용)
    "patch_mean_concat_p_t_p_tk",
    # P_tk + M concat (next-frame P + motion)
    "patch_mean_concat_p_tk_m",
}
CLS_MODES_ALL = CLS_MODES_SINGLE | CLS_MODES_PAIRED

_CONCAT_2_MODES = {
    "cls_concat_enc",
    "patch_mean_concat_enc_only",
    "patch_mean_concat_enc_d_prime",
    "patch_mean_concat_enc_phase3",
    "patch_mean_concat_p_enc_d_prime",
    "patch_mean_concat_p_enc_phase3",
    "patch_mean_concat_p_t_p_tk",
    "patch_mean_concat_p_tk_m",
}
_CONCAT_3_MODES = {
    "patch_mean_concat_all",  # M_enc + P_enc + D'
}


def _embed_dim(cls_mode: str, base_dim: int = 768) -> int:
    if cls_mode in _CONCAT_3_MODES:
        return base_dim * 3
    if cls_mode in _CONCAT_2_MODES:
        return base_dim * 2
    return base_dim


# ============================================================================
# Model loading
# ============================================================================

def load_v11_model(
    checkpoint_path: str,
    p_depth: int = 12,
    m_depth: int = 6,
    device: str = "cuda",
):
    """Instantiate v11 with mask_ratio=0 (deterministic) and load weights."""
    from src.models import TwoStreamV11Model

    # mask_ratio=0 → 모든 patches visible → deterministic
    # rotation_aug=False → eval에서 어차피 비활성, 명시적으로 끔
    model = TwoStreamV11Model(
        embed_dim=768,
        p_depth=p_depth,
        m_depth=m_depth,
        num_heads=12,
        mlp_ratio=4.0,
        image_size=224,
        patch_size=16,
        mask_ratio_m=0.0,
        mask_ratio_p=0.0,
        decoder_depth_m=3,
        interpreter_depth=3,
        num_motion_iters=2,
        rotation_aug=False,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # DDP prefix strip
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 3: {missing[:3]})")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 3: {unexpected[:3]})")

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ============================================================================
# Step-by-step forward 재현 (D' 추출용)
# ============================================================================
#
# v11 forward 단계를 정확히 매칭:
#   1. preprocessing(img_t, img_tk) → m_channel(3ch), p_channel(5ch)
#   2. M encoder: patch_embed_m → +cls_token_m → +pos_embed_m
#                 → blocks_m × m_depth → norm_m
#      (mask=0 instance라 _encode_stream_visible의 visible 분기는 patches 전체 통과)
#   3. P encoder: patch_embed_p → +cls_token_p → +pos_embed_p
#                 → blocks_p × p_depth → norm_p
#   4. M decoder: _inject_mask_tokens (mask 0 → 효과 없음)
#                 → +dec_pos_embed_m → m_decoder_blocks × 3 → m_decoder_norm
#   5. P decoder Phase 1: _inject_mask_tokens (mask 0 → 효과 없음)
#                         → +dec_pos_embed_p → interpreter_1 × 3 → interpreter_1_norm
#                         → p_semantic_t  (= C, ablation에서 사용 안 함)
#   6. P decoder Phase 2: motion_routing × num_motion_iters
#                         → p_state  (= D')
#   7. P decoder Phase 3: interpreter_2 × 3 → interpreter_2_norm
#                         → p_semantic_tk  (= D)
# ============================================================================

@torch.no_grad()
def _m_encoder_forward(model, m_channel: torch.Tensor) -> torch.Tensor:
    """M encoder full-sequence (no mask). Returns [B, 1+N, D] post-norm."""
    B = m_channel.shape[0]
    m_patches = model.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
    m_cls = model.cls_token_m.expand(B, -1, -1)
    m_tokens = torch.cat([m_cls, m_patches], dim=1) + model.pos_embed_m
    for block in model.blocks_m:
        m_tokens = block(m_tokens, freqs_cis=None)
    return model.norm_m(m_tokens)


@torch.no_grad()
def _p_encoder_forward(model, p_channel: torch.Tensor) -> torch.Tensor:
    """P encoder full-sequence (no mask). Returns [B, 1+N, D] post-norm."""
    B = p_channel.shape[0]
    p_patches = model.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
    p_cls = model.cls_token_p.expand(B, -1, -1)
    p_tokens = torch.cat([p_cls, p_patches], dim=1) + model.pos_embed_p
    for block in model.blocks_p:
        p_tokens = block(p_tokens, freqs_cis=None)
    return model.norm_p(p_tokens)


@torch.no_grad()
def _full_forward_with_d_prime(model, img_t: torch.Tensor, img_tk: torch.Tensor) -> dict:
    """v11 forward를 step-by-step 재현. D' (motion-routing 후, interp_2 전) 노출.

    학습된 forward와 동일 동작 (mask=0 instance라 deterministic).
    src/models/two_stream_v11.py의 forward() 단계와 1:1 대응.

    Returns:
        m_encoded:        [B, 1+N, D]  M encoder out
        p_encoded:        [B, 1+N, D]  P encoder out
        m_completed:      [B, 1+N, D]  M decoder out
        p_semantic_t:     [B, 1+N, D]  Phase 1 out (C, ablation에서 사용 X)
        p_state_routing:  [B, 1+N, D]  Phase 2 out (D' — motion-routing 후, interp_2 전)
        p_semantic_tk:    [B, 1+N, D]  Phase 3 out (D)
    """
    # Step 1: preprocessing
    m_channel, p_channel = model.preprocessing(img_t, img_tk)
    B = img_t.shape[0]
    device = img_t.device

    # Step 2-3: encoders
    m_encoded = _m_encoder_forward(model, m_channel)
    p_encoded = _p_encoder_forward(model, p_channel)

    # mask=0 → all-False mask (forward와 동일)
    mask_zero = torch.zeros(B, model.num_patches, dtype=torch.bool, device=device)

    # Step 4: M decoder
    # _inject_mask_tokens: mask 0이라 visible == 전체 patches → full로 그대로 복원
    m_full = model._inject_mask_tokens(m_encoded, mask_zero, model.mask_token_m)
    m_full = m_full + model.dec_pos_embed_m
    for block in model.m_decoder_blocks:
        m_full = block(m_full, freqs_cis=None)
    m_completed = model.m_decoder_norm(m_full)

    # Step 5: P decoder Phase 1 (interpreter_1)
    p_full = model._inject_mask_tokens(p_encoded, mask_zero, model.mask_token_p)
    p_full = p_full + model.dec_pos_embed_p
    p_semantic_t = model._run_interpreter(
        p_full, model.interpreter_1, model.interpreter_1_norm,
    )

    # Step 6: P decoder Phase 2 (motion-routing × N) — D'
    p_state = p_semantic_t
    for routing_block in model.motion_routing:
        p_state = routing_block(p_state, m_completed)
    # ★ p_state = D' (interpreter_2 전)

    # Step 7: P decoder Phase 3 (interpreter_2) — D
    p_semantic_tk = model._run_interpreter(
        p_state, model.interpreter_2, model.interpreter_2_norm,
    )

    return {
        "m_encoded": m_encoded,
        "p_encoded": p_encoded,
        "m_completed": m_completed,
        "p_semantic_t": p_semantic_t,
        "p_state_routing": p_state,
        "p_semantic_tk": p_semantic_tk,
    }


# ============================================================================
# Representation extraction
# ============================================================================

@torch.no_grad()
def extract_repr(model, pixel_values: torch.Tensor, mode: str) -> torch.Tensor:
    """Extract probing representation from v11 model.

    Args:
        pixel_values: [B, 6, H, W] (img_t ⊕ img_t+gap)
        mode: one of CLS_MODES_ALL
    Returns:
        embedding: [B, D]
    """
    img_t = pixel_values[:, :3]
    img_tk = pixel_values[:, 3:]

    # ── Single-image path (M 정보 불필요) ─────────────────────────────────
    if mode in CLS_MODES_SINGLE:
        # P channel만 필요. preprocessing(img, img) → ΔL=0이지만 P channel은
        # Sobel(L) + RGB이므로 정상. img_tk variant는 next frame을 P encoder에 통과.
        src_img = img_tk if mode == "patch_mean_p_enc_tk" else img_t
        _, p_channel = model.preprocessing(src_img, src_img)
        p_encoded = _p_encoder_forward(model, p_channel)  # [B, 1+N, D]
        if mode == "cls_p_enc":
            return p_encoded[:, 0]
        if mode in {"patch_mean_p_enc", "patch_mean_p_enc_tk"}:
            return p_encoded[:, 1:].mean(dim=1)
        raise ValueError(f"Unhandled single mode: {mode}")

    # ── P_t + P_tk concat (P encoder 두 frame 모두 통과) ──────────────────
    if mode == "patch_mean_concat_p_t_p_tk":
        _, p_channel_t = model.preprocessing(img_t, img_t)
        _, p_channel_tk = model.preprocessing(img_tk, img_tk)
        p_enc_t = _p_encoder_forward(model, p_channel_t)
        p_enc_tk = _p_encoder_forward(model, p_channel_tk)
        return torch.cat(
            [p_enc_t[:, 1:].mean(dim=1), p_enc_tk[:, 1:].mean(dim=1)],
            dim=-1,
        )

    # ── P_tk + M concat ───────────────────────────────────────────────────
    if mode == "patch_mean_concat_p_tk_m":
        m_channel, _ = model.preprocessing(img_t, img_tk)
        _, p_channel_tk = model.preprocessing(img_tk, img_tk)
        m_encoded = _m_encoder_forward(model, m_channel)
        p_enc_tk = _p_encoder_forward(model, p_channel_tk)
        return torch.cat(
            [p_enc_tk[:, 1:].mean(dim=1), m_encoded[:, 1:].mean(dim=1)],
            dim=-1,
        )

    # ── M encoder만 필요한 paired modes ───────────────────────────────────
    if mode in {"cls_m_enc", "patch_mean_m_enc"}:
        m_channel, _ = model.preprocessing(img_t, img_tk)
        m_encoded = _m_encoder_forward(model, m_channel)  # [B, 1+N, D]
        if mode == "cls_m_enc":
            return m_encoded[:, 0]
        return m_encoded[:, 1:].mean(dim=1)

    # ── M+P encoder만 (decoder 일체 X — 표준 MAE baseline) ────────────────
    if mode in {"cls_concat_enc", "patch_mean_concat_enc_only"}:
        m_channel, p_channel = model.preprocessing(img_t, img_tk)
        m_encoded = _m_encoder_forward(model, m_channel)
        p_encoded = _p_encoder_forward(model, p_channel)
        if mode == "cls_concat_enc":
            return torch.cat([m_encoded[:, 0], p_encoded[:, 0]], dim=-1)
        # patch_mean_concat_enc_only
        return torch.cat(
            [m_encoded[:, 1:].mean(dim=1), p_encoded[:, 1:].mean(dim=1)],
            dim=-1,
        )

    # ── P decoder까지 가는 modes (D / D' 또는 그 concat) ──────────────────
    out = _full_forward_with_d_prime(model, img_t, img_tk)
    if mode == "patch_mean_p_state_after_routing":
        return out["p_state_routing"][:, 1:].mean(dim=1)
    if mode == "cls_p_state_after_routing":
        return out["p_state_routing"][:, 0]
    if mode == "patch_mean_p_features_tk":
        return out["p_semantic_tk"][:, 1:].mean(dim=1)
    if mode == "cls_p_features_tk":
        # D 위치 CLS = interpreter_2 후 CLS (= v13 predicted_cls_tk, DINO target과 align)
        return out["p_semantic_tk"][:, 0]
    if mode == "patch_mean_concat_enc_d_prime":
        return torch.cat(
            [out["m_encoded"][:, 1:].mean(dim=1),
             out["p_state_routing"][:, 1:].mean(dim=1)],
            dim=-1,
        )
    if mode == "patch_mean_concat_enc_phase3":
        return torch.cat(
            [out["m_encoded"][:, 1:].mean(dim=1),
             out["p_semantic_tk"][:, 1:].mean(dim=1)],
            dim=-1,
        )
    if mode == "patch_mean_concat_p_enc_d_prime":
        # B + D' (P encoder + motion-routing 후)
        return torch.cat(
            [out["p_encoded"][:, 1:].mean(dim=1),
             out["p_state_routing"][:, 1:].mean(dim=1)],
            dim=-1,
        )
    if mode == "patch_mean_concat_p_enc_phase3":
        # B + D (P encoder + interpreter_2 후) — v13 디자인 충실 concat
        return torch.cat(
            [out["p_encoded"][:, 1:].mean(dim=1),
             out["p_semantic_tk"][:, 1:].mean(dim=1)],
            dim=-1,
        )
    if mode == "patch_mean_concat_all":
        # A + B + D' (M_enc + P_enc + motion-routing 후)
        return torch.cat(
            [out["m_encoded"][:, 1:].mean(dim=1),
             out["p_encoded"][:, 1:].mean(dim=1),
             out["p_state_routing"][:, 1:].mean(dim=1)],
            dim=-1,
        )
    raise ValueError(f"Unknown cls_mode: {mode}")


# ============================================================================
# Embedding extraction loop
# ============================================================================

def extract_embeddings(model, dataloader, device, mode: str):
    all_emb = []
    all_act = []
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        actions = batch["action"]
        emb = extract_repr(model, pixel_values, mode)
        all_emb.append(emb.cpu())
        all_act.append(actions)
    return torch.cat(all_emb, dim=0), torch.cat(all_act, dim=0)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Action Probing — Two-Stream v11")
    parser.add_argument("--encoder", type=str, default="two-stream-v11",
                        choices=["two-stream-v11"],
                        help="Fixed for this script (kept for CLI compat)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="v11 model checkpoint path")
    parser.add_argument("--egodex-root", type=str, required=True)
    parser.add_argument("--frames-root", type=str, required=True)
    parser.add_argument("--egodex-split", type=str, default="test")
    parser.add_argument("--gap", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--cls-mode", type=str, default="patch_mean_p_features_tk",
                        choices=sorted(CLS_MODES_ALL))
    parser.add_argument("--p-depth", type=int, default=12)
    parser.add_argument("--m-depth", type=int, default=6)
    parser.add_argument("--max-videos", type=int, default=None,
                        help="(probe_action.py 호환) limit videos for dry-run")
    parser.add_argument("--output-dir", type=str, default="data/probing_results")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Encoder: {args.encoder} (p_depth={args.p_depth}, m_depth={args.m_depth})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"EgoDex split: {args.egodex_split}, gap: {args.gap}")
    print(f"CLS mode: {args.cls_mode}")

    # ---- 1. Load model ----
    print("\n" + "=" * 60)
    print("Loading v11 model (mask_ratio=0 for deterministic probing)...")
    print("=" * 60)
    t0 = time.time()
    model = load_v11_model(
        args.checkpoint, p_depth=args.p_depth, m_depth=args.m_depth, device=device,
    )
    embed_dim = _embed_dim(args.cls_mode)
    print(f"Model loaded in {time.time() - t0:.1f}s, embed_dim={embed_dim}")

    # ---- 2. Build datasets ----
    print("\n" + "=" * 60)
    print("Building datasets...")
    print("=" * 60)
    t0 = time.time()
    train_ds, eval_ds = build_datasets(
        args.egodex_root, args.frames_root, args.egodex_split,
        gap=args.gap, max_videos=args.max_videos,
    )
    print(f"Datasets built in {time.time() - t0:.1f}s")
    if len(train_ds) == 0 or len(eval_ds) == 0:
        print("ERROR: No valid samples found.")
        sys.exit(1)

    # ---- 3. Extract embeddings ----
    print("\n" + "=" * 60)
    print("Extracting embeddings (frozen v11)...")
    print("=" * 60)
    t0 = time.time()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    train_emb, train_act = extract_embeddings(model, train_loader, device, args.cls_mode)
    eval_emb, eval_act = extract_embeddings(model, eval_loader, device, args.cls_mode)

    print(f"Extracted in {time.time() - t0:.1f}s")
    print(f"  Train: {train_emb.shape} embeddings, {train_act.shape} actions")
    print(f"  Eval:  {eval_emb.shape} embeddings, {eval_act.shape} actions")
    print(f"  Action mean abs: {train_act.abs().mean():.6f}, std: {train_act.std():.6f}")

    # ---- 4. Train probe ----
    print("\n" + "=" * 60)
    print(f"Training {args.probe} probe...")
    print("=" * 60)
    probe = LinearProbe(embed_dim) if args.probe == "linear" else MLPProbe(embed_dim)
    print(f"Probe params: {sum(p.numel() for p in probe.parameters()):,}")

    best_metrics = train_probe(
        probe=probe,
        train_emb=train_emb, train_act=train_act,
        eval_emb=eval_emb, eval_act=eval_act,
        epochs=args.epochs,
        batch_size=min(256, len(train_ds)),
        lr=args.lr,
        device=device,
    )

    # ---- 5. Report ----
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Encoder:    {args.encoder}")
    print(f"CLS mode:   {args.cls_mode}")
    print(f"Probe:      {args.probe}")
    print(f"Split:      {args.egodex_split}, gap={args.gap}")
    print(f"R²:         {best_metrics['r2']:.4f}  "
          f"{'PASS' if best_metrics['r2'] > 0.7 else 'FAIL'} (threshold: 0.7)")
    print(f"MSE:        {best_metrics['mse']:.6f}")
    print(f"Cosine Sim: {best_metrics['cosine_sim']:.4f}")
    print("\nPer-joint R²:")
    for joint, r2 in best_metrics["per_joint_r2"].items():
        print(f"  {joint:30s}: {r2:.4f}")

    # ---- 6. Save ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "encoder": args.encoder,
        "probe": args.probe,
        "cls_mode": args.cls_mode,
        "gap": args.gap,
        "checkpoint": args.checkpoint,
        "egodex_split": args.egodex_split,
        "p_depth": args.p_depth,
        "m_depth": args.m_depth,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_videos": args.max_videos,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "embed_dim": embed_dim,
        "timestamp": timestamp,
        **best_metrics,
    }
    ckpt_tag = Path(args.checkpoint).stem
    result_path = (output_dir /
                   f"probe_v11_{ckpt_tag}_{args.cls_mode}_gap{args.gap}_"
                   f"{args.egodex_split}_{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    main()
