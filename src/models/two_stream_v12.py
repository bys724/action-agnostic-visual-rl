"""
Two-Stream v12: Semantic Residual + EMA Teacher (post-CoRL follow-up idea).

v11 (motion-routing patch reconstruction) 위에 CLS-level semantic residual path를 추가:
    semantic_p_tk = semantic_p_t + semantic_m              (의도된 inductive bias)
    predicted_p_tk = CrossAttnPredictor(semantic_p_t, semantic_m)   (학습)
    target_p_tk    = stop-grad(Teacher_P(unmasked frame_{t+k}).CLS)

추가 loss:
    L_residual = SmoothL1(predicted_p_tk, target_p_tk)
    L_var      = VICReg variance hinge over {sem_m, sem_p_t, predicted_p_tk}
    L_cov      = VICReg off-diagonal squared

v8 EMA failure 방어:
    - VICReg variance + covariance 명시 적용
    - Teacher: encoder + head 모두 EMA copy (moving target 방지)
    - Predictor: student only (BYOL/SimSiam asymmetric prediction)
    - Cross-Attention predictor: Q from P (semantic core), K/V from M (motion operator)
      v11 motion-routing 철학과 일관 (M이 attention 결정)
    - 보수적 λ_residual (default 0.05)

v11 reconstruction path는 그대로 유지 — L_total = L_t + L_tk + λ·L_residual + α·L_var + β·L_cov

상세 설계: vault `Projects/Action-Agnostic Paper/v12 - Semantic Residual + EMA (Future).md`
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .two_stream_v11 import TwoStreamV11Model


# ============================================================================
# VICReg loss helpers (V-JEPA 1 / VICReg 표준)
# ============================================================================

def vicreg_var_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Variance hinge: penalize per-dim std below 1.

    Args:
        z: [B, D]
    Returns:
        scalar — mean over D of max(0, 1 - sqrt(var(z) + eps))
    """
    # FP32 강제 — BF16 var는 underflow 위험
    z = z.float()
    std = torch.sqrt(z.var(dim=0, unbiased=True) + eps)
    return F.relu(1.0 - std).mean()


def vicreg_cov_loss(z: torch.Tensor) -> torch.Tensor:
    """Off-diagonal squared covariance, normalized by D.

    Args:
        z: [B, D]
    Returns:
        scalar — sum_{i!=j} cov(z)_{ij}^2 / D
    """
    z = z.float()
    B, D = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / max(B - 1, 1)
    off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
    return (cov[off_diag_mask] ** 2).sum() / D


# ============================================================================
# Cross-Attention Predictor (single-token, 1 transformer block)
# ============================================================================

class CrossAttnPredictor(nn.Module):
    """Predict semantic_p_tk from (semantic_p_t, semantic_m) via single cross-attn block.

    Q from P (semantic core), K/V from M (motion operator) — v11 motion-routing의 mirror.
    Single-token cross-attn은 capacity 작음 → MLP 4× expansion으로 보강.
    """

    def __init__(self, embed_dim: int, num_heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

        self.norm_ffn = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, semantic_p_t: torch.Tensor, semantic_m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_p_t: [B, D] — Q source (P CLS through P_head)
            semantic_m:   [B, D] — K/V source (M CLS through M_head)
        Returns:
            predicted_p_tk: [B, D]
        """
        B, D = semantic_p_t.shape

        # Add seq_len dim for SDPA: [B, 1, D]
        q_in = self.norm_q(semantic_p_t).unsqueeze(1)
        kv_in = self.norm_kv(semantic_m).unsqueeze(1)

        q = self.q_proj(q_in).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv_in).reshape(B, 1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)  # [B, H, 1, D_head]
        attn = attn.transpose(1, 2).reshape(B, 1, D).squeeze(1)  # [B, D]

        # Residual + projection (input = semantic_p_t, P stream baseline)
        x = semantic_p_t + self.proj_out(attn)
        # FFN + residual
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ============================================================================
# Teacher P branch (EMA copy of student P encoder + P head)
# ============================================================================

class TeacherP(nn.Module):
    """EMA copy of student P encoder + P head. Forward는 frozen, no gradient."""

    def __init__(self, student: "TwoStreamV12Model"):
        super().__init__()
        # Deep copy student modules
        self.preprocessing = copy.deepcopy(student.preprocessing)
        self.patch_embed_p = copy.deepcopy(student.patch_embed_p)
        self.cls_token_p = nn.Parameter(student.cls_token_p.data.clone())
        self.pos_embed_p = nn.Parameter(student.pos_embed_p.data.clone())
        self.blocks_p = copy.deepcopy(student.blocks_p)
        self.norm_p = copy.deepcopy(student.norm_p)
        self.p_head = copy.deepcopy(student.p_head)

        # Freeze all teacher params (EMA-only update path)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, image_future: torch.Tensor) -> torch.Tensor:
        """Teacher P branch on unmasked frame_{t+k}.

        Returns:
            semantic_p_tk_target: [B, D] — frozen target representation
        """
        B = image_future.shape[0]
        # P channel from frame_{t+k} only (encode_single 패턴 — preprocessing 재사용)
        p_channel = self.preprocessing.compute_p_channel(image_future)

        p_patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)  # [B, N, D]
        p_cls = self.cls_token_p.expand(B, -1, -1)
        x = torch.cat([p_cls, p_patches], dim=1) + self.pos_embed_p

        for block in self.blocks_p:
            x = block(x, freqs_cis=None)
        x = self.norm_p(x)

        cls_out = x[:, 0]
        return self.p_head(cls_out)

    @torch.no_grad()
    def update(self, student: "TwoStreamV12Model", momentum: float):
        """EMA update: teacher = momentum * teacher + (1 - momentum) * student.

        Encoder + head 모두 업데이트. Buffer는 student 그대로 복사
        (preprocessing 내부 sobel_x/y, luminance_weights 등은 register_buffer라 학습 안됨).
        """
        student_modules = {
            "patch_embed_p": student.patch_embed_p,
            "blocks_p": student.blocks_p,
            "norm_p": student.norm_p,
            "p_head": student.p_head,
        }
        teacher_modules = {
            "patch_embed_p": self.patch_embed_p,
            "blocks_p": self.blocks_p,
            "norm_p": self.norm_p,
            "p_head": self.p_head,
        }
        for key in student_modules:
            for p_t, p_s in zip(
                teacher_modules[key].parameters(), student_modules[key].parameters()
            ):
                p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)

        # cls_token_p, pos_embed_p (top-level Parameters)
        self.cls_token_p.data.mul_(momentum).add_(student.cls_token_p.data, alpha=1.0 - momentum)
        self.pos_embed_p.data.mul_(momentum).add_(student.pos_embed_p.data, alpha=1.0 - momentum)


# ============================================================================
# v12 Model
# ============================================================================

class TwoStreamV12Model(TwoStreamV11Model):
    """Two-Stream v12: v11 + Semantic Residual + EMA Teacher.

    v11 모든 reconstruction path 유지 (L_t + L_tk).
    추가:
        - M_head, P_head: 2-layer MLP (D → D)
        - CrossAttnPredictor: 1 transformer block, single-token cross-attn (Q←P, K/V←M)
        - TeacherP: EMA copy of P encoder + P head, frozen forward on unmasked frame_{t+k}
        - Loss: λ_residual·SmoothL1 + α·VICReg-var + β·VICReg-cov

    EMA update는 training loop에서 매 optimizer step 후 호출:
        model.update_teacher(momentum)

    Args:
        residual_weight: λ_residual (default 0.05, v8 1차 0.2 scale 실패 교훈)
        vicreg_var_weight: α (default 1.0, V-JEPA 1 standard)
        vicreg_cov_weight: β (default 1.0)
        predictor_num_heads: cross-attn heads (default 12)
        그 외 args는 TwoStreamV11Model과 동일.

    NOTE: TeacherP는 self.__init__ 마지막에 student 모듈을 deepcopy해서 만들어진다.
    TeacherP를 만들기 전에 student의 모든 모듈 init이 완료되어야 한다.
    """

    def __init__(
        self,
        # v11 args
        embed_dim: int = 768,
        p_depth: int = 12,
        m_depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        image_size: int = 224,
        patch_size: int = 16,
        mask_ratio_m: float = 0.3,
        mask_ratio_p: float = 0.75,
        decoder_depth_m: int = 3,
        interpreter_depth: int = 3,
        num_motion_iters: int = 2,
        rotation_aug: bool = False,
        independent_rotation_prob: float = 0.1,
        routing_mode: str = "v_from_p",
        # v12 args
        residual_weight: float = 0.05,
        vicreg_var_weight: float = 1.0,
        vicreg_cov_weight: float = 1.0,
        predictor_num_heads: int = 12,
    ):
        super().__init__(
            embed_dim=embed_dim,
            p_depth=p_depth,
            m_depth=m_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            image_size=image_size,
            patch_size=patch_size,
            mask_ratio_m=mask_ratio_m,
            mask_ratio_p=mask_ratio_p,
            decoder_depth_m=decoder_depth_m,
            interpreter_depth=interpreter_depth,
            num_motion_iters=num_motion_iters,
            rotation_aug=rotation_aug,
            independent_rotation_prob=independent_rotation_prob,
            routing_mode=routing_mode,
        )

        # ── v12 hyperparameters ────────────────────────────────────────
        self.residual_weight = residual_weight
        self.vicreg_var_weight = vicreg_var_weight
        self.vicreg_cov_weight = vicreg_cov_weight

        # ── Semantic heads (M, P): 2-layer MLP D → D, GELU ─────────────
        # Bias 포함 (representation projection 표준)
        def _build_head(dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

        self.m_head = _build_head(embed_dim)
        self.p_head = _build_head(embed_dim)

        # ── Predictor: 1 cross-attn block, single-token ────────────────
        self.predictor = CrossAttnPredictor(
            embed_dim=embed_dim,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
        )

        # ── Teacher: EMA copy of student P encoder + p_head ────────────
        # student 모든 init 끝난 뒤에 deepcopy. self.add_module로 등록되지만
        # teacher_p.parameters()는 모두 requires_grad=False라 optimizer에서 제외됨.
        self.teacher_p = TeacherP(self)

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """Training loop에서 매 optimizer step 후 호출.

        Args:
            momentum: EMA momentum (0.996 → 0.9999 linear warmup over training)
        """
        self.teacher_p.update(self, momentum)

    def forward(
        self,
        image_current: torch.Tensor,
        image_future: torch.Tensor,
    ) -> dict:
        """v11 forward 위에 semantic residual + VICReg loss 추가.

        Returns dict (v11 output + 다음 키 추가):
            loss_residual: scalar, λ_residual·SmoothL1
            loss_var:      scalar, α·VICReg variance hinge
            loss_cov:      scalar, β·VICReg off-diagonal squared
            semantic_m:        [B, D]
            semantic_p_t:      [B, D]
            predicted_p_tk:    [B, D]
            teacher_p_tk:      [B, D]
        """
        # ── v11 reconstruction path (rotation aug는 super 내부에서 처리) ─
        # super().forward()가 image_current/image_future를 rotation aug로 변형할 수
        # 있으므로, teacher input도 동일하게 변형된 future를 받아야 함.
        # → super().forward에서 rotated 후 결과만 받아서는 teacher가 unrotated future를
        #   보게 됨. rotation aug를 v12 forward에서 명시적으로 먼저 적용하고
        #   super().forward에는 aug-off 우회로 호출하기 위해 v11 internals를 재호출.
        # 그러나 v11 forward는 rotation_aug를 self.training 기준으로 처리해서
        # v12에서 super 호출 시 self.training=True면 다시 회전됨 → 이중 회전.
        # 해결: rotation aug를 v12에서 직접 처리하고, super 호출은 그대로 하되
        #       rotation_aug 임시 비활성화.
        if self.rotation_aug and self.training:
            image_current, image_future = self._apply_rotation_aug(
                image_current, image_future
            )

        # super().forward는 rotation aug를 다시 적용하지 않도록 일시 비활성화
        rotation_was_on = self.rotation_aug
        self.rotation_aug = False
        try:
            v11_out = super().forward(image_current, image_future)
        finally:
            self.rotation_aug = rotation_was_on

        # ── Semantic heads — student CLS 통과 ──────────────────────────
        # cls_m / cls_p는 v11이 encoder norm 직후 CLS 그대로 노출 (masked encoding 결과)
        cls_m = v11_out["cls_m"]      # [B, D]
        cls_p = v11_out["cls_p"]      # [B, D]
        semantic_m = self.m_head(cls_m)
        semantic_p_t = self.p_head(cls_p)

        # ── Predictor (student) ────────────────────────────────────────
        predicted_p_tk = self.predictor(semantic_p_t, semantic_m)

        # ── Teacher target (frozen, unmasked frame_{t+k}) ──────────────
        # rotation_aug 후의 image_future를 teacher에게 전달 (위 변형 일관성)
        with torch.no_grad():
            teacher_p_tk = self.teacher_p(image_future).detach()

        # ── Residual prediction loss (Smooth L1, V-JEPA 2 표준) ────────
        # FP32 강제 — BF16 autocast 환경에서 magnitude robust하게.
        loss_residual = F.smooth_l1_loss(predicted_p_tk.float(), teacher_p_tk.float())

        # ── VICReg loss over 3 representations ─────────────────────────
        # variance: per-dim std hinge → mean
        # covariance: off-diagonal squared, normalized by D
        var_m = vicreg_var_loss(semantic_m)
        var_pt = vicreg_var_loss(semantic_p_t)
        var_pred = vicreg_var_loss(predicted_p_tk)
        loss_var = (var_m + var_pt + var_pred) / 3.0

        cov_m = vicreg_cov_loss(semantic_m)
        cov_pt = vicreg_cov_loss(semantic_p_t)
        cov_pred = vicreg_cov_loss(predicted_p_tk)
        loss_cov = cov_m + cov_pt + cov_pred

        # ── Total loss ─────────────────────────────────────────────────
        loss_recon = v11_out["loss"]   # L_t + L_tk
        loss = (
            loss_recon
            + self.residual_weight * loss_residual
            + self.vicreg_var_weight * loss_var
            + self.vicreg_cov_weight * loss_cov
        )

        out = dict(v11_out)
        out["loss"] = loss
        out["loss_recon"] = loss_recon
        out["loss_residual"] = loss_residual
        out["loss_var"] = loss_var
        out["loss_cov"] = loss_cov
        out["semantic_m"] = semantic_m
        out["semantic_p_t"] = semantic_p_t
        out["predicted_p_tk"] = predicted_p_tk
        out["teacher_p_tk"] = teacher_p_tk
        return out

    def compute_loss(
        self, image_current: torch.Tensor, image_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(image_current, image_future)
        return out["loss"], out["pred_tk"]


# ============================================================================
# v12 Encoder Wrapper (downstream — student encoder만 로드)
# ============================================================================
# Encoder structure는 v11과 동일 (M/P encoder만 추출). 따라서 downstream 평가는
# 기존 TwoStreamV11Encoder 그대로 v12 ckpt에 사용 가능 (state_dict prefix 일치).
# 별도 wrapper 불필요. v12-specific semantic head 평가는 별도 probing 스크립트에서.
