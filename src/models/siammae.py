"""
SiamMAE: Siamese Masked Autoencoders (Gupta et al., NeurIPS 2023).

이 프로젝트의 Paper 2 controlled baseline. 원논문(arXiv 2305.14344) §3 + 비공식
재구현(Robiwan245/SiamMAE) 교차검증으로 충실 재현. 상세 spec: docs/siammae_baseline_plan.md.

핵심 (표준 MAE / VideoMAE-ours와의 차이):
- Siamese 2D 인코더: 두 프레임 f1, f2를 가중치 공유 ViT로 **독립** 인코딩 (3D tubelet 아님)
- 비대칭 마스킹: f1 0% (full) / f2 95%
- cross-self decoder: f2 토큰이 f1 토큰을 cross-attend → f2 self-attend → masked f2 픽셀 예측
- Loss: normalized-pixel L2, masked f2 패치만

parity: 인코더 block은 VideoMAE-ours와 동일 `Block`(modeling_finetune) + ViT-B(768/12/12).
원논문 ViT-S에서 scale-up (capacity를 VideoMAE-ours와 통제, objective/구조만 변수화).
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

VIDEOMAE_PATH = Path(__file__).parent.parent.parent / "external" / "VideoMAE"
sys.path.insert(0, str(VIDEOMAE_PATH))

from modeling_finetune import Block, get_sinusoid_encoding_table
from timm.models.layers import trunc_normal_ as _trunc_normal_


def trunc_normal_(tensor, mean=0.0, std=1.0):
    _trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


# Encoder size presets. decoder는 두 size 공통 dec512d8b (원논문 ViT-S도 동일).
#   small = 원논문 SiamMAE 기본 (ViT-S/16, ~22M encoder)
#   base  = baseline-matched (ViT-B/16, ~86M encoder = VideoMAE-ours parity)
SIAMMAE_SIZES = {
    "small": dict(encoder_embed_dim=384, encoder_depth=12, encoder_num_heads=6),
    "base": dict(encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12),
}


class PatchEmbed2D(nn.Module):
    """2D patch embedding: image [B, C, H, W] → patches [B, N, D]."""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(image).flatten(2).transpose(1, 2)


class SiamMAEEncoder(nn.Module):
    """Siamese 2D ViT encoder. 한 프레임을 인코딩 (가중치는 두 프레임에 공유)."""

    def __init__(
        self,
        image_size=224, patch_size=16, in_channels=3,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed2D(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0.0,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def random_masking(x: torch.Tensor, mask_ratio: float):
        """
        MAE-standard per-sample random masking.

        Returns:
            x_masked: [B, N_keep, D] visible tokens (shuffled)
            mask:     [B, N] 1=masked, 0=visible (original order)
            ids_restore: [B, N] indices to unshuffle full set back to original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, image: torch.Tensor, mask_ratio: float):
        """image [B, C, H, W] → (visible tokens, mask, ids_restore). pos embed는 마스킹 전 부여 (MAE)."""
        x = self.patch_embed(image)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, ids_restore


class CrossAttention(nn.Module):
    """Multi-head cross-attention: query=x2(f2), key/value=x1(f1)."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class CrossSelfDecoderBlock(nn.Module):
    """
    SiamMAE cross-self decoder block (논문 §3 / 재구현 일치):
        x2 = x2 + cross_attn(LN(x2)_q, LN(x1)_kv)   # f2가 f1 참조
        x2 = x2 + self_attn(LN(x2))                  # f2 self-attn (ablation상 필수)
        x2 = x2 + mlp(LN(x2))
    x1 (f1 메모리)은 블록 간 고정 (transformer decoder의 encoder memory와 동일).
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x2 = x2 + self.cross_attn(self.norm_q(x2), self.norm_kv(x1))
        x2 = x2 + self.self_attn(self.norm_self(x2))
        x2 = x2 + self.mlp(self.norm_mlp(x2))
        return x2


class SiamMAEModel(nn.Module):
    """
    SiamMAE: 비대칭 2-frame masked autoencoder via cross-self prediction.

    인터페이스는 VideoMAEModel과 동일 (compute_loss(img_t, img_tk) → (loss, pred)).
    training loop는 두 모델을 같은 분기로 처리.

    config (ViT-B/16 + dec512d8b):
        encoder 768/12/12, decoder 512/8/16, mask_ratio_f2=0.95, normalized pixel target.
    """

    def __init__(
        self,
        image_size=224, patch_size=16, in_channels=3,
        size="base",
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, mask_ratio=0.95, normalize_target=True,
    ):
        super().__init__()
        # size preset이 주어지면 encoder dim override (small=ViT-S / base=ViT-B)
        cfg = SIAMMAE_SIZES[size]
        encoder_embed_dim = cfg["encoder_embed_dim"]
        encoder_depth = cfg["encoder_depth"]
        encoder_num_heads = cfg["encoder_num_heads"]
        self.size = size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        self.normalize_target = normalize_target

        self.encoder = SiamMAEEncoder(
            image_size=image_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=encoder_embed_dim, depth=encoder_depth,
            num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
        )
        self.num_patches = self.encoder.num_patches

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = get_sinusoid_encoding_table(self.num_patches, decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList([
            CrossSelfDecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, in_channels * patch_size * patch_size)

        trunc_normal_(self.mask_token, std=0.02)

    def _to_decoder_tokens(self, enc_tokens, ids_restore=None):
        """encoder 출력 → decoder dim + pos embed. ids_restore 주어지면 mask token 채워 full set 복원."""
        B = enc_tokens.shape[0]
        x = self.encoder_to_decoder(enc_tokens)
        if ids_restore is not None:
            n_masked = self.num_patches - x.shape[1]
            mask_tokens = self.mask_token.expand(B, n_masked, -1)
            x = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        pos = self.decoder_pos_embed.type_as(x).to(x.device).clone().detach()
        return x + pos

    def forward(self, image_current: torch.Tensor, image_future: torch.Tensor):
        """
        Args: image_current=f1 [B,3,H,W], image_future=f2 [B,3,H,W]
        Returns: pred [B, N, patch_pixels] (full set, masked 위치가 학습 대상), mask2 [B, N] (1=masked f2)
        """
        # Siamese 인코딩: f1 full(0%), f2 95% masked
        x1_enc, _, _ = self.encoder(image_current, mask_ratio=0.0)
        x2_enc, mask2, ids_restore2 = self.encoder(image_future, mask_ratio=self.mask_ratio)

        # decoder 입력: x1=f1 메모리(full), x2=f2 full set(mask token 복원)
        x1 = self._to_decoder_tokens(x1_enc, ids_restore=None)
        x2 = self._to_decoder_tokens(x2_enc, ids_restore=ids_restore2)

        for block in self.decoder_blocks:
            x2 = block(x1, x2)
        pred = self.decoder_head(self.decoder_norm(x2))
        return pred, mask2

    def _patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs [B,C,H,W] → [B, N, p*p*C]."""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, h * w, p * p * C)

    def _normalize_target(self, patches: torch.Tensor) -> torch.Tensor:
        """MAE norm_pix_loss: 패치별 픽셀 벡터 정규화 (last dim)."""
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, unbiased=True, keepdim=True)
        return (patches - mean) / (var + 1e-6).sqrt()

    def compute_loss(self, image_current: torch.Tensor, image_future: torch.Tensor):
        """
        VideoMAEModel.compute_loss와 동일 시그니처. masked f2 패치만 reconstruction loss.

        Returns: (loss scalar, pred [B, N, patch_pixels])
        """
        pred, mask2 = self.forward(image_current, image_future)

        target = self._patchify(image_future)
        if self.normalize_target:
            target = self._normalize_target(target)

        loss = ((pred - target) ** 2).mean(dim=-1)        # [B, N]
        loss = (loss * mask2).sum() / mask2.sum().clamp(min=1.0)
        return loss, pred


class SiamMAEEncoderForVLA(nn.Module):
    """
    Downstream(BC-T / probing)용 SiamMAE 인코더 wrapper.

    Siamese 인코더는 프레임별 단독 인코딩이므로, 입력 6ch(prev⊕curr) 중 current 프레임을
    마스킹 없이 인코딩 → patch tokens [B, N, D]. (single-frame 어댑터 패턴과 호환)
    """

    def __init__(
        self, checkpoint_path: str | None = None, size="base",
        embed_dim=None, depth=None, num_heads=None, image_size=224, patch_size=16,
    ):
        super().__init__()
        cfg = SIAMMAE_SIZES[size]
        embed_dim = embed_dim or cfg["encoder_embed_dim"]
        depth = depth or cfg["encoder_depth"]
        num_heads = num_heads or cfg["encoder_num_heads"]
        self._embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.encoder = SiamMAEEncoder(
            image_size=image_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        )
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def load_from_checkpoint(self, checkpoint_path: str):
        """학습된 SiamMAE 체크포인트에서 encoder weight만 로드 (DDP 'module.' prefix 자동 제거)."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                          for k, v in state_dict.items()}
        encoder_state = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
        result = self.load_state_dict(encoder_state, strict=False)
        if result.missing_keys:
            print(f"  WARNING: {len(result.missing_keys)} missing encoder keys "
                  f"(first 3: {result.missing_keys[:3]})")
        print(f"Loaded SiamMAE encoder weights from: {checkpoint_path}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values [B, 6, H, W] (prev⊕curr) → current 프레임 patch tokens [B, N, D] (no mask)."""
        image_current = pixel_values[:, 3:]
        tokens, _, _ = self.encoder(image_current, mask_ratio=0.0)
        return tokens
