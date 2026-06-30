#!/usr/bin/env python
"""
Action Probing Experiment

논문 핵심 주장 검증: "action-agnostic으로 학습해도 행동 정보가 인코딩된다"

EgoDex (img_t, img_t+1)을 인코더에 넣고,
frozen embedding에서 linear probe로 hand pose delta(action)를 예측할 수 있는지 테스트.

GO/NO-GO 기준: R² > 0.7

인코더 4종:
    - two-stream: TwoStreamEncoder (체크포인트 필요)
    - videomae: VideoMAEEncoderForVLA (체크포인트 필요)
    - clip: CLIPVisionModel (HuggingFace, pretrained)
    - dinov2: Dinov2Model (HuggingFace, pretrained)

Usage:
    # 학습 데이터(part1)에서 probing
    python scripts/eval/probe_action.py \\
        --encoder two-stream \\
        --checkpoint /mnt/data/checkpoints/two_stream/.../best_model.pt \\
        --egodex-root /mnt/data/egodex --egodex-split part1

    # 미사용 데이터(part4)에서 probing
    python scripts/eval/probe_action.py \\
        --encoder two-stream \\
        --checkpoint /mnt/data/checkpoints/two_stream/.../best_model.pt \\
        --egodex-root /mnt/data/egodex --egodex-split part4

    # CLIP / DINOv2 baseline
    python scripts/eval/probe_action.py --encoder clip --egodex-root /mnt/data/egodex --egodex-split part1
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Support both docker (/workspace) and local execution
_project_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _project_root)
sys.path.insert(0, "/workspace")

# ============================================================================
# Target joints for action probing
# ============================================================================

TARGET_JOINTS = [
    "rightHand",
    "rightThumbTip",
    "rightIndexFingerTip",
    "rightMiddleFingerTip",
    "rightRingFingerTip",
    "rightLittleFingerTip",
]
ACTION_DIM = len(TARGET_JOINTS) * 3  # 6 joints × 3D = 18

CONFIDENCE_THRESHOLD = 0.3


# ============================================================================
# 1. EgoDexProbingDataset
# ============================================================================

class EgoDexProbingDataset(Dataset):
    """
    EgoDex dataset for action probing.

    MP4 비디오 + HDF5 손 포즈를 동시 로드.
    gap=1 고정 (연속 프레임 간 action 예측).

    Temporal Alignment:
        Input:  (img_t, img_t+1) - 관찰된 변화
        Target: delta_t = pos_t+1 - pos_t - 이 변화를 만든 action

        의미: "변화 임베딩이 action을 역추론할 수 있는가?"
              → Frozen encoder로 R² > 0.7이면
              → 임베딩이 action-informative함을 증명
              → VLM이 임베딩만으로 action planning 가능

    Returns:
        pixel_values: [6, 224, 224] - img_t + img_t+1 채널 concat
        action: [18] - 6관절 × 3D position delta (≈ velocity)
    """

    def __init__(
        self,
        data_root: str,
        frames_root: str,
        video_ids: list,
        split: str,
        gap: int = 1,
        img_size: int = 224,
        target_mode: str = "same",
    ):
        """
        Args:
            data_root: EgoDex 원본 경로 (HDF5 포함, e.g. /mnt/data/egodex)
            frames_root: 추출된 프레임 경로 (e.g. /mnt/data/egodex_frames)
            video_ids: (task, video_id) 튜플 리스트
            split: e.g. "part1", "part4"
            gap: 프레임 간격 (1=연속, 5=5프레임 간격, ...)
            img_size: 출력 이미지 크기
            target_mode:
                "same" (default): target = pose[t+gap] - pose[t]
                  (변화 인지 — features의 frame pair가 이미 정보 보유)
                "future": target = pose[t+2*gap] - pose[t+gap]
                  (causal future prediction — features가 본 frame 너머의 변화)
        """
        assert target_mode in {"same", "future"}, f"Invalid target_mode: {target_mode}"
        self.frames_root = Path(frames_root)
        self.split = split
        self.gap = gap
        self.img_size = img_size
        self.target_mode = target_mode

        # Build (frame_dir, frame_idx, action) tuples
        self.samples = []
        skipped_conf = 0
        skipped_read = 0

        for task, vid_id in video_ids:
            hdf5_path = Path(data_root) / split / task / f"{vid_id}.hdf5"
            frame_dir = self.frames_root / split / task / str(vid_id)

            if not hdf5_path.exists() or not frame_dir.exists():
                continue

            try:
                with h5py.File(str(hdf5_path), "r") as f:
                    transforms = {}
                    confidences = {}
                    for joint in TARGET_JOINTS:
                        t_key = f"transforms/{joint}"
                        c_key = f"confidences/{joint}"
                        if t_key not in f or c_key not in f:
                            break
                        transforms[joint] = f[t_key][()]  # (T, 4, 4)
                        confidences[joint] = f[c_key][()]  # (T,)
                    else:
                        num_frames = transforms[TARGET_JOINTS[0]].shape[0]
                        num_extracted = len(list(frame_dir.glob("frame_*.jpg")))
                        num_frames = min(num_frames, num_extracted)

                        # future mode는 t+2*gap까지 필요
                        max_t = num_frames - (2 * gap if target_mode == "future" else gap)

                        for t in range(max_t):
                            # confidence check: same은 t, t+gap / future는 t+gap, t+2*gap
                            if target_mode == "future":
                                check_indices = (t + gap, t + 2 * gap)
                            else:
                                check_indices = (t, t + gap)

                            valid = True
                            for joint in TARGET_JOINTS:
                                if any(confidences[joint][idx] < CONFIDENCE_THRESHOLD
                                       for idx in check_indices):
                                    valid = False
                                    break

                            if not valid:
                                skipped_conf += 1
                                continue

                            action = np.zeros(ACTION_DIM, dtype=np.float32)
                            for i, joint in enumerate(TARGET_JOINTS):
                                if target_mode == "future":
                                    pos_a = transforms[joint][t + gap, :3, 3]
                                    pos_b = transforms[joint][t + 2 * gap, :3, 3]
                                else:
                                    pos_a = transforms[joint][t, :3, 3]
                                    pos_b = transforms[joint][t + gap, :3, 3]
                                action[i * 3 : (i + 1) * 3] = pos_b - pos_a

                            self.samples.append({
                                "frame_dir": str(frame_dir),
                                "frame_idx": t,
                                "action": action,
                            })
            except Exception:
                skipped_read += 1
                continue

        print(f"EgoDexProbingDataset: {len(self.samples)} samples from {len(video_ids)} videos")
        if skipped_conf > 0:
            print(f"  Skipped (low confidence): {skipped_conf}")
        if skipped_read > 0:
            print(f"  Skipped (read error): {skipped_read}")

    def _load_frame(self, frame_dir: str, frame_idx: int) -> torch.Tensor:
        """Load a pre-extracted JPG frame."""
        frame_path = Path(frame_dir) / f"frame_{frame_idx:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Failed to read: {frame_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 추출 프레임은 256x256, probing은 224x224 center crop
        if frame.shape[0] != self.img_size:
            h, w = frame.shape[:2]
            top = (h - self.img_size) // 2
            left = (w - self.img_size) // 2
            frame = frame[top:top + self.img_size, left:left + self.img_size]

        frame = torch.from_numpy(frame).float() / 255.0  # [H, W, C]
        frame = frame.permute(2, 0, 1)  # [C, H, W]
        return frame

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_dir = sample["frame_dir"]
        t = sample["frame_idx"]

        img_t = self._load_frame(frame_dir, t)
        img_t1 = self._load_frame(frame_dir, t + self.gap)

        pixel_values = torch.cat([img_t, img_t1], dim=0)  # [6, H, W]
        action = torch.from_numpy(sample["action"])  # [18]

        return {"pixel_values": pixel_values, "action": action}


def build_datasets(
    egodex_root: str,
    frames_root: str,
    egodex_split: str = "part1",
    gap: int = 1,
    max_videos: int = None,
    train_ratio: float = 0.8,
    target_mode: str = "same",
):
    """Build train/eval datasets with video-level split.

    Args:
        egodex_root: 원본 경로 (HDF5 포함, e.g. /mnt/data/egodex)
        frames_root: 추출된 프레임 경로 (e.g. /mnt/data/egodex_frames)
        egodex_split: Which split to use (e.g. "part1", "part4")
        max_videos: Limit number of videos (for debugging)
        train_ratio: Train/eval split ratio
    """
    split_dir = Path(egodex_root) / egodex_split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # (task, video_id) 쌍 수집
    video_ids = []
    for mp4_path in sorted(split_dir.glob("**/*.mp4")):
        task = mp4_path.parent.name
        vid_id = mp4_path.stem
        video_ids.append((task, vid_id))

    print(f"\nFound {len(video_ids)} videos in {split_dir}")

    if max_videos:
        video_ids = video_ids[:max_videos]

    # Video-level 80/20 split
    n_train = int(len(video_ids) * train_ratio)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(video_ids))
    train_ids = [video_ids[i] for i in indices[:n_train]]
    eval_ids = [video_ids[i] for i in indices[n_train:]]

    print(f"Dataset split: {len(train_ids)} train / {len(eval_ids)} eval videos")

    train_ds = EgoDexProbingDataset(egodex_root, frames_root, train_ids, egodex_split,
                                    gap=gap, target_mode=target_mode)
    eval_ds = EgoDexProbingDataset(egodex_root, frames_root, eval_ids, egodex_split,
                                   gap=gap, target_mode=target_mode)

    return train_ds, eval_ds


# ============================================================================
# 2. Encoder loading
# ============================================================================

# =============================================================================
# TODO: Phase 2 baseline encoder 로더 보강 (full training 체크포인트 확보 후)
# =============================================================================
# 현재 지원: two-stream, videomae, clip, dinov2
# 누락 (Phase 2 probing 전 반드시 추가):
#
#   (1) SigLIP-Base — Internet-scale VL baseline
#       - 공식 저장소: google-research/big_vision (원본) / HuggingFace 공식 미러
#       - 권장: `google/siglip-base-patch16-224` (HF official) 사용
#       - `transformers.SiglipVisionModel.from_pretrained(...)`
#       - 공식 preprocessing: mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), 224x224
#       - 검증: HuggingFace model card 의 sample code 를 먼저 재현해서 feature
#         shape / norm 이 문서와 일치하는지 확인 후 통합
#
#   (2) VC-1-Base — embodied AI 표준 baseline
#       - 공식 저장소: facebookresearch/eai-vc
#       - 설치: `pip install vc_models` (or git clone + pip install -e .)
#       - API: `from vc_models.models.vit import model_utils;
#                model, _, _, _ = model_utils.load_model(model_utils.VC1_BASE_NAME)`
#       - 공식 preprocessing: ImageNet mean/std, 224x224
#       - 검증: eai-vc README 의 feature extraction 예제 재현 후 통합
#       - 절대 자체 구현하지 말 것. 공식 로더 그대로 사용.
#
#   (3) V-JEPA-ours — 우리가 학습한 체크포인트
#       - src/models/v_jepa.py::VJEPAModel.extract_features() 사용
#       - `load_from_checkpoint` 유사 패턴으로 x_encoder 만 로드 (y_encoder 불필요)
#       - EMA y_encoder 는 체크포인트에서 제외 가능 (frozen inference 에는 x 만 필요)
#
# CLIP 제거 검토:
#   - 5-encoder 실험 목록에 없음 (SigLIP 이 Internet-scale VL 역할 담당)
#   - Phase 1 에서 baseline 참고용으로만 써서 남아 있음
#   - Phase 2 이후 제거 or "legacy baseline" 으로 라벨링
#
# 2-frame 입력 규약 (모든 단일 프레임 baseline 공통):
#   (a) 각 프레임 독립 forward → concat 방식 권장 (코드 기존 방식 그대로)
#   (b) 각 encoder 의 "공식 preprocessing" 을 그대로 사용 (mean/std, resolution)
#   (c) feature 추출 위치: CLS token 또는 patch mean pool — probe 실험에서 결정
#
# 공식 저장소 검증 절차 (각 baseline 통합 전 반드시):
#   1. 공식 README 의 feature extraction quickstart 코드 실행
#   2. 공식 예제 이미지로 feature shape / norm / dtype 확인
#   3. 공식 문서가 명시한 layer / token 선택 규약을 그대로 따름
#   4. 우리 래퍼에서 동일 입력 → 동일 feature 재현 확인 (값까지 같아야 함)
# =============================================================================
def load_encoder(name: str, checkpoint: str = None, device: str = "cuda",
                 cls_mode: str = "average", depth: int = 12, num_stages: int = 3):
    """
    Load encoder by name.

    Returns:
        (encoder, embed_dim) - frozen encoder module and its output dimension
    """
    if name == "two-stream":
        from src.models.two_stream import TwoStreamEncoder
        assert checkpoint, "--checkpoint required for two-stream ('random' for untrained baseline)"
        ckpt_arg = None if checkpoint == "random" else checkpoint
        encoder = TwoStreamEncoder(
            checkpoint_path=ckpt_arg, depth=depth, num_stages=num_stages,
        )
        encoder.to(device)
        encoder.eval()
        base_dim = encoder._embed_dim  # 768
        K = getattr(encoder.encoder, 'num_p_cls', 1)
        if cls_mode in ("concat", "patch_mean_concat"):
            embed_dim = base_dim * 2
        elif cls_mode == "all_cls_concat":
            embed_dim = base_dim * (1 + K)  # v7-big: 3D, legacy: 2D
        else:  # average, m_only, p_only, patch_mean, patch_mean_m/p, cls_p_bg, cls_p_motion
            embed_dim = base_dim
        return encoder, embed_dim

    elif name == "parvo":
        # Parvo (TwoStreamV15Model, no-Sobel: P=RGB 3ch, M=ΔL 1ch). pair_mode.
        from src.models.two_stream_v15 import TwoStreamV15Model
        assert checkpoint and checkpoint != "random", "--checkpoint required for parvo"
        ckpt = torch.load(checkpoint, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # arch(dim/head/depth)는 ckpt에서 추론 — ViT-B(768) vs ViT-S(384/6, CoMP-MAE-S) 자동 대응
        # (siammae 경로와 동일 원칙: shape mismatch silent-skip 방지). head_dim=64 표준.
        _ed = next(v.shape[-1] for k, v in sd.items() if k == "pos_embed_p")
        _md = len({k.split(".")[1] for k in sd if k.startswith("blocks_m.")})
        _comp = any("m_recon" in k for k in sd)  # CoMP-MAE = M-recon 분기 보유
        encoder = TwoStreamV15Model(
            embed_dim=_ed, num_heads=_ed // 64, m_depth=_md, comp_mae=_comp,
            pair_mode=True, use_sobel=False, masked_anchor=True,
        )
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        # P/M encoder 가중치가 실제로 로드됐는지 확인 (random이면 probe 무의미)
        _enc_missing = [k for k in missing if k.startswith(("blocks_p", "blocks_m", "patch_embed_p", "patch_embed_m"))]
        assert not _enc_missing, f"parvo: P/M encoder 가중치 미로드 {_enc_missing[:5]}"
        encoder.to(device).eval()
        base_dim = encoder.embed_dim
        embed_dim = base_dim * 2 if cls_mode in ("patch_mean_concat_p_t_p_tk",
                                                 "patch_mean_concat_p_t_m") else base_dim
        return encoder, embed_dim

    elif name == "videomae":
        from src.models.videomae import VideoMAEEncoderForVLA
        assert checkpoint, "--checkpoint required for videomae"
        encoder = VideoMAEEncoderForVLA(checkpoint_path=checkpoint)
        encoder.to(device)
        encoder.eval()
        # patch_mean_concat_p_t_p_tk → 같은 frame 2회 forward 후 concat → 2× embed_dim
        embed_dim = encoder.embed_dim * (2 if cls_mode == "patch_mean_concat_p_t_p_tk" else 1)
        return encoder, embed_dim

    elif name == "siammae":
        # SiamMAE (external reference). 프레임별 단독 인코딩 → frame_t/tk 각자 patch mean → concat.
        from src.models.siammae import SiamMAEEncoderForVLA
        assert checkpoint, "--checkpoint required for siammae"
        # size(small=384/base=768)는 체크포인트 embed_dim에서 추론 (mismatch silent-skip 방지)
        _sd = torch.load(checkpoint, map_location="cpu")
        _sd = _sd.get("model_state_dict", _sd)
        _ed = next((v.shape[0] for k, v in _sd.items() if k.endswith("patch_embed.projection.weight")), 768)
        encoder = SiamMAEEncoderForVLA(checkpoint_path=checkpoint, size="small" if _ed == 384 else "base")
        encoder.to(device).eval()
        embed_dim = encoder.embed_dim * (2 if cls_mode == "patch_mean_concat_p_t_p_tk" else 1)
        return encoder, embed_dim

    elif name == "clip":
        from transformers import CLIPVisionModel
        model_id = "openai/clip-vit-base-patch16"
        encoder = CLIPVisionModel.from_pretrained(model_id)
        encoder.to(device)
        encoder.eval()
        # CLS from each frame → concat → 768*2 = 1536
        embed_dim = encoder.config.hidden_size * 2
        return encoder, embed_dim

    elif name == "dinov2":
        from transformers import AutoModel
        model_id = "facebook/dinov2-base"
        encoder = AutoModel.from_pretrained(model_id)
        encoder.to(device)
        encoder.eval()
        embed_dim = encoder.config.hidden_size * 2
        return encoder, embed_dim

    elif name == "siglip":
        from transformers import SiglipVisionModel
        model_id = "google/siglip-base-patch16-224"
        encoder = SiglipVisionModel.from_pretrained(model_id)
        encoder.to(device)
        encoder.eval()
        embed_dim = encoder.config.hidden_size * 2  # 768*2 = 1536
        return encoder, embed_dim

    elif name == "vc1":
        from vc_models.models.vit import model_utils
        model, _, model_transforms, _ = model_utils.load_model(
            model_utils.VC1_BASE_NAME
        )
        model.to(device)
        model.eval()
        embed_dim = 768 * 2  # ViT-B, 각 프레임 CLS concat
        # model_transforms를 encoder에 붙여서 encode_batch에서 접근
        model._vc1_transforms = model_transforms
        return model, embed_dim

    elif name == "vjepa2":
        import sys
        hub_path = os.path.expanduser("~/.cache/torch/hub/facebookresearch_vjepa2_main")
        if hub_path not in sys.path:
            sys.path.insert(0, hub_path)
        from src.hub.backbones import vjepa2_1_vit_base_384, _clean_backbone_key
        encoder, _ = vjepa2_1_vit_base_384(pretrained=False)
        ckpt_path = "/proj/external_group/mrg/checkpoints/vjepa2_official/vjepa2_1_vitb_384.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        encoder_sd = _clean_backbone_key(ckpt["ema_encoder"])
        encoder.load_state_dict(encoder_sd, strict=True)
        encoder.to(device)
        encoder.eval()
        embed_dim = 768 * 2  # 각 프레임 patch mean → concat
        return encoder, embed_dim

    elif name == "videomae-official":
        from transformers import VideoMAEModel
        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        # 2-frame용: position embedding을 196개로 slice
        orig_pe = model.embeddings.position_embeddings  # [1, 1568, 768]
        model.embeddings.position_embeddings = torch.nn.Parameter(orig_pe[:, :196, :])
        model.config.num_frames = 2
        model.to(device)
        model.eval()
        embed_dim = 768  # patch mean pool
        return model, embed_dim

    else:
        raise ValueError(f"Unknown encoder: {name}")


@torch.no_grad()
def encode_batch(encoder, name: str, pixel_values: torch.Tensor, cls_mode: str = "average") -> torch.Tensor:
    """
    Encode a batch of 6-channel images into embeddings.

    Args:
        encoder: loaded encoder
        name: encoder name
        pixel_values: [B, 6, 224, 224]
        cls_mode: Two-Stream CLS 조합 방식
            "average" - (m_cls + p_cls) / 2 → [B, 768]
            "concat"  - [m_cls; p_cls] → [B, 1536]
            "m_only"  - m_cls only → [B, 768]
            "p_only"  - p_cls only → [B, 768]

    Returns:
        embeddings: [B, D]
    """
    if name == "two-stream":
        image_current = pixel_values[:, :3]
        image_future = pixel_values[:, 3:]
        m_channel, p_channel = encoder.preprocessing(image_current, image_future)
        m_tokens, p_tokens = encoder.encoder(m_channel, p_channel)
        # v7-big 호환: P stream은 num_p_cls개 CLS 토큰 (legacy: 1, v7-big: 2)
        K = getattr(encoder.encoder, 'num_p_cls', 1)
        m_cls = m_tokens[:, 0]       # [B, D]
        p_cls_bg = p_tokens[:, 0]    # [B, D]  (v7-big에선 P_cls_bg, legacy에선 일반 P_cls)

        m_patches = m_tokens[:, 1:]  # [B, N, D]
        p_patches = p_tokens[:, K:]  # [B, N, D]  (CLS K개 건너뛰고 patches)

        if cls_mode == "average":
            return (m_cls + p_cls_bg) / 2
        elif cls_mode == "concat":
            return torch.cat([m_cls, p_cls_bg], dim=-1)  # [B, 2D]
        elif cls_mode == "m_only":
            return m_cls
        elif cls_mode == "p_only":
            return p_cls_bg
        elif cls_mode == "patch_mean":
            # M+P patch mean pool → VideoMAE와 동일 조건
            all_patches = torch.cat([m_patches, p_patches], dim=1)  # [B, 2N, D]
            return all_patches.mean(dim=1)  # [B, D]
        elif cls_mode == "patch_mean_m":
            return m_patches.mean(dim=1)
        elif cls_mode == "patch_mean_p":
            return p_patches.mean(dim=1)
        elif cls_mode == "patch_mean_concat":
            # M/P 각각 mean pool 후 concat → probe가 stream별 가중치 학습
            m_mean = m_patches.mean(dim=1)  # [B, D]
            p_mean = p_patches.mean(dim=1)  # [B, D]
            return torch.cat([m_mean, p_mean], dim=-1)  # [B, 2D]
        # v7-big 전용 CLS specialization 검증 모드
        elif cls_mode == "cls_p_bg":
            assert K >= 1, "cls_p_bg needs num_p_cls >= 1"
            return p_tokens[:, 0]  # [B, D]
        elif cls_mode == "cls_p_motion":
            assert K >= 2, "cls_p_motion needs num_p_cls >= 2 (v7-big only)"
            return p_tokens[:, 1]  # [B, D]
        elif cls_mode == "all_cls_concat":
            # v7-big: [CLS_M, CLS_P_bg, CLS_P_motion] concat
            if K == 1:
                return torch.cat([m_cls, p_cls_bg], dim=-1)
            return torch.cat([m_cls] + [p_tokens[:, i] for i in range(K)], dim=-1)
        else:
            raise ValueError(f"Unknown cls_mode: {cls_mode}")

    elif name == "parvo":
        img_t = pixel_values[:, :3]
        img_tk = pixel_values[:, 3:]
        if cls_mode == "patch_mean_p_t":
            # P encoder만, frame_t 단독 patch_mean (단일프레임 appearance) → [B, D].
            # Paper 1 single-frame image MAE readout과 직접 대응되는 baseline.
            p_t = encoder.preprocessing.compute_p_channel(img_t)
            return encoder._encode_p_unmasked(p_t)[:, 1:].mean(dim=1)
        elif cls_mode == "patch_mean_concat_p_t_p_tk":
            # P encoder만: frame_t, frame_tk 각자 patch_mean → concat (appearance)
            p_t = encoder.preprocessing.compute_p_channel(img_t)
            p_tk = encoder.preprocessing.compute_p_channel(img_tk)
            feat_t = encoder._encode_p_unmasked(p_t)[:, 1:].mean(dim=1)   # [B, D]
            feat_tk = encoder._encode_p_unmasked(p_tk)[:, 1:].mean(dim=1)
            return torch.cat([feat_t, feat_tk], dim=-1)                   # [B, 2D]
        elif cls_mode == "patch_mean_concat_p_t_m":
            # P(frame_t) appearance ⊕ M(frame_t, frame_tk) motion → [B, 2D] (1536).
            # p_t_p_tk와 달리 M(motion) stream을 직접 사용 → Parvo 양 stream 통합 readout.
            # 주의: baseline엔 M stream 없어 matched-mode 비교 아님(best-mode-per-encoder).
            p_t = encoder.preprocessing.compute_p_channel(img_t)
            feat_p = encoder._encode_p_unmasked(p_t)[:, 1:].mean(dim=1)       # [B, D] appearance(t)
            m_chan = encoder.preprocessing.compute_m_channel(img_t, img_tk)
            feat_m = encoder._encode_m_unmasked(m_chan)[:, 1:].mean(dim=1)    # [B, D] motion(t,tk)
            return torch.cat([feat_p, feat_m], dim=-1)                        # [B, 2D]
        elif cls_mode == "patch_mean_m":
            # M encoder: frame_t,tk 조합(ΔL motion channel) patch_mean (motion)
            m_chan = encoder.preprocessing.compute_m_channel(img_t, img_tk)
            return encoder._encode_m_unmasked(m_chan)[:, 1:].mean(dim=1)  # [B, D]
        elif cls_mode == "patch_mean_routed_tk":
            # frame_t → P enc(anchor) → M routing → *추론된* frame_t+k repr의 patch_mean (mean only, NOT concat).
            # 실제 P(t+k)가 아니라 라우팅 예측을 씀 → "라우팅이 motion/action 정보를 담나" 검증.
            # 라우팅은 viz col7과 동일(full anchor _vjepa_p_one_segment).
            p_t = encoder.preprocessing.compute_p_channel(img_t)
            p_tk = encoder.preprocessing.compute_p_channel(img_tk)  # 함수 시그니처용(target), 미사용
            m_chan = encoder.preprocessing.compute_m_channel(img_t, img_tk)
            m_local = encoder._encode_m_unmasked(m_chan)
            _, pred_tk, _ = encoder._vjepa_p_one_segment(m_chan, p_t, p_tk, m_local_routing=m_local)
            return pred_tk[:, 1:].mean(dim=1)  # [B, D] routed prediction patch_mean (768)
        else:
            raise ValueError(f"parvo: unsupported cls_mode {cls_mode}")

    elif name == "videomae":
        # tubelet_size=2가 (frame_t, frame_tk)를 시공간 patch 1개로 묶음 → 단일 frame
        # representation을 뽑으려면 같은 frame 2회 복제 forward (paper_experiments_plan §C7).
        if cls_mode == "patch_mean_concat_p_t_p_tk":
            img_t = pixel_values[:, :3]
            img_tk = pixel_values[:, 3:]
            pv_t_only = torch.cat([img_t, img_t], dim=1)    # [B, 6, H, W] same-frame replica
            pv_tk_only = torch.cat([img_tk, img_tk], dim=1)
            patch_t = encoder(pv_t_only).mean(dim=1)        # [B, D]
            patch_tk = encoder(pv_tk_only).mean(dim=1)
            return torch.cat([patch_t, patch_tk], dim=-1)   # [B, 2D]
        # VideoMAE는 CLS 토큰 없음 → patch mean pooling (default: paired forward)
        patch_emb = encoder(pixel_values)  # [B, N, D]
        return patch_emb.mean(dim=1)  # [B, D]

    elif name == "siammae":
        # SiamMAE 인코더는 [B,6] 중 current(뒤 3ch)만 인코딩 → frame replica로 단일 프레임 추출.
        img_t = pixel_values[:, :3]
        img_tk = pixel_values[:, 3:]
        if cls_mode == "patch_mean_concat_p_t_p_tk":
            feat_t = encoder(torch.cat([img_t, img_t], dim=1)).mean(dim=1)     # [B, D]
            feat_tk = encoder(torch.cat([img_tk, img_tk], dim=1)).mean(dim=1)
            return torch.cat([feat_t, feat_tk], dim=-1)                        # [B, 2D]
        # default: anchor frame_t patch mean
        return encoder(torch.cat([img_t, img_t], dim=1)).mean(dim=1)          # [B, D]

    elif name == "clip":
        img_t = pixel_values[:, :3]   # [B, 3, H, W]
        img_t1 = pixel_values[:, 3:]  # [B, 3, H, W]

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=pixel_values.device).view(1, 3, 1, 1)
        img_t_norm = (img_t - mean) / std
        img_t1_norm = (img_t1 - mean) / std

        hidden_t = encoder(pixel_values=img_t_norm).last_hidden_state   # [B, N+1, 768]
        hidden_t1 = encoder(pixel_values=img_t1_norm).last_hidden_state

        if cls_mode == "patch_mean":
            out_t = hidden_t[:, 1:].mean(dim=1)    # patch mean [B, 768]
            out_t1 = hidden_t1[:, 1:].mean(dim=1)
        else:
            out_t = hidden_t[:, 0]    # CLS [B, 768]
            out_t1 = hidden_t1[:, 0]
        return torch.cat([out_t, out_t1], dim=-1)  # [B, 1536]

    elif name == "dinov2":
        img_t = pixel_values[:, :3]
        img_t1 = pixel_values[:, 3:]

        mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device).view(1, 3, 1, 1)
        img_t_norm = (img_t - mean) / std
        img_t1_norm = (img_t1 - mean) / std

        hidden_t = encoder(pixel_values=img_t_norm).last_hidden_state   # [B, N+1, 768]
        hidden_t1 = encoder(pixel_values=img_t1_norm).last_hidden_state

        if cls_mode == "patch_mean":
            out_t = hidden_t[:, 1:].mean(dim=1)
            out_t1 = hidden_t1[:, 1:].mean(dim=1)
        else:
            out_t = hidden_t[:, 0]
            out_t1 = hidden_t1[:, 0]
        return torch.cat([out_t, out_t1], dim=-1)  # [B, 1536]

    elif name == "siglip":
        img_t = pixel_values[:, :3]
        img_t1 = pixel_values[:, 3:]

        # SigLIP 공식 preprocessing: mean/std = 0.5
        mean = torch.tensor([0.5, 0.5, 0.5], device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=pixel_values.device).view(1, 3, 1, 1)
        img_t_norm = (img_t - mean) / std
        img_t1_norm = (img_t1 - mean) / std

        hidden_t = encoder(pixel_values=img_t_norm).last_hidden_state   # [B, N, 768] (no CLS)
        hidden_t1 = encoder(pixel_values=img_t1_norm).last_hidden_state

        # SigLIP은 CLS 토큰 없음 → patch mean pool
        out_t = hidden_t.mean(dim=1)
        out_t1 = hidden_t1.mean(dim=1)
        return torch.cat([out_t, out_t1], dim=-1)  # [B, 1536]

    elif name == "vc1":
        img_t = pixel_values[:, :3]
        img_t1 = pixel_values[:, 3:]

        # VC-1 공식 preprocessing: ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device).view(1, 3, 1, 1)
        img_t_norm = (img_t - mean) / std
        img_t1_norm = (img_t1 - mean) / std

        # VC-1은 CLS token 출력 [B, 768]
        out_t = encoder(img_t_norm)
        out_t1 = encoder(img_t1_norm)
        return torch.cat([out_t, out_t1], dim=-1)  # [B, 1536]

    elif name == "vjepa2":
        img_t = pixel_values[:, :3]   # [B, 3, H, W]
        img_t1 = pixel_values[:, 3:]

        # V-JEPA 2.1 입력: [B, C, T, H, W] — 224→384 resize 필요
        # probing dataset은 224x224 → 384x384로 resize
        img_t_384 = F.interpolate(img_t, size=(384, 384), mode="bilinear", align_corners=False)
        img_t1_384 = F.interpolate(img_t1, size=(384, 384), mode="bilinear", align_corners=False)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device).view(1, 3, 1, 1)
        img_t_norm = (img_t_384 - mean) / std
        img_t1_norm = (img_t1_384 - mean) / std

        # [B, C, T=2, H, W]
        video = torch.stack([img_t_norm, img_t1_norm], dim=2)
        tokens = encoder(video)  # [B, N, 768]
        # patch mean pool 후 단일 embedding (2프레임 동시 처리이므로 concat 불필요)
        out = tokens.mean(dim=1)  # [B, 768]
        # 다른 모델과 차원 맞추기: 2배 (768*2=1536은 concat 기반 모델용)
        # V-JEPA는 2프레임을 동시에 보므로 768로 충분, 하지만 공정 비교를 위해 반복
        return torch.cat([out, out], dim=-1)  # [B, 1536]

    elif name == "videomae-official":
        # VideoMAE-official: [B, T=2, C, H, W] 형태
        img_t = pixel_values[:, :3]
        img_t1 = pixel_values[:, 3:]

        mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device).view(1, 3, 1, 1)
        img_t_norm = (img_t - mean) / std
        img_t1_norm = (img_t1 - mean) / std

        # [B, T=2, C, H, W]
        video = torch.stack([img_t_norm, img_t1_norm], dim=1)
        out = encoder(video).last_hidden_state  # [B, 196, 768]
        return out.mean(dim=1)  # [B, 768]

    else:
        raise ValueError(f"Unknown encoder: {name}")


def encode_batch_tokens(encoder, name: str, pixel_values: torch.Tensor, cls_mode: str) -> torch.Tensor:
    """Attentive-pooling 경로: mean pool 없이 patch 토큰을 그대로 반환.

    반환: [B, n_streams * n_patch, D]  (stream 순서대로 token-dim concat)
      · 단일 stream (attentive_p_t / attentive_m / attentive): [B, n_patch, D]
      · 2 stream  (attentive_concat_*):                       [B, 2*n_patch, D]
    AttentivePoolProbe가 stream별 learnable query로 pool. 인코더는 frozen → 토큰 1회 캐싱.
    encode_batch의 mean-pool 모드와 1:1 대응 (matched 비교용).
    """
    if name == "parvo":
        img_t = pixel_values[:, :3]
        img_tk = pixel_values[:, 3:]
        if cls_mode == "attentive_p_t":                       # 1 stream: P(t) 단독 (input-only null anchor)
            p_t = encoder.preprocessing.compute_p_channel(img_t)
            return encoder._encode_p_unmasked(p_t)[:, 1:]
        elif cls_mode == "attentive_concat_p_t_p_tk":         # 2 stream: P(t), P(t+k) (배포 P readout)
            p_t = encoder.preprocessing.compute_p_channel(img_t)
            p_tk = encoder.preprocessing.compute_p_channel(img_tk)
            tok_t = encoder._encode_p_unmasked(p_t)[:, 1:]
            tok_tk = encoder._encode_p_unmasked(p_tk)[:, 1:]
            return torch.cat([tok_t, tok_tk], dim=1)
        elif cls_mode == "attentive_m":                       # 1 stream: M(t,t+k) motion 진단
            m_chan = encoder.preprocessing.compute_m_channel(img_t, img_tk)
            return encoder._encode_m_unmasked(m_chan)[:, 1:]
        elif cls_mode == "attentive_concat_p_m":              # 2 stream: P(t) ⊕ M(t,t+k) — M 기여 판정자
            p_t = encoder.preprocessing.compute_p_channel(img_t)
            tok_p = encoder._encode_p_unmasked(p_t)[:, 1:]
            m_chan = encoder.preprocessing.compute_m_channel(img_t, img_tk)
            tok_m = encoder._encode_m_unmasked(m_chan)[:, 1:]
            return torch.cat([tok_p, tok_m], dim=1)
        else:
            raise ValueError(f"parvo: unsupported attentive cls_mode {cls_mode}")

    elif name == "siammae":
        img_t = pixel_values[:, :3]
        img_tk = pixel_values[:, 3:]
        if cls_mode == "attentive":                           # 1 stream: anchor frame_t
            return encoder(torch.cat([img_t, img_t], dim=1))
        elif cls_mode == "attentive_concat_p_t_p_tk":         # 2 stream: frame_t, frame_t+k (matched to parvo P)
            tok_t = encoder(torch.cat([img_t, img_t], dim=1))
            tok_tk = encoder(torch.cat([img_tk, img_tk], dim=1))
            return torch.cat([tok_t, tok_tk], dim=1)
        else:
            raise ValueError(f"siammae: unsupported attentive cls_mode {cls_mode}")

    elif name == "videomae":
        if cls_mode == "attentive_concat_p_t_p_tk":           # 2 stream: same-frame replica forward
            img_t = pixel_values[:, :3]
            img_tk = pixel_values[:, 3:]
            tok_t = encoder(torch.cat([img_t, img_t], dim=1))
            tok_tk = encoder(torch.cat([img_tk, img_tk], dim=1))
            return torch.cat([tok_t, tok_tk], dim=1)
        elif cls_mode == "attentive":                         # 1 stream: paired forward
            return encoder(pixel_values)
        else:
            raise ValueError(f"videomae: unsupported attentive cls_mode {cls_mode}")

    else:
        raise ValueError(f"encode_batch_tokens: encoder '{name}' not wired for attentive modes")


# ============================================================================
# 3. Probe definitions
# ============================================================================

class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int = ACTION_DIM):
        super().__init__()
        self.linear = nn.Linear(embed_dim, action_dim)

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256, action_dim: int = ACTION_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class AttentivePoolProbe(nn.Module):
    """Minimal attentive-pooling probe (capacity ≈ linear probe + per-stream query).

    stream별 learnable query 1개(single-head)로 patch 토큰을 softmax-weighted pool →
    stream pooled vector concat → linear head. M/P는 별도 stream = 별도 query (구조적 분리).
    큰 MLP를 피해 capacity를 linear와 맞춰 'structure vs probe capacity' 혼동을 차단
    (input-only null anchor 비교가 깨끗해짐).

    입력 x: [B, n_streams * n_patch, D]  (encode_batch_tokens 출력)
    """
    def __init__(self, embed_dim: int, n_streams: int = 1, n_patch: int = 196,
                 action_dim: int = ACTION_DIM):
        super().__init__()
        self.n_streams = n_streams
        self.n_patch = n_patch
        self.embed_dim = embed_dim
        self.query = nn.Parameter(torch.randn(n_streams, embed_dim) * 0.02)
        self.head = nn.Linear(n_streams * embed_dim, action_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.n_streams, self.n_patch, self.embed_dim)
        attn = torch.einsum("bsnd,sd->bsn", x, self.query) * self.scale  # [B, S, N]
        attn = attn.softmax(dim=-1)
        pooled = torch.einsum("bsn,bsnd->bsd", attn, x)                  # [B, S, D]
        pooled = pooled.reshape(B, self.n_streams * self.embed_dim)
        return self.head(pooled)


# ============================================================================
# 4. Metrics
# ============================================================================

def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute R², MSE, Cosine Similarity.

    Args:
        predictions: [N, 18]
        targets: [N, 18]
    """
    # MSE
    mse = np.mean((predictions - targets) ** 2)

    # R² (coefficient of determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # Cosine similarity (per-sample, then average)
    pred_norm = np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8
    tgt_norm = np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8
    cos_sim = np.mean(np.sum(predictions * targets, axis=1) / (pred_norm.squeeze() * tgt_norm.squeeze()))

    # Per-joint R²
    per_joint_r2 = {}
    for i, joint in enumerate(TARGET_JOINTS):
        j_pred = predictions[:, i * 3 : (i + 1) * 3]
        j_tgt = targets[:, i * 3 : (i + 1) * 3]
        ss_res_j = np.sum((j_tgt - j_pred) ** 2)
        ss_tot_j = np.sum((j_tgt - j_tgt.mean(axis=0)) ** 2)
        per_joint_r2[joint] = float(1 - ss_res_j / (ss_tot_j + 1e-8))

    return {
        "r2": float(r2),
        "mse": float(mse),
        "cosine_sim": float(cos_sim),
        "per_joint_r2": per_joint_r2,
    }


# ============================================================================
# 5. Training / Evaluation loop
# ============================================================================

def extract_embeddings(encoder, encoder_name, dataloader, device, cls_mode="average"):
    """Extract frozen embeddings for the entire dataset.

    attentive_* 모드: mean-pool 대신 patch 토큰 [N, n_streams*n_patch, D]를 1회 캐싱
    (frozen 인코더라 deterministic → probe epoch마다 재forward 불필요).
    그 외: 기존 pooled 벡터 [N, D].
    """
    is_attentive = cls_mode.startswith("attentive")
    # pre-allocation: 토큰 캐시는 거대(예: 180k×392×384). torch.cat은 list+결과를 동시
    # 보유해 peak ~2× (실측 MaxRSS 279GB) → OOM. 전체 텐서를 1회 할당해 index로 채워
    # peak = 텐서 크기로 고정. attentive는 fp16(frozen feature라 무손실).
    n_total = len(dataloader.dataset)
    embeddings = None
    all_actions = []
    idx = 0

    encoder.eval()
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        actions = batch["action"]

        if is_attentive:
            emb = encode_batch_tokens(encoder, encoder_name, pixel_values, cls_mode=cls_mode).half().cpu()
        else:
            emb = encode_batch(encoder, encoder_name, pixel_values, cls_mode=cls_mode).cpu()

        if embeddings is None:
            embeddings = torch.empty((n_total, *emb.shape[1:]), dtype=emb.dtype)
        b = emb.shape[0]
        embeddings[idx:idx + b] = emb
        idx += b
        all_actions.append(actions)

    embeddings = embeddings[:idx]  # drop_last 미사용이라 보통 n_total과 일치
    actions = torch.cat(all_actions, dim=0)  # [N, 18]
    return embeddings, actions


def train_probe(
    probe: nn.Module,
    train_emb: torch.Tensor,
    train_act: torch.Tensor,
    eval_emb: torch.Tensor,
    eval_act: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """Train the linear/MLP probe on pre-extracted embeddings."""
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    # Create simple tensor datasets
    train_dataset = torch.utils.data.TensorDataset(train_emb, train_act)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_r2 = -float("inf")
    best_metrics = None

    for epoch in range(1, epochs + 1):
        # Train
        probe.train()
        epoch_loss = 0
        n_batches = 0
        for emb_batch, act_batch in train_loader:
            emb_batch = emb_batch.to(device).float()  # fp16 캐시면 GPU에서 float 복원
            act_batch = act_batch.to(device)

            pred = probe(emb_batch)
            loss = F.mse_loss(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Evaluate (배치 처리 — attentive 토큰 eval 전체를 GPU에 한 번에 올리면 OOM)
        probe.eval()
        eval_preds = []
        with torch.no_grad():
            for i in range(0, len(eval_emb), batch_size):
                chunk = eval_emb[i:i + batch_size].to(device).float()
                eval_preds.append(probe(chunk).cpu())
        eval_pred = torch.cat(eval_preds, dim=0).numpy()
        eval_targets = eval_act.numpy()

        metrics = compute_metrics(eval_pred, eval_targets)
        metrics["train_mse"] = avg_train_loss

        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_metrics = metrics.copy()

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train MSE: {avg_train_loss:.6f} | "
            f"Eval R²: {metrics['r2']:.4f} | "
            f"MSE: {metrics['mse']:.6f} | "
            f"Cos: {metrics['cosine_sim']:.4f}"
        )

    return best_metrics


# ============================================================================
# 6. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Action Probing Experiment")

    parser.add_argument("--encoder", type=str, required=True,
                        choices=["two-stream", "parvo", "videomae", "clip", "dinov2", "siglip", "vc1", "vjepa2", "videomae-official"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Encoder checkpoint path (required for two-stream, videomae)")
    parser.add_argument("--egodex-root", type=str, default="/mnt/data/egodex",
                        help="EgoDex 원본 경로 (MP4+HDF5)")
    parser.add_argument("--frames-root", type=str, default="/mnt/data/egodex_frames",
                        help="추출된 프레임 경로 (JPG)")
    parser.add_argument("--egodex-split", type=str, default="part1",
                        help="EgoDex split to use (e.g. part1, part4)")
    parser.add_argument("--gap", type=int, default=1,
                        help="Frame gap for action delta (default: 1)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Probing epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Probe learning rate (default: 1e-3)")
    parser.add_argument("--probe", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Probe type (default: linear)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Limit number of videos (for debugging)")
    parser.add_argument("--cls-mode", type=str, default="average",
                        choices=["average", "concat", "m_only", "p_only",
                                 "patch_mean", "patch_mean_m", "patch_mean_p",
                                 "patch_mean_concat",
                                 "patch_mean_p_t",
                                 "patch_mean_concat_p_t_p_tk", "patch_mean_concat_p_t_m",
                                 "patch_mean_routed_tk",
                                 "cls_p_bg", "cls_p_motion", "all_cls_concat",
                                 # attentive-pooling (readout 축): mean 모드와 1:1, AttentivePoolProbe 경로
                                 "attentive_p_t", "attentive_concat_p_t_p_tk",
                                 "attentive_m", "attentive_concat_p_m", "attentive"],
                        help="Two-Stream embedding 추출 방식 (default: average). "
                             "patch_mean_concat_p_t_p_tk: videomae §C7 catalyst evidence. "
                             "attentive_*: mean 대신 stream별 learnable query pool (AttentivePoolProbe)")
    parser.add_argument("--depth", type=int, default=12,
                        help="Two-Stream transformer depth (default: 12)")
    parser.add_argument("--num-stages", type=int, default=3,
                        help="Two-Stream CLS exchange stages (default: 3)")
    parser.add_argument("--output-dir", type=str, default="data/probing_results",
                        help="Output directory")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Encoder: {args.encoder}")
    print(f"Probe: {args.probe}")
    print(f"Checkpoint: {args.checkpoint or '(pretrained)'}")
    print(f"EgoDex split: {args.egodex_split}")
    print(f"Gap: {args.gap}")
    print(f"CLS mode: {args.cls_mode}")

    # ---- 1. Load encoder ----
    print("\n" + "=" * 60)
    print("Loading encoder...")
    print("=" * 60)
    t0 = time.time()
    encoder, embed_dim = load_encoder(
        args.encoder, args.checkpoint, device,
        cls_mode=args.cls_mode, depth=args.depth, num_stages=args.num_stages,
    )
    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    print(f"Encoder loaded in {time.time() - t0:.1f}s, embed_dim={embed_dim}")

    # ---- 2. Build datasets ----
    print("\n" + "=" * 60)
    print("Building datasets...")
    print("=" * 60)
    t0 = time.time()
    train_ds, eval_ds = build_datasets(args.egodex_root, args.frames_root, args.egodex_split, gap=args.gap, max_videos=args.max_videos)
    print(f"Datasets built in {time.time() - t0:.1f}s")

    if len(train_ds) == 0 or len(eval_ds) == 0:
        print("ERROR: No valid samples found. Check data path and HDF5 files.")
        sys.exit(1)

    # ---- 3. Extract embeddings ----
    print("\n" + "=" * 60)
    print("Extracting embeddings (frozen encoder)...")
    print("=" * 60)
    t0 = time.time()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_emb, train_act = extract_embeddings(encoder, args.encoder, train_loader, device, cls_mode=args.cls_mode)
    eval_emb, eval_act = extract_embeddings(encoder, args.encoder, eval_loader, device, cls_mode=args.cls_mode)

    print(f"Extracted in {time.time() - t0:.1f}s")
    print(f"  Train: {train_emb.shape} embeddings, {train_act.shape} actions")
    print(f"  Eval:  {eval_emb.shape} embeddings, {eval_act.shape} actions")

    # Action statistics
    act_mean = train_act.mean(dim=0)
    act_std = train_act.std(dim=0)
    print(f"  Action mean: {act_mean.abs().mean():.6f}")
    print(f"  Action std:  {act_std.mean():.6f}")

    # ---- 4. Train probe ----
    print("\n" + "=" * 60)
    print(f"Training {args.probe} probe...")
    print("=" * 60)

    if args.cls_mode.startswith("attentive"):
        # 캐시된 토큰 [N, S*n_patch, D] → stream 수는 cls_mode로 결정 (concat=2, 그외=1)
        n_streams = 2 if "concat" in args.cls_mode else 1
        base_dim = train_emb.shape[-1]
        n_patch = train_emb.shape[1] // n_streams
        probe = AttentivePoolProbe(base_dim, n_streams=n_streams, n_patch=n_patch)
        print(f"AttentivePoolProbe: D={base_dim}, n_streams={n_streams}, n_patch={n_patch}")
    elif args.probe == "linear":
        probe = LinearProbe(embed_dim)
    else:
        probe = MLPProbe(embed_dim)

    print(f"Probe params: {sum(p.numel() for p in probe.parameters()):,}")

    best_metrics = train_probe(
        probe=probe,
        train_emb=train_emb,
        train_act=train_act,
        eval_emb=eval_emb,
        eval_act=eval_act,
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
    print(f"Probe:      {args.probe}")
    print(f"Split:      {args.egodex_split}")
    print(f"R²:         {best_metrics['r2']:.4f}  {'PASS' if best_metrics['r2'] > 0.7 else 'FAIL'} (threshold: 0.7)")
    print(f"MSE:        {best_metrics['mse']:.6f}")
    print(f"Cosine Sim: {best_metrics['cosine_sim']:.4f}")
    print(f"\nPer-joint R²:")
    for joint, r2 in best_metrics["per_joint_r2"].items():
        print(f"  {joint:30s}: {r2:.4f}")

    # ---- 6. Save results ----
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

    result_path = output_dir / f"probe_{args.encoder}_{args.cls_mode}_gap{args.gap}_{args.egodex_split}_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    main()
