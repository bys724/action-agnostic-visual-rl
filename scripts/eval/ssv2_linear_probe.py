#!/usr/bin/env python
"""SSv2 2-frame linear probe — 경로 1 (docs/correspondence_eval_plan.md §7).

Artifact-완화 설계: frozen encoder + **mean-pool + linear** 한정 (attentive/concat
token probe 금지 — 2회 철회 실패모드). matched 2-frame 규약: 전 encoder 동일
pair(t, t+gap≈1s = 12frames @ 12fps)·해상도 224·mean readout.

방향성 control: validation은 pair 순서 반전 feature도 추출 → 학습된 probe의
acc 하락폭 = motion 방향 정보 사용의 직접 증거 (SSv2는 방향 민감 태스크).

gate 사전 등록 (§7, 2026-07-12): ① acc(p_t_m) − acc(p_t_p_tk) ≥ +2%p (주)
② max(CoMP) ≥ max(DINOv2, SigLIP) − 2%p (부). 둘 다 충족 시만 보고.

디코드 결정 (2026-07-12 샘플 테스트): cv2 VP9 정상, fps=12.0 고정,
meta frame count == 실제 (신뢰 가능), 240×가변폭 → 224² resize.
라벨: template의 '[',']' 제거 == labels.json 키 (train/val 100% 매치 검증).

사용:
  # 1) feature 추출 (train+val+val_rev) + 2) linear probe 까지 한 번에
  python scripts/eval/ssv2_linear_probe.py --encoder parvo --parvo-mode p_t_m \
      --checkpoint <ckpt> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

cv2.setNumThreads(0)  # DataLoader worker × cv2 내부 스레드 경합 방지

N_CLASSES = 174
FPS_GAP = 12          # 1.0s @ 12fps (EgoDex 학습 분포 정합)
IMG_SIZE = 224        # 🔴 parity: 전 encoder 동일
PAIRS = {"train": 2, "validation": 3}


# ─────────────────────────────────────────────────────────────────────────
# 데이터셋: webm → 결정론적 pair (t, t+gap)
# ─────────────────────────────────────────────────────────────────────────

class SSv2PairDataset(Dataset):
    def __init__(self, root: str, split: str, ann_dir: str | None = None):
        """ann_dir: 대체 split 어노테이션 dir (예: Something-Else compositional —
        물체 train/val 완전 disjoint). None = 공식 split (root의 json)."""
        self.video_dir = Path(root) / "20bn-something-something-v2"
        ad = Path(ann_dir) if ann_dir else Path(root)
        labels = json.load(open(ad / "labels.json"))
        anns = json.load(open(ad / f"{split}.json"))
        self.items = [(d["id"], int(labels[d["template"].replace("[", "").replace("]", "")]))
                      for d in anns]
        self.n_pairs = PAIRS[split]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        clip_id, label = self.items[i]
        cap = cv2.VideoCapture(str(self.video_dir / f"{clip_id}.webm"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 샘플 테스트: meta==real 신뢰
        gap = min(FPS_GAP, max(n - 1, 1))
        # 결정론적 시작점: [0, n-1-gap] 균등 분할 (RNG 불요 = 재현성)
        starts = np.linspace(0, max(n - 1 - gap, 0), self.n_pairs).round().astype(int)
        wanted = sorted(set(starts.tolist()) | set((starts + gap).tolist()))
        frames = {}
        idx = 0
        while wanted and idx <= wanted[-1]:
            ok, img = cap.read()
            if not ok:
                break
            if idx in wanted:
                img = cv2.resize(img[:, :, ::-1], (IMG_SIZE, IMG_SIZE))  # BGR→RGB
                frames[idx] = img
            idx += 1
        cap.release()
        if not frames:  # 손상 파일 가드: 검정 프레임 (한 파일이 잡 전체를 죽이지 않게)
            print(f"  ⚠️ decode 실패: {clip_id}.webm → zero frames", flush=True)
            pairs = np.zeros((self.n_pairs, 2, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            return torch.from_numpy(pairs), label, int(clip_id)
        last = max(frames)
        pairs = np.stack([
            np.stack([frames.get(s, frames[min(frames, key=lambda k: abs(k - s))]),
                      frames.get(s + gap, frames[last])])
            for s in starts
        ])  # (P, 2, H, W, 3) uint8
        return torch.from_numpy(pairs), label, int(clip_id)


def collate(batch):
    pairs = torch.cat([b[0] for b in batch])              # (ΣP, 2, H, W, 3)
    labels = torch.tensor([b[1] for b in batch])
    clip_ids = torch.tensor([b[2] for b in batch])
    n_pairs = torch.tensor([b[0].shape[0] for b in batch])
    return pairs, labels, clip_ids, n_pairs


# ─────────────────────────────────────────────────────────────────────────
# Encoder → mean-pool 2-frame feature (전부 기존 규약 재사용)
# ─────────────────────────────────────────────────────────────────────────

class MeanFeatureExtractor:
    """readout: mean = patch-mean (기본, gate 프로토콜) / meanmax = mean⊕max.

    meanmax는 readout-병목 정량화용 (파라미터 0 유지 = artifact 규율 내):
    M 신호는 공간 국소(CALVIN attn>mean 관찰)라 mean이 희석 가능 → max가 국소 peak 보존.
    전 encoder 동일 적용 시 parity 유지, mean 결과와 나란히 비교.
    """

    def __init__(self, encoder: str, checkpoint: str | None, parvo_mode: str, device,
                 readout: str = "mean"):
        self.encoder, self.mode, self.device = encoder, parvo_mode, device
        assert readout in ("mean", "meanmax")
        self.readout = readout
        if encoder == "parvo":
            from scripts.eval.probe_action_libero import build_parvo_encoder
            self.model = build_parvo_encoder(checkpoint, device)
        elif encoder == "videomae-vla":
            from scripts.eval.probe_action_libero import build_videomae_token_encoder
            self.model = build_videomae_token_encoder(checkpoint, device)
        elif encoder in ("dinov2", "siglip"):
            if encoder == "dinov2":
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained("facebook/dinov2-base")
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            else:
                from transformers import SiglipVisionModel
                self.model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
                mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            self.model.to(device).eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self._mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
            self._std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        else:
            raise ValueError(encoder)

    def _pool(self, tok: torch.Tensor) -> torch.Tensor:
        """(n, n_patch, D) → mean: (n, D) / meanmax: (n, 2D)."""
        if self.readout == "mean":
            return tok.mean(dim=1)
        return torch.cat([tok.mean(dim=1), tok.max(dim=1).values], dim=-1)

    @torch.no_grad()
    def _hf_tok(self, x: torch.Tensor) -> torch.Tensor:
        hid = self.model(pixel_values=(x - self._mean) / self._std).last_hidden_state
        return hid[:, 1:] if self.encoder == "dinov2" else hid

    @torch.no_grad()
    def pair_features(self, prev: torch.Tensor, curr: torch.Tensor,
                      with_reverse: bool = False):
        """prev/curr (N,3,H,W)[0,1] CPU → (fwd, rev|None) 각 (N,D') fp16.

        학습효율 개선 (mean 런 병목 분석: HF fp32 forward가 지배·rev 전체 재인코딩 낭비):
        - autocast fp16 forward (V100 tensor core). caveat: 게이트 mean 런은 fp32 —
          fp16 feature 노이즈는 probing 스케일에서 미미하나 정밀 비교 시 주의.
        - 반전 feature: appearance 계열(순서 swap = concat 순서만 반전)은 **토큰 재사용,
          재인코딩 0**. 진짜 재인코딩은 p_t_m의 M-half(ΔL 부호 반전)뿐.
        """
        outs, outs_rev = [], []
        for s in range(0, prev.shape[0], 256):
            p = prev[s:s+256].to(self.device)
            c = curr[s:s+256].to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=self.device.type == "cuda"):
                if self.encoder == "parvo":
                    enc_p = lambda x: self.model._encode_p_unmasked(
                        self.model.preprocessing.compute_p_channel(x))[:, 1:]
                    enc_m = lambda x, y: self.model._encode_m_unmasked(
                        self.model.preprocessing.compute_m_channel(x, y))[:, 1:]
                    if self.mode == "m_only":  # 단독 진단 (1-stream, 차원 caveat)
                        outs.append(self._pool(enc_m(p, c)).cpu())
                        if with_reverse:
                            outs_rev.append(self._pool(enc_m(c, p)).cpu())
                        continue
                    if self.mode == "p_t_only":  # 단독 진단 — 반전 = 기준 frame이 c로 바뀜
                        outs.append(self._pool(enc_p(p)).cpu())
                        if with_reverse:
                            outs_rev.append(self._pool(enc_p(c)).cpu())
                        continue
                    a = self._pool(enc_p(p))
                    if self.mode == "p_t_m":
                        b = self._pool(enc_m(p, c))
                        outs.append(torch.cat([a, b], -1).cpu())
                        if with_reverse:  # P(c)·M(c,p) 모두 신규 — 재사용 불가
                            outs_rev.append(torch.cat(
                                [self._pool(enc_p(c)), self._pool(enc_m(c, p))], -1).cpu())
                        continue
                    b = self._pool(enc_p(c))
                elif self.encoder == "videomae-vla":
                    a = self._pool(self.model(torch.cat([p, p], dim=1)))  # self-pair (§C7)
                    b = self._pool(self.model(torch.cat([c, c], dim=1)))
                else:  # dinov2/siglip: frame별 인코딩 (single_frame adapter 규약)
                    a, b = self._pool(self._hf_tok(p)), self._pool(self._hf_tok(c))
            outs.append(torch.cat([a, b], -1).cpu())
            if with_reverse:  # 순서 swap = 토큰 재사용 (재인코딩 0)
                outs_rev.append(torch.cat([b, a], -1).cpu())
        fwd = torch.cat(outs).half()
        rev = torch.cat(outs_rev).half() if with_reverse else None
        return fwd, rev


def extract_split(extractor, root, split, with_reverse, batch_clips, workers,
                  ann_dir=None, log_every=200):
    ds = SSv2PairDataset(root, split, ann_dir=ann_dir)
    dl = DataLoader(ds, batch_size=batch_clips, num_workers=workers,
                    collate_fn=collate, shuffle=False)
    feats, feats_rev, labels, clip_ids = [], [], [], []
    t0 = time.time()
    for bi, (pairs, lab, cid, n_pairs) in enumerate(dl):
        x = pairs.permute(0, 1, 4, 2, 3).float().div_(255.0)  # (ΣP, 2, 3, H, W)
        prev, curr = x[:, 0], x[:, 1]
        f, fr = extractor.pair_features(prev, curr, with_reverse=with_reverse)
        feats.append(f)
        if with_reverse:
            feats_rev.append(fr)
        labels.append(torch.repeat_interleave(lab, n_pairs))
        clip_ids.append(torch.repeat_interleave(cid, n_pairs))
        if (bi + 1) % log_every == 0:
            done = (bi + 1) * batch_clips
            print(f"  [{split}] {done}/{len(ds)} clips ({time.time()-t0:.0f}s)", flush=True)
    return {
        "features": torch.cat(feats).numpy(),
        "features_rev": torch.cat(feats_rev).numpy() if with_reverse else None,
        "labels": torch.cat(labels).numpy(),
        "clip_ids": torch.cat(clip_ids).numpy(),
    }


# ─────────────────────────────────────────────────────────────────────────
# Linear probe (mean-pool feature → 174-way, artifact-완화 readout)
# ─────────────────────────────────────────────────────────────────────────

def train_probe(tr, va, device, epochs=20, bs=8192, lr=1e-3):
    X = torch.from_numpy(tr["features"]).float()
    y = torch.from_numpy(tr["labels"]).long()
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
    X = ((X - mu) / sd).to(device); y = y.to(device)
    probe = nn.Linear(X.shape[1], N_CLASSES).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s+bs]
            loss = F.cross_entropy(probe(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * idx.numel()
        sched.step()
        print(f"  probe ep{ep+1}/{epochs} loss={tot/n:.4f}", flush=True)

    def evaluate(feats_np):
        Xv = ((torch.from_numpy(feats_np).float() - mu) / sd).to(device)
        with torch.no_grad():
            logits = torch.cat([probe(Xv[s:s+bs]) for s in range(0, Xv.shape[0], bs)])
        # 클립 레벨: pair logit 평균
        cids = torch.from_numpy(va["clip_ids"])
        uniq, inv = torch.unique(cids, return_inverse=True)
        agg = torch.zeros(len(uniq), N_CLASSES).index_add_(0, inv, logits.cpu())
        clip_label = torch.zeros(len(uniq), dtype=torch.long).index_copy_(
            0, inv, torch.from_numpy(va["labels"]).long())
        top = agg.topk(5, dim=1).indices
        top1 = (top[:, 0] == clip_label).float().mean().item()
        top5 = (top == clip_label[:, None]).any(1).float().mean().item()
        return top1, top5

    return evaluate


def main():
    ap = argparse.ArgumentParser("SSv2 2-frame linear probe")
    ap.add_argument("--encoder", required=True,
                    choices=["parvo", "dinov2", "siglip", "videomae-vla"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--parvo-mode", default="p_t_m",
                    choices=["p_t_m", "p_t_p_tk", "m_only", "p_t_only"],
                    help="m_only/p_t_only = 단독 stream 진단 (⚠️ 1-stream = 절반 차원, "
                         "2-stream 조건과 probe 용량 다름 — 차이 비교는 p_t_m−p_t_p_tk가 정규)")
    ap.add_argument("--readout", default="mean", choices=["mean", "meanmax"],
                    help="meanmax = readout-병목 정량화 (파라미터 0, 전 encoder 동일 적용)")
    ap.add_argument("--ssv2-root", default="/proj/external_group/mrg/datasets/ssv2")
    ap.add_argument("--ann-dir", default=None,
                    help="대체 split (예: splits_something_else/compositional — 물체 disjoint)")
    ap.add_argument("--max-clips", type=int, default=0, help="sanity용 제한 (양 split 공통)")
    ap.add_argument("--batch-clips", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tag = args.encoder + (f"-{args.parvo_mode}" if args.encoder == "parvo" else "")
    print(f"[ssv2] {tag} device={device.type} gap={FPS_GAP}f img={IMG_SIZE} "
          f"pairs={PAIRS} probe=linear(mean-pool)", flush=True)

    extractor = MeanFeatureExtractor(args.encoder, args.checkpoint, args.parvo_mode,
                                     device, readout=args.readout)

    if args.max_clips:  # sanity: 데이터셋 절단
        SSv2PairDataset_orig = SSv2PairDataset.__init__
        def _init(self, root, split, ann_dir=None):
            SSv2PairDataset_orig(self, root, split, ann_dir=args.ann_dir)
            self.items = self.items[: args.max_clips]
        SSv2PairDataset.__init__ = _init

    tr = extract_split(extractor, args.ssv2_root, "train", False,
                       args.batch_clips, args.workers, ann_dir=args.ann_dir)
    va = extract_split(extractor, args.ssv2_root, "validation", True,
                       args.batch_clips, args.workers, ann_dir=args.ann_dir)

    evaluate = train_probe(tr, va, device, epochs=args.epochs)
    top1, top5 = evaluate(va["features"])
    top1_rev, _ = evaluate(va["features_rev"])
    summary = {
        "encoder": tag, "checkpoint": args.checkpoint,
        "n_train_pairs": int(tr["labels"].shape[0]),
        "n_val_clips": int(np.unique(va["clip_ids"]).shape[0]),
        "gap_frames": FPS_GAP, "img_size": IMG_SIZE, "pairs": PAIRS,
        "probe": f"linear_{args.readout}", "readout": args.readout, "epochs": args.epochs,
        "ann_dir": args.ann_dir,
        "top1": top1, "top5": top5,
        "top1_reversed": top1_rev,
        "direction_drop": top1 - top1_rev,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ssv2] DONE {tag}: top1={top1:.4f} top5={top5:.4f} "
          f"rev={top1_rev:.4f} (Δdir={top1-top1_rev:+.4f}) → {out}/summary.json", flush=True)


if __name__ == "__main__":
    main()
