# Archived Artifacts

Paper main (v15)에서 **사용하지 않는** 산출물. 삭제 대신 보존 — reviewer-defense / regression test / history reference 시 필요.

## 보관 사유별 목록

### Deprecated 모델 (v13)

| Path | 사유 |
|------|------|
| `two_stream_v13_architecture.mmd/.png` | v13 (Dual-Frame + DINO global CLS) deprecated — paradigm conflict로 collapse |
| `v13_encroute_train_samples/epoch_004_nomask.png`, `epoch_008_nomask.png` | v13 학습 dynamics viz — collapse 진단 시 사용 |
| `v13_loss_progress.png` | v13 학습 loss 진행 |
| `v13_sanity_dynamics.png` | v13 sanity test dynamics |

→ **paper §method history**에서 v13는 "stream-wise paradigm separation motivation"으로 1문장 언급 가능. Appendix F (학습 결함 분석)에서 일부 figure 재인용 가능.

### Sanity / 검증용

| Path | 사유 |
|------|------|
| `v3_sanity_aug_check.png` | v3 augmentation sanity check — Phase 1 fundamental 확정 시 사용 |

### Phase 2.5 Negative Result (value alignment)

| Path | 사유 |
|------|------|
| `value_alignment/` 전체 | Phase 2.5 VIP-style value alignment — v11 꼴찌 (+0.47 vs VC-1 +0.80), **negative result** |

→ **paper에서 1문장 negative result 또는 §6 limitations 인용 가능** ("we also explored VIP-style state-similarity probing — results showed limitations of contrastive-objective-pretrained encoders favorability; we leave detailed analysis to future work"). Reviewer가 "VIP-style probing 했나"고 물으면 즉시 참조.

## 복구 방법

본 폴더의 모든 파일은 git history에 남아있음:

```bash
git log --all --oneline -- paper_artifacts/_archive/<file>
git checkout <commit> -- paper_artifacts/_archive/<file>
```

또는 단순히 _archive에서 상위로 mv.
