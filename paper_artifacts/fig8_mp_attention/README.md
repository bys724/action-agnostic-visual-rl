# Fig 8 — M vs P Stream Visualization (Appendix D)

**Paper position**: Appendix D (qualitative attention + channel inputs)
**Status**: 🟡 partial — v11 통합본만 `docs/architecture/`에 존재, **M-only / P-only 분리 미생성**

## 폴더 구조 (사용자 요구 반영)

> 통합본 외 **M stream / P stream 각각의 그림도 별도로 관리** — 다른 figure 조립에 재사용 가능하도록.

```
fig8_mp_attention/
├── combined/   # 통합 grid (input + M attn + P attn + recon, 8-col 형식)
├── m_only/     # M stream 전용: m_channel_input + m_attention (별도 PNG)
└── p_only/     # P stream 전용: p_channel_input + p_attention (별도 PNG)
```

**파일명 규칙** (제안):
- combined: `{model}_{epoch}_combined.png` (예: `v15_ep32_combined.png`)
- m_only: `{model}_{epoch}_m_input.png`, `{model}_{epoch}_m_attn.png`
- p_only: `{model}_{epoch}_p_input.png`, `{model}_{epoch}_p_attn.png`
- sample 다양성 필요 시: `_s{N}` suffix (예: `v15_ep32_m_attn_s2.png`)

## Spec (Vault [[7. Outline § Fig 8]])

- 6×3 grid (combined): original frame | M attention overlay | P attention overlay
- 다양한 manipulation phase + gap variation (3-5 sample)
- **개선 (v15)**: dual-frame composite (past frame ghost + current sharp), motion-peak anchor

## Current artifacts (이 폴더)

없음. Source data는 모두 다른 위치에 있음 — 아래 참조.

## 기존 source (활용 가능)

| 위치 | 내용 | 비고 |
|------|------|------|
| [`docs/architecture/attn_v11_ep44_nomask.png`](../../docs/architecture/attn_v11_ep44_nomask.png) | v11 ep44 attention combined (8-col) | paper main 후보, **이미 통합본** |
| `docs/architecture/attn_v11_ep{4,8,12,16,20,24,48,50}_nomask.png` | v11 attention progression | supplementary |
| [`docs/architecture/sample_detail/`](../../docs/architecture/sample_detail/) | v4용 stream-wise 분리 viz (`02_m_channel_input.png` / `03_p_channel_input.png` / `04_m_attention.png` / `05_p_attention.png`) | **참고용 — v11/v15 동일 패턴으로 재생성 필요** |

## TODO

### v11 (method history figure에 활용 가능)

- [ ] `v11_ep44_combined.png` ← `docs/architecture/attn_v11_ep44_nomask.png` 심볼릭/복사 (current canonical)
- [ ] `v11_ep44_m_input.png` — M 채널만 (ΔL + 2 Sobel(ΔL)) 시각화
- [ ] `v11_ep44_p_input.png` — P 채널만 (RGB + 2 Sobel(L)) 시각화
- [ ] `v11_ep44_m_attn.png` — M encoder attention overlay only
- [ ] `v11_ep44_p_attn.png` — P encoder attention overlay only

### v15 (paper main qualitative)

- [ ] `v15_ep32_combined.png` (또는 `_ep50_`) — 8-col grid 통합본
- [ ] `v15_ep32_m_input.png`, `_p_input.png` — 입력 채널 분리
- [ ] `v15_ep32_m_attn.png`, `_p_attn.png` — attention 분리
- [ ] **3-frame triple 시각화 옵션 고려** — v15는 (prev, curr, target) 3-frame, 2-frame과 다름

### viz 스크립트 분리 출력 옵션 추가

현재 [`scripts/eval/visualize_attn_v11.py`](../../scripts/eval/visualize_attn_v11.py) / [`visualize_v15_no_mask.py`](../../scripts/eval/visualize_v15_no_mask.py) 는 통합 grid만 출력. **`--split-streams` 옵션 추가** 필요:
- `--out-mode combined` (default) → 기존 grid
- `--out-mode split` → m_input / p_input / m_attn / p_attn 각각 별도 PNG
- `--out-mode all` → 둘 다

## Notes

- 입력 채널 viz는 §3.1 Method (Fig 1 inset)에서도 활용 가능 — 본 폴더에 두되 fig1 README에서 cross-link
- attention overlay 색 컨벤션: M=blue heat / P=red heat (paper 전체 일관)
