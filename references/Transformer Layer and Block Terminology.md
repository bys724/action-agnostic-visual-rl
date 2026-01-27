# Transformer Layer and Block Terminology

> **핵심 질문**: Original Transformer (2017)에서 말하는 "layer"와 현대 Transformer에서 말하는 "layer/block"은 동일한가?

---

## TL;DR

**현대적 용법**: **Layer = Block** (거의 동의어)
- 하나의 layer/block = **Attention + FFN**
- "12-layer Transformer" = 12개의 (Attention + FFN) blocks

**Original Transformer (2017)**에서는 Encoder/Decoder block 전체를 "layer"라고 불렀고, Attention과 FFN을 "sub-layer"로 구분했으나, 현대에는 이 구분을 거의 사용하지 않음.

---

## 1. Original Transformer (2017) - "Attention is All You Need"

**구조:**

```
┌─────────────────────────────────────────────────┐
│                  ENCODER                        │
├─────────────────────────────────────────────────┤
│  Encoder Block 1 (= "layer" in paper):          │
│  ├─ Multi-Head Self-Attention                   │
│  ├─ Add & Norm                                  │
│  ├─ Feed-Forward Network (FFN)                  │
│  └─ Add & Norm                                  │
│                                                  │
│  Encoder Block 2:                               │
│  ├─ Multi-Head Self-Attention                   │
│  └─ ... (same structure)                        │
│                                                  │
│  ... (총 6개 blocks)                            │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│                  DECODER                        │
├─────────────────────────────────────────────────┤
│  Decoder Block 1 (= "layer" in paper):          │
│  ├─ Masked Multi-Head Self-Attention            │
│  ├─ Add & Norm                                  │
│  ├─ Cross-Attention (attends to encoder)        │
│  ├─ Add & Norm                                  │
│  ├─ Feed-Forward Network (FFN)                  │
│  └─ Add & Norm                                  │
│                                                  │
│  ... (총 6개 blocks)                            │
└─────────────────────────────────────────────────┘
```

**Original 논문의 용어:**
- **"Layer" = Block** (논문에서는 "layer"라고 불렀음)
- Encoder는 6 layers = 6 blocks
- Decoder도 6 layers = 6 blocks
- 각 block 안에는 여러 **sub-layers** (Attention, FFN)

**인용:**
> "Each of these layers has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network."

---

## 2. 현대 Transformer Variants

원래 Encoder-Decoder 구조에서 분화:

### A. Encoder-only (BERT, ViT, CLIP)

```
┌─────────────────────────────────────┐
│  Transformer Block (= Layer):       │
│  ├─ Multi-Head Self-Attention       │
│  ├─ Add & Norm                      │
│  ├─ FFN                             │
│  └─ Add & Norm                      │
└─────────────────────────────────────┘
    ↓ (반복 N번)
```

**특징:**
- Cross-attention 없음
- Self-attention + FFN만 반복
- 양방향 attention (모든 토큰이 서로를 볼 수 있음)

**예시:**
- **BERT**: 12 layers (base) / 24 layers (large)
- **ViT**: 12 layers (base) / 24 layers (large)
- **CLIP image encoder**: 12 layers

### B. Decoder-only (GPT, LLaMA, Qwen)

```
┌─────────────────────────────────────┐
│  Transformer Block (= Layer):       │
│  ├─ Masked Self-Attention           │
│  ├─ Add & Norm                      │
│  ├─ FFN                             │
│  └─ Add & Norm                      │
└─────────────────────────────────────┘
    ↓ (반복 N번)
```

**특징:**
- Cross-attention 없음
- **Causal masking** (이전 토큰만 볼 수 있음)
- Autoregressive generation

**예시:**
- **GPT-3**: 96 layers
- **LLaMA-2 7B**: 32 layers
- **Qwen2-VL 7B**: 32 layers

### C. Encoder-Decoder (T5, BART) - 여전히 사용됨

Original Transformer 구조를 유지:
- Encoder: Self-attention only
- Decoder: Masked self-attention + Cross-attention to encoder

**사용 사례:**
- Seq2seq tasks (번역, 요약)
- T5, BART, mT5

---

## 3. 용어 정리 요약

| 용어 | Original Transformer (2017) | 현대 용법 | 예시 |
|------|----------------------------|----------|------|
| **Layer** | Encoder/Decoder block 전체 | Transformer block 전체 | "32-layer Qwen2-VL" |
| **Block** | 명시적으로 사용 안 함 | = Layer (동의어) | "Transformer block" |
| **Sub-layer** | Attention, FFN 각각 | 거의 사용 안 함 | - |

**현대적 관례:**
- **Layer = Block** (거의 동의어로 사용)
- 하나의 layer/block = **Attention + FFN**
- "12-layer Transformer" = 12개의 (Attention + FFN) blocks

**원래 Transformer와의 차이:**
1. Encoder-Decoder → Encoder-only 또는 Decoder-only
2. Cross-attention 사라짐 (대부분의 경우)
3. "Layer"가 전체 block을 의미 (sub-layer 구분 거의 안 함)

---

## 4. 실제 코드 예시

### PyTorch Transformer (Encoder-only)

```python
class TransformerLayer(nn.Module):
    """
    하나의 Transformer layer = block
    """
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Sub-layer 1: Self-Attention (with residual)
        attn_out = self.self_attn(x, x, x)[0]
        x = self.norm1(x + attn_out)

        # Sub-layer 2: FFN (with residual)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

# Full model
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=12, d_model=768, nhead=12, dim_feedforward=3072):
        super().__init__()
        # "12 layers" = 12개의 TransformerLayer
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

---

## 5. 주요 모델별 Layer 수

| 모델 | Architecture | Layers | Hidden Dim | Attention Heads |
|------|-------------|--------|-----------|----------------|
| **BERT-base** | Encoder-only | 12 | 768 | 12 |
| **BERT-large** | Encoder-only | 24 | 1024 | 16 |
| **GPT-2** | Decoder-only | 12/24/36/48 | varies | varies |
| **GPT-3** | Decoder-only | 96 | 12288 | 96 |
| **ViT-B/16** | Encoder-only | 12 | 768 | 12 |
| **ViT-L/16** | Encoder-only | 24 | 1024 | 16 |
| **LLaMA-2 7B** | Decoder-only | 32 | 4096 | 32 |
| **Qwen2-VL 7B** | Decoder-only | 32 | 4096 | varies |

---

## 6. Residual Connection과 Layer 구조

**Original Transformer의 핵심: Residual (skip) connections**

```
입력 ──┬──> Attention ──> + ──> LayerNorm ──┬──> FFN ──> + ──> LayerNorm ──> 출력
       │                   ↑                  │            ↑
       └───────────────────┘                  └────────────┘
        residual connection                    residual connection
```

**역할:**
1. **Gradient flow 개선**: Deep network에서 vanishing gradient 방지
2. **Identity mapping 보존**: 필요시 입력을 그대로 통과 가능
3. **학습 안정성**: 각 layer가 "변화량"만 학습 (Δx)

**Pre-LN vs Post-LN:**

```
Post-LN (Original Transformer):
x → Attention → Add → LayerNorm → FFN → Add → LayerNorm

Pre-LN (현대적, GPT-3/LLaMA):
x → LayerNorm → Attention → Add → LayerNorm → FFN → Add
```

현대 모델들은 **Pre-LN**을 주로 사용 (학습 안정성 ↑).

---

## 7. 관련 개념

- [[Attention Mechanism]] - Self-attention, cross-attention의 작동 원리
- [[Residual Connection]] - Skip connection의 역할과 중요성
- [[Layer Normalization]] - Batch norm vs Layer norm

---

## 8. 관련 논문

- [[Transformer (2017)]] - "Attention is All You Need", original architecture
- [[BERT (2018)]] - Encoder-only, bidirectional pre-training
- [[GPT-2 (2019)]] - Decoder-only, autoregressive generation
- [[ViT (2020)]] - Vision Transformer, image patches as tokens
- [[TwinBrainVLA (2026)]] - Dual-VLM with layer-by-layer AsyMoT interaction

---

## 태그

#transformer #architecture #layer #block #terminology #deep-learning
