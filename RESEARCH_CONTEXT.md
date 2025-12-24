# Research Context: Action-Agnostic Visual Representation for Robotic Manipulation

**Target**: RSS 2026 submission
**Focus**: Learning visual representations that generalize across different action spaces and embodiments

---

## Research Motivation

### The Core Problem

Current robot learning suffers from **action-conditioned representation collapse**:
- Visual encoders learn features entangled with specific action spaces
- Representations don't transfer across different robots or tasks
- Pre-trained vision models (PVMs) show promise but underperform expectations in MBRL

### Why This Matters

For robots to operate in diverse real-world environments:
1. **Cross-embodiment transfer**: Same visual understanding across different robots
2. **Task generalization**: Learn "what matters" visually, independent of "how to act"
3. **Data efficiency**: Leverage large-scale pre-training without action information

---

## Key Insights from Literature

### PVM in MBRL (2025)
- **Finding**: Partial fine-tuning of visual encoders outperforms both frozen and full fine-tuning
- **Why**: Preserves pre-trained semantic knowledge while adapting to task-specific features
- **Encoders**: DINOv2 and CLIP show strong results
- **Architecture**: DreamerV3 with momentum encoder (EMA-based teacher network)

### LAPA (2024)
- **Finding**: Behavior representation learning benefits from action-agnostic visual features
- **Evaluation**: Uses SIMPLER benchmark for sim-to-real validation
- **Key technique**: EMA momentum encoder for stable learning targets

### DINO (2021) & Self-Supervised Learning
- **Core idea**: Self-distillation with momentum teacher creates rich visual features
- **No action information**: Learns purely from visual data
- **Generalization**: Features transfer well across domains

### EMA (Exponential Moving Average)
- **Role**: Creates stable teacher networks in self-supervised learning
- **Formula**: θ_teacher ← β · θ_teacher + (1-β) · θ_student
- **Typical β**: 0.996 ~ 0.9999 for very slow updates
- **Effect**: Prevents moving target problem, enables stable learning

---

## Research Hypothesis

**Main Claim**:
Visual representations learned without action conditioning generalize better across:
- Different action spaces (continuous vs discrete, different DoF)
- Different embodiments (different robot morphologies)
- Different tasks (manipulation, navigation, etc.)

**Technical Approach**:
- Decouple visual encoding from action prediction
- Use self-supervised objectives (like DINO) or action-masked training
- Evaluate generalization on SIMPLER benchmark

---

## Experimental Strategy

### Evaluation Platform: SIMPLER Benchmark
- **Why SIMPLER**: Standardized sim-to-real evaluation with ~1,500 real-world validation episodes
- **Simulator**: SAPIEN (ray-traced, photorealistic rendering)
- **Tasks**: google_robot_pick, widowx_spoon_on_towel, etc.
- **Metrics**: Success rate, average steps, generalization across embodiments

### Baseline Comparisons
1. **Random init**: CNN encoder trained from scratch with RL
2. **Frozen PVM**: DINOv2/CLIP frozen during training
3. **Full fine-tuned PVM**: End-to-end training with pre-trained init
4. **Partial fine-tuned**: Fine-tune only specific layers (from PVM in MBRL)
5. **Our method**: Action-agnostic visual representation learning

### Ablation Studies
- Different visual backbones: DINOv2, CLIP, MAE, etc.
- Varying β for momentum encoder: 0.996, 0.999, 0.9999
- Partial vs full fine-tuning strategies
- With/without EMA teacher network

---

## Technical Requirements

### Hardware
- **GPU**: RTX series with ray tracing support (RT cores) required for SAPIEN
- **VRAM**: 12GB+ recommended, 24GB+ for large models
- **Multi-GPU**: Optional but helpful for parallel evaluation

### Software Stack
- **Simulator**: SIMPLER (built on SAPIEN + ManiSkill2/3)
- **Deep Learning**: PyTorch, transformers (for DINO/CLIP)
- **RL Framework**: Depends on approach (Dreamer, SAC, etc.)

### Action Space Format (SIMPLER Standard)
- **Format**: [dx, dy, dz, rx, ry, rz, gripper] (7-dim)
- **Rotation**: Axis-angle representation
- **Gripper**: Single scalar value
- **Note**: Different models may output different formats - conversion needed

---

## Expected Challenges

1. **Policy Integration**: SIMPLER requires specific interface (reset(), step() methods)
2. **Action Conversion**: Quaternion ↔ axis-angle, different gripper conventions
3. **Image Preprocessing**: Model-specific requirements (resolution, normalization)
4. **Debugging**: Sim environment behavior, action space mismatches

**Realistic Timeline**: 2-3 weeks for SIMPLER setup + initial experiments

---

## Success Criteria

### Quantitative Metrics
- **Primary**: Success rate on SIMPLER tasks > baselines
- **Secondary**:
  - Cross-embodiment transfer (Google Robot → WidowX)
  - Sample efficiency (performance vs training steps)
  - Real-world validation (if feasible)

### Qualitative Analysis
- Visualization of learned representations (t-SNE, attention maps)
- Comparison of feature space across different action spaces
- Ablation study insights on what makes representations action-agnostic

---

## Related Work in This Vault

### Papers
- `[[PVM in MBRL (2025)]]`: Partial fine-tuning strategy, DreamerV3 architecture
- `[[LAPA (2024)]]`: Behavior representation, SIMPLER evaluation
- `[[DINO (2021)]]`: Self-supervised learning, momentum encoder
- `[[OpenVLA (2024)]]`: Vision-language-action models, cross-embodiment
- `[[CLIP (2021)]]`: Vision-language pre-training

### Concepts
- `[[EMA (Exponential Moving Average)]]`: Mathematical foundation for momentum encoders
- `[[Momentum Encoder and Self-Distillation]]`: Teacher-student training paradigm

---

## Notes for Implementation

### Code Philosophy
- **Modularity**: Separate visual encoder, policy network, and RL algorithm
- **Configurability**: Use config files for different model variants
- **Reproducibility**: Log hyperparameters, random seeds, environment details

### Experiment Tracking
- Track all hyperparameters, model checkpoints
- Save evaluation videos for qualitative analysis
- Record GPU usage, training time for efficiency analysis

### Documentation
- Document design decisions and their rationale
- Keep experiment logs with observations
- Maintain comparison tables for different approaches

---

**Last Updated**: 2025-12-24
**Status**: Initial setup phase

---

## Quick Start Guide

1. **Clone repository** on workstation with RTX GPU
2. **Install SIMPLER**: Follow SIMPLER official setup (SAPIEN + ManiSkill)
3. **Run baseline**: Test with existing models (RT-1, Octo) to verify setup
4. **Implement custom policy**:
   - Visual encoder (DINOv2/CLIP)
   - Policy interface for SIMPLER (reset/step methods)
   - Action conversion to SIMPLER format
5. **Evaluate**: Compare against baselines on standard tasks
6. **Iterate**: Ablation studies, hyperparameter tuning

**Key Reference**: Read `PVM in MBRL (2025)` paper notes for architectural insights.
