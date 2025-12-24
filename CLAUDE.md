# Claude Development Guide

## Project Overview
This project implements action-agnostic visual representation learning for robotic manipulation. The core hypothesis is that visual features learned without action conditioning generalize better across different robots and tasks.

## Current Status
- Initial project structure created
- Basic configuration files set up
- Dependencies listed in requirements.txt
- SIMPLER added as submodule in third_party/SimplerEnv
- Test script created: scripts/test_simpler_env.py
- Environment wrapper created: src/envs/simpler_wrapper.py
- Setup guide: docs/SIMPLER_SETUP.md

## Next Steps

### 1. SIMPLER Environment Setup ✅
```bash
# Quick setup (using Python venv, no conda needed)
bash setup_env.sh

# Or manual installation
python3 -m venv venv
source venv/bin/activate
pip install numpy==1.24.4
cd third_party/SimplerEnv/ManiSkill2_real2sim && pip install -e .
cd ../ && pip install -e .
cd ../../ && pip install -r requirements.txt

# Test installation
python scripts/test_simpler_env.py --list-envs
python scripts/test_simpler_env.py --env google_robot_pick_coke_can --steps 100
```

### 2. Core Components to Implement

#### Visual Encoder (src/models/visual_encoder.py)
- DINOv2 backbone with partial fine-tuning
- CLIP backbone alternative
- EMA momentum teacher network (β = 0.999)

#### Policy Network (src/models/policy.py)
- MLP policy for SIMPLER action space (7-dim)
- Action format: [dx, dy, dz, rx, ry, rz, gripper]
- Quaternion to axis-angle conversion

#### Training Loop (scripts/train.py)
- Integration with SIMPLER reset() and step() methods
- Visual encoder + policy co-training
- EMA update for momentum teacher

### 3. Key Technical Details

#### SIMPLER Action Space
- **Format**: 7-dimensional continuous
- **Position**: Delta end-effector position (dx, dy, dz)
- **Rotation**: Axis-angle representation (rx, ry, rz)
- **Gripper**: Continuous value (open/close)

#### EMA Update Formula
```python
# Momentum encoder update
for param_q, param_k in zip(student.parameters(), teacher.parameters()):
    param_k.data = param_k.data * beta + param_q.data * (1 - beta)
```

#### Partial Fine-tuning Strategy
Based on PVM in MBRL paper:
- Freeze early layers (feature extraction)
- Fine-tune later layers (task-specific)
- Keep momentum encoder for stable targets

### 4. Experiment Pipeline

1. **Baseline Tests**
   - Random initialization
   - Frozen DINOv2/CLIP
   - Full fine-tuning

2. **Our Method**
   - Action-agnostic visual learning
   - EMA teacher-student
   - Partial fine-tuning

3. **Evaluation Metrics**
   - Success rate on SIMPLER tasks
   - Sample efficiency
   - Cross-embodiment transfer

### 5. Common Issues & Solutions

#### GPU Memory
- Use gradient accumulation if OOM
- Reduce batch size or image resolution
- Consider mixed precision training

#### SIMPLER Integration
- Ensure RT cores available (RTX GPU)
- Check SAPIEN renderer initialization
- Verify action space conversion

#### Training Stability
- Start with lower learning rate (1e-4)
- Warm-up EMA decay (0.996 → 0.999)
- Monitor gradient norms

## Important Commands

```bash
# Run lint and type checking (if available)
# npm run lint
# npm run typecheck
# ruff check .
# mypy src/

# Training
python scripts/train.py --config configs/experiment.yaml

# Evaluation
python scripts/evaluate.py --checkpoint path/to/model.pt

# Tensorboard
tensorboard --logdir experiments/
```

## Research Context
Refer to RESEARCH_CONTEXT.md for:
- Detailed research motivation
- Literature insights (PVM in MBRL, LAPA, DINO)
- Evaluation strategy on SIMPLER benchmark
- Expected challenges and timeline

## Development Priority
1. Get SIMPLER working with dummy policy ✅ (Next)
2. Implement visual encoder with EMA
3. Integrate RL training loop
4. Run baseline comparisons
5. Implement our proposed method
6. Ablation studies

## Notes
- Keep experiments minimal and focused
- Prioritize getting results over perfect code
- Document findings in experiments/
- Use configs/ for all hyperparameters