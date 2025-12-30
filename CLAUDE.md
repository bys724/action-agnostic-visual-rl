# Claude Development Guide

## Project Overview
This project implements action-agnostic visual representation learning for robotic manipulation. The core hypothesis is that visual features learned without action conditioning generalize better across different robots and tasks.

## Current Status
- Initial project structure created
- Basic configuration files set up
- Dependencies listed in requirements.txt
- SIMPLER added as submodule in third_party/SimplerEnv
- Docker-based development environment configured
- Test script created: docker/test_env.py
- Environment wrapper created: src/envs/simpler_wrapper.py
- Setup guides: docs/DOCKER_SETUP.md, docs/SIMPLER_SETUP.md

## Development with Docker

### Quick Start
```bash
# Build and run Docker environment
./docker/build.sh
./docker/run.sh

# Inside container:
python docker/test_env.py
```

### Current Structure
```
/workspace/
├── docker/           # Docker scripts and tests
├── docs/            # Documentation
├── src/envs/        # Environment wrappers
└── third_party/     # SIMPLER environment
```

## Next Steps (To be implemented)

### 1. Core Components
- [ ] Visual encoder (DINOv2/CLIP)
- [ ] Policy network
- [ ] Training loop
- [ ] Evaluation pipeline

### 2. Technical Notes

#### SIMPLER Action Space
- 7-dimensional: [dx, dy, dz, rx, ry, rz, gripper]
- Position: Delta end-effector position
- Rotation: Axis-angle representation
- Gripper: Continuous value

#### EMA Update
```python
# Momentum encoder update
for param_q, param_k in zip(student.parameters(), teacher.parameters()):
    param_k.data = param_k.data * beta + param_q.data * (1 - beta)
```

## Important Commands

```bash
# Docker commands
docker-compose up simpler-dev    # Development environment
docker-compose up jupyter         # Jupyter Lab

# Inside container
cd /workspace
python docker/test_env.py        # Test SIMPLER environment
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