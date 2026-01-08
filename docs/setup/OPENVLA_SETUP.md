# OpenVLA Integration Guide

## Overview
OpenVLA (Open Vision-Language-Action Model) is a 7B parameter model for robotic manipulation. This guide explains how to use OpenVLA with SimplerEnv evaluation framework.

## Installation

### 1. Docker Environment (Recommended)
The Docker environment already includes all necessary dependencies:
```bash
# Build Docker image
docker build -t simpler-env:latest .

# Run container
docker compose up -d eval
docker compose exec eval bash
```

### 2. Manual Installation
If installing outside Docker:
```bash
pip install transformers==4.46.0
pip install accelerate==0.32.1
pip install timm==0.9.10
pip install tokenizers==0.15.2
pip install sentencepiece pillow
```

## Usage

### Evaluation
```bash
# Evaluate OpenVLA on SimplerEnv tasks
python src/eval_simpler.py \
    --model "openvla/openvla-7b" \
    --n-episodes 24 \
    --max-steps 300
```

### Trajectory Collection
```bash
# Collect successful trajectories
python src/collect_trajectories.py \
    --model "openvla/openvla-7b" \
    --n-per-task 25 \
    --max-steps 300
```

### Quick Test
```bash
# Run integration test
./scripts/test_openvla.sh
```

## Model Variants

1. **openvla/openvla-7b**: Base model
2. **openvla/openvla-7b-finetuned**: Fine-tuned on specific tasks
3. Local checkpoint: `--model /path/to/checkpoint`

## Implementation Details

### Policy Wrapper
The OpenVLA policy is implemented in `src/policies/openvla/openvla_model.py`:

- **OpenVLAPolicy**: Main policy class
  - Loads model from HuggingFace or local path
  - Handles image preprocessing
  - Generates 7D actions for SimplerEnv
  
- **Action Space**: 
  - 7D continuous actions: [dx, dy, dz, rx, ry, rz, gripper]
  - Scaled appropriately for SimplerEnv tasks

### Integration Points
- `src/eval_simpler.py`: Model loading and evaluation
- `src/collect_trajectories.py`: Trajectory collection

## Performance Notes

### GPU Requirements
- Minimum: 24GB VRAM (RTX 3090, A5000)
- Recommended: 40GB+ VRAM (A100, H100)

### Optimization Tips
1. Use bfloat16 precision (default)
2. Enable flash attention if available
3. Batch inference when possible

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size in evaluation
--n-episodes 1  # Process one episode at a time
```

### Model Loading Issues
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/

# Download model explicitly
huggingface-cli download openvla/openvla-7b
```

### Slow Inference
- Ensure GPU is being used: `nvidia-smi`
- Check CUDA version compatibility
- Consider using smaller test episodes first

## References

- [OpenVLA Paper](https://arxiv.org/abs/2406.09246)
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [SimplerEnv-OpenVLA Fork](https://github.com/DelinQu/SimplerEnv-OpenVLA)
- [HuggingFace Models](https://huggingface.co/openvla)

## Expected Performance

Based on SimplerEnv benchmarks:
- Zero-shot: 20-40% success rate
- Fine-tuned: 60-80% success rate
- Inference speed: ~2-5 Hz

## Next Steps

1. Fine-tune on collected trajectories
2. Compare with other baselines (RT-1, Octo)
3. Implement multi-task evaluation
4. Add visual matching evaluation mode