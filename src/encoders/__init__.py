"""Pre-trained vision encoder loaders + LIBERO BC-T 어댑터.

각 encoder는 native input format을 받음 (D3):
  · Two-Stream v11 / VideoMAE-ours : 2-frame pair → 1 token/timestep
  · DINOv2 / SigLIP / VC-1          : 1-frame × 2 → concat → 1 token/timestep
  · V-JEPA 2.1                      : 16-frame 누적 sliding window → 1 token/timestep
"""
