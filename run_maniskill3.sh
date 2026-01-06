#!/bin/bash
# Run Docker container for ManiSkill3

echo "Starting ManiSkill3 Docker container..."

# Allow X11 connections (for GUI if needed)
xhost +local:docker 2>/dev/null || true

docker run -it --rm \
    --gpus all \
    --network host \
    --ipc=host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v $(pwd):/workspace \
    -w /workspace \
    simpler-maniskill3:latest \
    bash
