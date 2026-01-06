#!/bin/bash
# Docker 컨테이너 실행 스크립트

echo "Starting SIMPLER development environment..."

# Xvfb를 백그라운드에서 실행하고 bash 쉘 시작
docker run -it --rm \
    --runtime=nvidia \
    --gpus all \
    --shm-size=16gb \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY=:99 \
    -e MUJOCO_GL=egl \
    -e PYOPENGL_PLATFORM=egl \
    -v $(pwd):/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --network=host \
    --name simpler-dev \
    action-agnostic-visual-rl:latest \
    bash -c "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & exec bash"