#!/bin/bash
# Xvfb 시작 스크립트 (컨테이너 내부에서 실행)

echo "Starting Xvfb virtual display..."
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
echo "Xvfb started on DISPLAY=:99"
echo ""
echo "You can now run SIMPLER tests:"
echo "  python docker/test_env.py"