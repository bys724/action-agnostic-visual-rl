#!/bin/bash
# Docker 이미지 빌드 스크립트

echo "Building Docker image for SIMPLER environment..."

# Docker 이미지 빌드
docker build -t action-agnostic-visual-rl:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo ""
    echo "To start the development environment, run:"
    echo "  ./docker/run.sh"
    echo ""
    echo "To start Jupyter Lab, run:"
    echo "  docker-compose up jupyter"
else
    echo "❌ Docker build failed!"
    exit 1
fi