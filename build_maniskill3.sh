#!/bin/bash
# Build Docker image for ManiSkill3 version

echo "Building Docker image for ManiSkill3..."
echo "This will use the GPU-accelerated version which is more stable"

docker build -f Dockerfile.maniskill3 -t simpler-maniskill3:latest .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo ""
    echo "To run the container:"
    echo "  ./run_maniskill3.sh"
else
    echo ""
    echo "❌ Docker build failed!"
    echo "Please check the error messages above."
fi