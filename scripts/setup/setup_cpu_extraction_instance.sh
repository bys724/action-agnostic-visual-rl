#!/bin/bash
# Setup script for c7i.16xlarge CPU instance for full EgoDex frame extraction

set -e

echo "=== EgoDex Full Frame Extraction Setup ==="
echo "Instance: c7i.16xlarge (64 vCPU, 128GB RAM)"
echo "Task: Extract frames from EgoDex part1-5 (1.68TB)"
echo ""

# Update system
echo "[1/6] Updating system..."
sudo apt update
sudo apt install -y python3-pip git awscli htop iotop

# Install Python dependencies
echo "[2/6] Installing Python dependencies..."
pip3 install opencv-python tqdm pillow numpy --break-system-packages

# Create workspace
echo "[3/6] Creating workspace..."
sudo mkdir -p /workspace/data
sudo chown ubuntu:ubuntu /workspace/data
cd /workspace/data

# Clone repository
echo "[4/6] Cloning repository..."
cd /home/ubuntu
git clone https://github.com/bys724/action-agnostic-visual-rl.git
cd action-agnostic-visual-rl

# Download EgoDex data (part1-5 in parallel)
echo "[5/6] Downloading EgoDex data (1.68TB)..."
echo "This will take ~2-3 hours depending on S3 bandwidth"
cd /workspace/data
mkdir -p egodex

# Download all parts in parallel
for part in part1 part2 part3 part4 part5; do
  echo "Starting download: $part..."
  aws s3 sync s3://egodex/$part egodex/$part &
done

# Wait for all downloads
echo "Waiting for downloads to complete..."
wait

echo "[6/6] Downloading partial frames from part1..."
aws s3 sync s3://bys724-research-2026/egodex_frames_partial \
  /workspace/data/egodex_frames

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Verify downloads: du -sh /workspace/data/egodex/*"
echo "2. Start extraction: cd /home/ubuntu/action-agnostic-visual-rl"
echo "3. Run: ./scripts/start_full_extraction.sh"
echo ""
echo "Estimated extraction time: ~25 days"
echo "Estimated cost: $1,632 (instance) + $160 (EBS) = $1,792"
