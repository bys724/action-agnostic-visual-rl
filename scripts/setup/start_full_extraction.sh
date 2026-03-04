#!/bin/bash
# Start full EgoDex frame extraction (part1-5 in parallel)

set -e

cd /home/ubuntu/action-agnostic-visual-rl

# Create log directory
mkdir -p /workspace/data/logs

echo "=== Starting Full Frame Extraction ==="
echo "Parts: part1-5 (parallel)"
echo "Workers per part: 12"
echo "Total CPU: 64 vCPU"
echo "Output: /workspace/data/egodex_frames"
echo ""

# Start extraction for each part in parallel
for part in part1 part2 part3 part4 part5; do
  echo "Starting extraction: $part"
  nohup python3 scripts/extract_frames.py \
    --egodex-root /workspace/data/egodex \
    --split $part \
    --output-dir /workspace/data/egodex_frames \
    --num-workers 12 \
    --img-size 224 \
    > /workspace/data/logs/extract_${part}.log 2>&1 &

  echo "  PID: $! (log: /workspace/data/logs/extract_${part}.log)"
done

echo ""
echo "=== Extraction Started ==="
echo ""
echo "Monitor progress:"
echo "  tail -f /workspace/data/logs/extract_part*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep extract_frames"
echo ""
echo "Disk usage:"
echo "  watch -n 60 'du -sh /workspace/data/egodex_frames'"
echo ""

# Show initial logs
sleep 5
echo "=== Initial Status ==="
for part in part1 part2 part3 part4 part5; do
  echo "--- $part ---"
  tail -5 /workspace/data/logs/extract_${part}.log 2>/dev/null || echo "  Starting..."
done

echo ""
echo "All extractions running in background!"
