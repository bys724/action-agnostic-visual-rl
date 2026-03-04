#!/bin/bash
# Dataset download script: BridgeData V2, LIBERO
# Usage: ./download_datasets.sh [bridge|libero|all]
#
# Resumable: wget -c (BridgeV2) and huggingface-cli (LIBERO) both support resume.
# Re-run this script after spot interruption to continue where it left off.
#
# Environment variables:
#   DATA_DIR   Base data directory (default: /workspace/data/datasets)
#   LOG_DIR    Log directory (default: /workspace/data/logs)

set -e

DATA_DIR="${DATA_DIR:-/workspace/data/datasets}"
LOG_DIR="${LOG_DIR:-/workspace/data/logs}"
mkdir -p "$DATA_DIR" "$LOG_DIR"

# BridgeData V2 다운로드
download_bridge() {
    echo "=== BridgeData V2 download ==="
    BRIDGE_DIR="$DATA_DIR/bridge_v2"
    mkdir -p "$BRIDGE_DIR"
    cd "$BRIDGE_DIR"

    # demos_8_17.zip (~411GB), -c: resume on re-run
    if [ ! -d "demos" ]; then
        echo "[1/2] Downloading demos_8_17.zip (~411GB)..."
        wget -c --progress=bar:force:noscroll \
            "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/demos_8_17.zip" \
            -O demos_8_17.zip 2>&1 | tee "$LOG_DIR/bridge_demos.log"
    else
        echo "[1/2] demos/ already extracted, skipping"
    fi

    # scripted_6_18.zip (~30GB), -c: resume on re-run
    if [ ! -d "scripted" ]; then
        echo "[2/2] Downloading scripted_6_18.zip (~30GB)..."
        wget -c --progress=bar:force:noscroll \
            "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/scripted_6_18.zip" \
            -O scripted_6_18.zip 2>&1 | tee "$LOG_DIR/bridge_scripted.log"
    else
        echo "[2/2] scripted/ already extracted, skipping"
    fi

    echo "=== BridgeData V2 download complete ==="
    echo "  Extract: cd $BRIDGE_DIR && unzip demos_8_17.zip && unzip scripted_6_18.zip"
}

# LIBERO 다운로드 (OpenVLA modified RLDS format, ~10GB)
# huggingface-cli resumes automatically on re-run
download_libero() {
    echo "=== LIBERO download (openvla/modified_libero_rlds) ==="
    LIBERO_DIR="$DATA_DIR/libero"
    mkdir -p "$LIBERO_DIR"

    if ! command -v huggingface-cli &> /dev/null; then
        echo "Installing huggingface_hub..."
        pip install -q huggingface_hub
    fi

    # Add .local/bin to PATH for pip-installed executables
    export PATH="$HOME/.local/bin:$PATH"

    echo "Downloading to $LIBERO_DIR/modified_libero_rlds (~10GB)..."
    huggingface-cli download openvla/modified_libero_rlds \
        --repo-type dataset \
        --local-dir "$LIBERO_DIR/modified_libero_rlds" \
        2>&1 | tee "$LOG_DIR/libero.log"

    echo "=== LIBERO download complete ==="
}

# 메인
case "${1:-bridge}" in
    bridge)
        download_bridge
        ;;
    libero)
        download_libero
        ;;
    all)
        # 병렬 다운로드
        download_bridge &
        BRIDGE_PID=$!
        download_libero &
        LIBERO_PID=$!
        wait $BRIDGE_PID
        wait $LIBERO_PID
        ;;
    *)
        echo "Usage: $0 [bridge|libero|all]"
        exit 1
        ;;
esac

echo ""
echo "Logs: $LOG_DIR/"
echo "Data: $DATA_DIR/"
