#!/bin/bash
# BridgeData V2 & DROID 데이터셋 다운로드 스크립트
# Usage: ./download_datasets.sh [bridge|droid|all]

set -e

DATA_DIR="/home/etri/action-agnostic-visual-rl/data/datasets"
LOG_DIR="/home/etri/action-agnostic-visual-rl/logs"
mkdir -p "$LOG_DIR"

# BridgeData V2 다운로드
download_bridge() {
    echo "=== BridgeData V2 다운로드 시작 ==="
    BRIDGE_DIR="$DATA_DIR/bridge_v2"
    mkdir -p "$BRIDGE_DIR"
    cd "$BRIDGE_DIR"

    # demos_8_17.zip (~411GB) - 메인 데모 데이터
    if [ ! -f "demos_8_17.zip" ] && [ ! -d "demos" ]; then
        echo "[1/2] demos_8_17.zip 다운로드 중 (~411GB)..."
        wget -c --progress=bar:force:noscroll \
            "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/demos_8_17.zip" \
            -O demos_8_17.zip 2>&1 | tee "$LOG_DIR/bridge_demos_download.log"
    else
        echo "[1/2] demos_8_17.zip 이미 존재함, 스킵"
    fi

    # scripted_6_18.zip (~30GB) - 스크립트 정책 데이터
    if [ ! -f "scripted_6_18.zip" ] && [ ! -d "scripted" ]; then
        echo "[2/2] scripted_6_18.zip 다운로드 중 (~30GB)..."
        wget -c --progress=bar:force:noscroll \
            "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/scripted_6_18.zip" \
            -O scripted_6_18.zip 2>&1 | tee "$LOG_DIR/bridge_scripted_download.log"
    else
        echo "[2/2] scripted_6_18.zip 이미 존재함, 스킵"
    fi

    echo "=== BridgeData V2 다운로드 완료 ==="
    echo "압축 해제: unzip demos_8_17.zip && unzip scripted_6_18.zip"
}

# DROID 다운로드 (gsutil 필요)
download_droid() {
    echo "=== DROID 데이터셋 다운로드 시작 ==="
    DROID_DIR="$DATA_DIR/droid"
    mkdir -p "$DROID_DIR"

    if ! command -v gsutil &> /dev/null; then
        echo "Error: gsutil이 설치되지 않았습니다."
        echo "설치: pip install gsutil 또는 apt install google-cloud-sdk"
        exit 1
    fi

    echo "DROID RLDS 다운로드 중 (~1.7TB)..."
    gsutil -m cp -r gs://gresearch/robotics/droid "$DROID_DIR/" 2>&1 | tee "$LOG_DIR/droid_download.log"

    echo "=== DROID 다운로드 완료 ==="
}

# 메인
case "${1:-bridge}" in
    bridge)
        download_bridge
        ;;
    droid)
        download_droid
        ;;
    all)
        download_bridge
        download_droid
        ;;
    *)
        echo "Usage: $0 [bridge|droid|all]"
        exit 1
        ;;
esac

echo ""
echo "다운로드 로그: $LOG_DIR/"
echo "데이터 위치: $DATA_DIR/"
