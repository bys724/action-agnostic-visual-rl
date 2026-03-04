#!/bin/bash
# 체크포인트 경로 설정 및 심볼릭 링크 생성
# Docker 서버가 기대하는 경로 구조로 맞춤

set -e

CHECKPOINT_DIR="/home/etri/action-agnostic-visual-rl/data/checkpoints"
LAPA_SRC="$CHECKPOINT_DIR/lapa/LAPA-7B-openx"
LAPA_DST="$CHECKPOINT_DIR/lapa"

echo "=== 체크포인트 경로 설정 ==="

# LAPA 심볼릭 링크 생성
# Docker 서버는 /app/checkpoints/lapa/{params,vqgan,tokenizer.model,action_scale.csv} 를 기대
setup_lapa() {
    echo "[LAPA] 경로 설정 중..."

    if [ -d "$LAPA_SRC" ]; then
        # params 심볼릭 링크
        if [ -f "$LAPA_SRC/params" ] && [ ! -e "$LAPA_DST/params" ]; then
            ln -sf "$LAPA_SRC/params" "$LAPA_DST/params"
            echo "  ✓ params 링크 생성"
        fi

        # vqgan 심볼릭 링크
        if [ -f "$LAPA_SRC/vqgan" ] && [ ! -e "$LAPA_DST/vqgan" ]; then
            ln -sf "$LAPA_SRC/vqgan" "$LAPA_DST/vqgan"
            echo "  ✓ vqgan 링크 생성"
        fi

        # tokenizer.model 심볼릭 링크
        if [ -f "$LAPA_SRC/tokenizer.model" ] && [ ! -e "$LAPA_DST/tokenizer.model" ]; then
            ln -sf "$LAPA_SRC/tokenizer.model" "$LAPA_DST/tokenizer.model"
            echo "  ✓ tokenizer.model 링크 생성"
        fi

        # action_scale.csv 생성 (없는 경우)
        if [ ! -f "$LAPA_DST/action_scale.csv" ]; then
            echo "  ⚠ action_scale.csv 없음 - LAPA 공식 저장소에서 확인 필요"
            # 임시 더미 파일 생성 (테스트용)
            # LAPA 공식 코드에서 action_scale.csv를 생성하는 방법 확인 필요
        fi

        echo "[LAPA] 완료"
    else
        echo "[LAPA] 소스 디렉토리 없음: $LAPA_SRC"
        echo "  다운로드 완료 후 다시 실행하세요"
    fi
}

# OpenVLA 경로 확인
setup_openvla() {
    echo "[OpenVLA] 경로 확인 중..."

    OPENVLA_DIR="$CHECKPOINT_DIR/openvla/openvla-7b"

    if [ -d "$OPENVLA_DIR" ]; then
        echo "  ✓ OpenVLA 체크포인트 존재: $OPENVLA_DIR"

        # 필수 파일 확인
        if [ -f "$OPENVLA_DIR/config.json" ]; then
            echo "  ✓ config.json 확인"
        else
            echo "  ⚠ config.json 없음 - 다운로드 진행 중일 수 있음"
        fi
    else
        echo "  ⚠ OpenVLA 체크포인트 없음"
        echo "  다운로드 완료 후 다시 실행하세요"
    fi
}

# 환경변수 템플릿 생성
create_env_template() {
    ENV_FILE="/home/etri/action-agnostic-visual-rl/.env"

    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << 'EOF'
# Docker 환경변수 설정

# OpenVLA
OPENVLA_MODEL_PATH=/app/checkpoints/openvla/openvla-7b

# LAPA
LAPA_CHECKPOINT_PATH=/app/checkpoints/lapa/params
LAPA_ACTION_SCALE=/app/checkpoints/lapa/action_scale.csv
LAPA_VQGAN=/app/checkpoints/lapa/vqgan
LAPA_VOCAB=/app/checkpoints/lapa/tokenizer.model
EOF
        echo "[ENV] .env 파일 생성됨"
    else
        echo "[ENV] .env 파일 이미 존재"
    fi
}

# 실행
setup_lapa
setup_openvla
create_env_template

echo ""
echo "=== 설정 완료 ==="
echo "다음 단계:"
echo "1. 다운로드 완료 확인: ls -la $CHECKPOINT_DIR/*/"
echo "2. Docker 시작: docker compose up -d"
echo "3. 서버 상태 확인: curl http://localhost:8001/health"
