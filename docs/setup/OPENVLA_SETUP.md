# OpenVLA 설정

## 빠른 시작 (API 모드)

```bash
# 서버 시작
docker compose up -d openvla

# 평가 실행
docker exec simpler-dev python src/eval_simpler.py \
    --model openvla --api-url http://localhost:8001 --n-episodes 24
```

## 로컬 모드 (eval 컨테이너 내)

```bash
# 모델 설치
docker exec simpler-dev python scripts/setup_openvla.py

# 평가
docker exec simpler-dev python src/eval_simpler.py --model "openvla/openvla-7b"
```

## 하드웨어 요구사항
- GPU: 24GB+ VRAM (RTX 3090, A5000, A100)
- RAM: 32GB+

## 참고
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [HuggingFace](https://huggingface.co/openvla)
