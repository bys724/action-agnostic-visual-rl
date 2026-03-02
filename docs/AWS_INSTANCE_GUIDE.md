# AWS EC2 Instance Setup Guide

## ⚠️ Critical: AMI Selection

### Common Mistake

**❌ 잘못된 AMI 선택 (실제 발생한 실수)**:
```bash
AMI: ami-0453ec754f44f9a4a
Name: "Amazon Linux 2023 AMI"
문제: NVIDIA 드라이버가 없음!
결과: nvidia-smi 실행 불가
```

### ✅ 올바른 AMI 선택

**Deep Learning OSS Nvidia Driver AMI** 사용 필수!

```bash
# 올바른 AMI 검색 (Ubuntu 22.04 + PyTorch 2.x)
aws ec2 describe-images --region us-west-2 \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 22.04)*" \
            "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].[ImageId,Name,CreationDate]' \
  --output table

# 2026-02-26 기준 최신:
AMI ID: ami-0f11d5e9f5325a8c9
Name: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) 20260222
```

### AMI 선택 체크리스트

- [ ] "Deep Learning" 포함
- [ ] "Nvidia Driver" 포함
- [ ] "Ubuntu 22.04" (또는 선호하는 OS)
- [ ] "PyTorch 2.x" 버전
- [ ] 최신 CreationDate

### 검증 방법

SSH 접속 후 즉시 확인:
```bash
nvidia-smi  # GPU 정상 인식되어야 함
python3 --version  # Python 3.9+
which python3
```

---

## Python Environment Setup (2026-03-01 경험)

### ⚠️ Deep Learning AMI의 함정

**문제**: "Deep Learning OSS Nvidia Driver AMI"는 **드라이버만** 제공하고 Python ML 패키지는 미설치!

```bash
# 잘못된 가정
python3 -c "import torch"  # ❌ ModuleNotFoundError: No module named 'torch'
pip3 list | grep torch     # ❌ 없음
```

### ✅ 올바른 Python 환경

AMI에는 `/opt/pytorch` 가상환경이 미리 설치되어 있음:

```bash
# 1. PyTorch 환경 확인
/opt/pytorch/bin/python3 -c "import torch; print(torch.__version__)"
# PyTorch: 2.7.0+cu128
# CUDA available: True
# GPU count: 4

# 2. 필요한 패키지 설치
/opt/pytorch/bin/pip install timm transformers tqdm matplotlib tensorboard opencv-python
```

### run_aws_training.sh 수정

스크립트에서 `/opt/pytorch/bin/python3` 사용하도록 수정 필요:

```bash
# scripts/run_aws_training.sh 수정
sed -i 's|python3|/opt/pytorch/bin/python3|g' scripts/run_aws_training.sh
```

### 필수 패키지 체크리스트

학습 시작 전 반드시 설치:

```bash
# Core (우리 프로젝트 필수)
/opt/pytorch/bin/pip install \
  timm \           # Vision models
  transformers \   # CLIP, DINOv2
  tqdm \           # Progress bar
  matplotlib \     # Visualization
  tensorboard \    # Logging
  opencv-python    # VideoMAE dependency
```

### 자동화 스크립트 (추천)

인스턴스 시작 후 자동으로 환경 설정:

```bash
# scripts/setup_aws_env.sh (미래 작업)
#!/bin/bash
/opt/pytorch/bin/pip install -q timm transformers tqdm matplotlib tensorboard opencv-python
cd /workspace/action-agnostic-visual-rl
sed -i 's|python3|/opt/pytorch/bin/python3|g' scripts/run_aws_training.sh
git submodule update --init --recursive external/VideoMAE
echo "Environment ready!"
```

---

## Instance Types & Quota

### G Instance Types (GPU)

| Type | GPUs | vCPU | RAM | Spot Price | On-Demand Price |
|------|------|------|-----|------------|-----------------|
| g5.8xlarge | 1× A10G | 32 | 128GB | $1.80/hr | $4.00/hr |
| g5.12xlarge | 4× A10G | 48 | 192GB | $3.50/hr | $7.00/hr |
| g5.24xlarge | 4× A10G | 96 | 384GB | $5.00/hr | $10.00/hr |
| g5.48xlarge | 8× A10G | 192 | 768GB | $10.00/hr | $20.00/hr |

### Quota Issues (2026-02-26 경험)

**문제 발견**:
```bash
# 서비스 쿼터 확인
aws service-quotas get-service-quota \
  --region us-west-2 \
  --service-code ec2 \
  --quota-code L-3819A6DF \  # G Spot vCPU
  --query 'Quota.[QuotaName,Value]'

# 결과: G Spot quota = 0 vCPU ❌
# 해결: On-demand 사용 (quota = 768 vCPU ✅)
```

**Quota Codes**:
- `L-3819A6DF`: All G and VT Spot Instance Requests
- `L-DB2E81BA`: Running On-Demand G and VT instances
- `L-34B43A08`: All Standard Spot Instance Requests

---

## Auto-Shutdown Mechanism

### 학습 완료 후 자동 종료

`run_aws_training.sh`에 이미 구현되어 있음:

```bash
# scripts/run_aws_training.sh 마지막 부분
if $SHUTDOWN; then
    echo "Shutting down instance in 60 seconds (Ctrl+C to cancel)..."
    sleep 60
    sudo shutdown -h now
fi
```

### 사용 방법

```bash
# 자동 종료 활성화 (기본값)
./scripts/run_aws_training.sh

# 자동 종료 비활성화 (디버깅용)
./scripts/run_aws_training.sh --no-shutdown
```

### Instance Termination Protection

**권장**: On-demand 인스턴스에는 termination protection 설정

```bash
# Termination protection 활성화
aws ec2 modify-instance-attribute \
  --region us-west-2 \
  --instance-id i-xxxxx \
  --disable-api-termination

# 학습 완료 후 수동으로 종료
aws ec2 modify-instance-attribute \
  --region us-west-2 \
  --instance-id i-xxxxx \
  --no-disable-api-termination

aws ec2 terminate-instances \
  --region us-west-2 \
  --instance-ids i-xxxxx
```

### CloudWatch Alarm (Optional)

학습이 멈춘 경우 자동 종료:

```bash
# CPU 사용률이 10% 미만으로 1시간 지속 시 종료
aws cloudwatch put-metric-alarm \
  --alarm-name training-idle-shutdown \
  --alarm-description "Stop instance if idle for 1 hour" \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 3600 \
  --evaluation-periods 1 \
  --threshold 10 \
  --comparison-operator LessThanThreshold \
  --dimensions Name=InstanceId,Value=i-xxxxx \
  --alarm-actions arn:aws:automate:region:ec2:stop
```

---

## Region & AZ Selection

### 2026-02-26 경험

**us-east-1** (버지니아):
- ❌ g5.8xlarge Spot: capacity-not-available
- ❌ g5.12xlarge Spot: capacity-not-available
- 문제: 지속적인 Spot interruption

**us-west-2** (오레곤):
- ✅ us-west-2a: 용량 부족 (g5.12xlarge On-demand)
- ✅ us-west-2b: 성공! (g5.12xlarge On-demand)
- ✅ us-west-2c: 가용 (확인됨)

### 권장 전략

1. **리전 선택**: us-west-2 > us-east-2 > us-east-1
2. **AZ 선택**: Placement 지정하지 않고 AWS가 자동 선택하게 하거나, 여러 AZ 시도
3. **인스턴스 타입**: Spot quota 0이면 On-demand 사용

---

## Cost Optimization

### EBS 볼륨 설정

```bash
# ❌ 잘못된 설정 (불필요한 비용)
--block-device-mappings '[{
  "DeviceName": "/dev/sda1",
  "Ebs": {
    "VolumeSize": 2048,  # 2TB
    "DeleteOnTermination": false  # ❌ 인스턴스 종료 후에도 유지
  }
}]'
# 결과: 월 $200 EBS 비용 발생

# ❌ 너무 작음 (500GB는 부족)
--block-device-mappings '[{
  "DeviceName": "/dev/sda1",
  "Ebs": {
    "VolumeSize": 500,  # 500GB → EgoDex part1 (336GB) + LIBERO (15GB) + 시스템으로 가득 참
    "VolumeType": "gp3",
    "DeleteOnTermination": true
  }
}]'
# 문제: PyTorch temp 파일 생성 불가, 학습 시작 실패

# ✅ 올바른 설정 (2026-02-28 업데이트)
--block-device-mappings '[{
  "DeviceName": "/dev/sda1",
  "Ebs": {
    "VolumeSize": 1024,  # 1TB (충분한 여유 공간)
    "VolumeType": "gp3",
    "DeleteOnTermination": true  # ✅ 인스턴스와 함께 삭제
  }
}]'
# EgoDex part1 (336GB) + LIBERO (15GB) + system (~50GB) + temp/cache (~100GB) = ~500GB 사용
# 1TB → 충분한 여유 확보
```

### S3 기반 데이터 관리

데이터는 S3에 보관하고 EC2는 캐시로만 사용:

```bash
# run_aws_training.sh가 자동으로 수행
# 1. S3에서 데이터 다운로드
# 2. 학습 진행
# 3. Checkpoint S3 업로드
# 4. 인스턴스 종료 → EBS 자동 삭제
```

---

## Security Best Practices

### Security Group

```bash
# SSH만 허용
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp --port 22 --cidr 0.0.0.0/0
```

### SSH Key Management

```bash
# Key를 여러 리전에 import
aws ec2 import-key-pair \
  --region us-west-2 \
  --key-name ml-research-key \
  --public-key-material fileb://<(ssh-keygen -y -f ~/.ssh/ml-research-key.pem)
```

---

## Troubleshooting

### Issue 1: nvidia-smi not found

**원인**: 잘못된 AMI 사용 (이 가이드 맨 위 참고)
**해결**: Deep Learning AMI 사용

### Issue 2: Spot capacity-not-available

**원인**: Spot quota 0 또는 용량 부족
**해결**:
1. Quota 확인 (service-quotas)
2. On-demand 사용
3. 다른 리전/AZ 시도

### Issue 3: MaxSpotInstanceCountExceeded

**원인**: G Spot quota = 0
**해결**: On-demand 인스턴스 사용

### Issue 4: ModuleNotFoundError: No module named 'torch' (2026-03-01)

**원인**: Deep Learning AMI는 드라이버만 제공, Python 패키지 미설치
**증상**:
```bash
python3 -c "import torch"
# ModuleNotFoundError: No module named 'torch'
```

**해결**:
1. `/opt/pytorch` 가상환경 사용
2. 필요한 패키지 설치
3. 스크립트 수정

```bash
# 1. PyTorch 환경 확인
/opt/pytorch/bin/python3 -c "import torch; print(torch.__version__)"

# 2. 패키지 설치
/opt/pytorch/bin/pip install timm transformers tqdm matplotlib tensorboard opencv-python

# 3. run_aws_training.sh 수정
cd /workspace/action-agnostic-visual-rl
sed -i 's|python3|/opt/pytorch/bin/python3|g' scripts/run_aws_training.sh
```

### Issue 5: ModuleNotFoundError: No module named 'modeling_finetune' (2026-03-01)

**원인**: VideoMAE 서브모듈 미초기화
**증상**:
```python
from modeling_finetune import Block, get_sinusoid_encoding_table
ModuleNotFoundError: No module named 'modeling_finetune'
```

**해결**:
```bash
# VideoMAE 서브모듈 초기화
cd /workspace/action-agnostic-visual-rl
git submodule update --init --recursive external/VideoMAE
```

**예방**: 새 인스턴스 시작 시 항상 서브모듈 초기화

### Issue 6: ModuleNotFoundError: No module named 'cv2' (2026-03-01)

**원인**: OpenCV 미설치 (VideoMAE dependency)
**해결**:
```bash
/opt/pytorch/bin/pip install opencv-python
```

### Issue 7: Instance auto-stopped after training start (2026-03-01)

**원인**:
1. Python 환경 문제로 학습 스크립트 즉시 실패
2. `run_aws_training.sh`의 auto-shutdown 로직 트리거
3. 인스턴스 stopped

**증상**:
- SSH 연결 실패 (Operation timed out)
- 인스턴스 상태: stopped
- 학습 로그에 에러만 있고 실제 학습 없음

**해결**:
1. 인스턴스 재시작: `aws ec2 start-instances --instance-ids i-xxxxx`
2. Python 환경 문제 해결 (Issue 4, 5, 6)
3. `--no-shutdown` 플래그 사용 (디버깅 중)

**예방**:
- 새 인스턴스는 항상 `--no-shutdown`으로 테스트 먼저
- Sanity test 후 자동 종료 활성화

### Issue 8: Slow training speed - DataLoader bottleneck (2026-03-02)

**증상**:
```
처리 속도: 0.402 batches/sec (38.5 samples/sec)
Epoch당 소요 시간: 33.3시간 (예상 40일)
CPU 사용률: 162% (48 코어 중 16.7%)
```

**원인**:
1. **VideoCapture 반복 생성**: 매 샘플당 `cv2.VideoCapture()` 2번 호출
   - Epoch당 924만 번 VideoCapture 생성
   - VideoCapture당 13ms 소요
2. **낮은 DataLoader workers**: `num_workers=8` (CPU 48코어 대비 부족)

**해결**:

```python
# src/models/two_stream.py

# 1. VideoCapture 재사용 최적화
def _load_frame_pair(self, video_path, frame_idx1, frame_idx2):
    """한 번의 VideoCapture로 두 프레임 로드"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx1)
    ret1, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx2)
    ret2, frame2 = cap.read()
    cap.release()
    # ... preprocessing
    return img1, img2

# 2. num_workers 증가 + persistent_workers
dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=effective_batch_size,
    shuffle=True,
    num_workers=32,  # 8 → 32
    pin_memory=True,
    persistent_workers=True,  # 추가
)
```

**예상 효과**:
- VideoCapture 재사용: 2배 개선
- num_workers 증가: 2-3배 개선
- 종합: 33.3시간/epoch → 5.5~11시간/epoch

**모니터링**:
```bash
# CPU 사용률 확인
ssh ubuntu@<IP> "top -b -n 1 | head -20"

# DataLoader worker 프로세스 확인
ps aux | grep pt_data

# 학습 진행률
tail -f /workspace/data/logs/train_two_stream.log
```

**예방**:
- 새 데이터셋 추가 시 DataLoader 성능 프로파일링 필수
- `test_dataloader.py`로 최적 num_workers 사전 확인
- GPU 대기 시간 모니터링 (GPU util < 80% → DataLoader 병목)

---

## Quick Reference

### Instance Launch (On-demand)

```bash
aws ec2 run-instances \
  --region us-west-2 \
  --image-id ami-0f11d5e9f5325a8c9 \
  --instance-type g5.12xlarge \
  --key-name ml-research-key \
  --security-group-ids sg-xxxxx \
  --iam-instance-profile Name=EC2-S3-FullAccess \
  --block-device-mappings '[{
    "DeviceName": "/dev/sda1",
    "Ebs": {
      "VolumeSize": 1024,
      "VolumeType": "gp3",
      "DeleteOnTermination": true
    }
  }]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ml-training}]'
```

### Connect & Setup

```bash
# 1. SSH 접속
ssh -i ~/.ssh/ml-research-key.pem ubuntu@<PUBLIC_IP>

# 2. 코드 가져오기
sudo mkdir -p /workspace && sudo chown ubuntu:ubuntu /workspace
cd /workspace
git clone https://github.com/bys724/action-agnostic-visual-rl.git
cd action-agnostic-visual-rl

# 3. Python 환경 설정 (필수!)
/opt/pytorch/bin/pip install timm transformers tqdm matplotlib tensorboard opencv-python
sed -i 's|python3|/opt/pytorch/bin/python3|g' scripts/run_aws_training.sh

# 4. 서브모듈 초기화 (VideoMAE)
git submodule update --init --recursive external/VideoMAE
```

### Start Training

```bash
# Sanity test 먼저 (5 videos, 1 epoch, auto-shutdown 비활성화)
./scripts/run_aws_training.sh --sanity --no-shutdown

# 전체 학습 (3 models × 30 epochs)
./scripts/run_aws_training.sh

# 특정 모델만
./scripts/run_aws_training.sh --model two-stream
```
