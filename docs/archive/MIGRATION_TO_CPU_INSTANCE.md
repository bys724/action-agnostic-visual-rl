# GPU → CPU 인스턴스 마이그레이션 가이드

## 목적
프레임 추출을 GPU 인스턴스(g5.12xlarge)에서 CPU 인스턴스(c7i.16xlarge)로 마이그레이션하여 비용 절감

## 비용 절감
- **Before**: g5.12xlarge (GPU 미사용) - $5.67/h × 120h = **$680**
- **After**: c7i.16xlarge (CPU 전용) - $2.72/h × 600h = **$1,632**
- Part1만 추출 시: $680 → $245 (64% 절감, $435 절감)
- 전체(part1-5) 추출 시: 25일 소요, $1,792 (GPU 대비 합리적)

## 현재 상태 (2026-03-02)

### g5.12xlarge 인스턴스
- IP: 35.86.118.174
- 작업: Part1 프레임 추출 (중단됨)
- 진행률: 18-20% (6,185,887 frames, 88GB)
- S3 백업: 진행 중 → s3://bys724-research-2026/egodex_frames_partial

## 마이그레이션 단계

### Step 1: S3 백업 완료 확인 (진행 중)

```bash
# 백업 상태 확인 (로컬에서)
# (현재 백그라운드 실행 중, 35-40분 소요 예상)

# 완료 후 검증
ssh -i ~/.ssh/ml-research-key.pem ubuntu@35.86.118.174 << 'EOF'
LOCAL_COUNT=$(find /workspace/data/egodex_frames -type f -name "*.jpg" | wc -l)
S3_COUNT=$(aws s3 ls s3://bys724-research-2026/egodex_frames_partial/ --recursive | wc -l)
echo "Local: $LOCAL_COUNT, S3: $S3_COUNT"
EOF
```

### Step 2: c7i.16xlarge 인스턴스 시작

**AWS Console**:
1. EC2 → Launch Instance
2. 설정:
   - Name: `egodex-frame-extraction-cpu`
   - AMI: Deep Learning Base AMI (Ubuntu 22.04) 또는 Ubuntu 22.04 LTS
   - Instance type: **c7i.16xlarge** (64 vCPU, 128GB RAM)
   - Key pair: `ml-research-key.pem`
   - Storage: **2TB gp3** (3000 IOPS, 125 MB/s throughput)
   - Security group: SSH (22) from My IP
   - IAM role: S3 full access (또는 기존 role)
3. Launch

**User data** (선택사항, 자동 설정):
```bash
#!/bin/bash
apt update
apt install -y python3-pip git awscli
```

### Step 3: 새 인스턴스 환경 설정

```bash
# 1. 새 인스턴스 접속
NEW_IP="<새_인스턴스_IP_입력>"
ssh -i ~/.ssh/ml-research-key.pem ubuntu@$NEW_IP

# 2. Setup 스크립트 실행
cd /home/ubuntu
git clone https://github.com/bys724/action-agnostic-visual-rl.git
cd action-agnostic-visual-rl
chmod +x scripts/setup_cpu_extraction_instance.sh
./scripts/setup_cpu_extraction_instance.sh

# 위 스크립트가 자동으로:
# - 시스템 업데이트
# - Python 패키지 설치
# - EgoDex part1-5 다운로드 (1.68TB, ~2-3시간)
# - Part1 partial 프레임 복구
```

### Step 4: EgoDex 데이터 다운로드 확인

```bash
# 다운로드 진행 확인
watch -n 10 'du -sh /workspace/data/egodex/*'

# 완료 후 확인
du -sh /workspace/data/egodex
# 예상: ~1.7TB

ls -lh /workspace/data/egodex/
# part1, part2, part3, part4, part5 확인
```

### Step 5: 전체 프레임 추출 시작

```bash
cd /home/ubuntu/action-agnostic-visual-rl
chmod +x scripts/start_full_extraction.sh
./scripts/start_full_extraction.sh

# 출력:
# Starting extraction: part1
#   PID: 12345 (log: /workspace/data/logs/extract_part1.log)
# Starting extraction: part2
#   PID: 12346 (log: /workspace/data/logs/extract_part2.log)
# ...
```

### Step 6: 진행 상황 모니터링

```bash
# 로그 실시간 확인
tail -f /workspace/data/logs/extract_part1.log

# 모든 part 상태 확인
for part in part1 part2 part3 part4 part5; do
  echo "=== $part ==="
  tail -3 /workspace/data/logs/extract_${part}.log
done

# 프로세스 확인
ps aux | grep extract_frames

# 디스크 사용량 모니터링
watch -n 60 'du -sh /workspace/data/egodex_frames'

# CPU 사용량
htop
```

### Step 7: Part별 S3 업로드 (완료되는 대로)

각 part 완료 시 자동으로 S3 업로드하는 스크립트:

```bash
# 별도 터미널에서 실행 (백그라운드 모니터링)
cd /home/ubuntu/action-agnostic-visual-rl

for part in part1 part2 part3 part4 part5; do
  # Part 완료 대기
  while pgrep -f "extract_frames.py.*$part" > /dev/null; do
    sleep 600  # 10분마다 체크
  done

  echo "[$part] Extraction complete! Uploading to S3..."

  # S3 업로드
  aws s3 sync /workspace/data/egodex_frames \
    s3://bys724-research-2026/egodex_frames \
    --storage-class INTELLIGENT_TIERING \
    --exclude "*" \
    --include "*/${part}/*"

  echo "[$part] Upload complete!"

done &

# 위 스크립트 PID 저장
echo $! > /tmp/upload_monitor.pid
```

### Step 8: 이전 g5 인스턴스 종료

**S3 백업 완료 후 바로 실행**:

```bash
# 체크포인트 백업 (혹시 모르니)
ssh -i ~/.ssh/ml-research-key.pem ubuntu@35.86.118.174 \
  "aws s3 sync /workspace/data/checkpoints s3://bys724-research-2026/checkpoints_backup"

# AWS Console에서 g5.12xlarge 인스턴스 종료
# 또는 CLI:
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
```

## 예상 타임라인

```
Day 0 (오늘):
  - S3 백업 완료 (1시간)
  - c7i 인스턴스 시작 (10분)
  - EgoDex 전체 다운로드 (2-3시간)
  - 프레임 추출 시작
  - g5 인스턴스 종료

Day 1-25:
  - 전체 프레임 추출 진행
  - 완료된 part별 S3 업로드

Day 25:
  - 모든 추출 완료
  - 최종 S3 sync
  - 검증
  - c7i 인스턴스 종료

Day 26+:
  - Part1으로 학습 시작 (g5 재시작)
```

## 최종 결과

### S3 구조
```
s3://bys724-research-2026/
├── egodex_frames/
│   ├── part1/
│   │   ├── task1/video1/frame_000000.jpg
│   │   └── ...
│   ├── part2/
│   ├── part3/
│   ├── part4/
│   └── part5/
└── egodex_frames_partial/  (삭제 가능)
```

### 비용
- **c7i.16xlarge (25일)**: $1,632
- **EBS (2TB, 1개월)**: $160
- **S3 storage (1.5TB, 장기)**: $34.5/month
- **총 1회 비용**: $1,792
- **장기 유지 비용**: $34.5/month (S3만)

## 트러블슈팅

### EgoDex 다운로드 실패
```bash
# 특정 part만 재다운로드
aws s3 sync s3://egodex/part1 /workspace/data/egodex/part1 --delete
```

### 프레임 추출 중단됨
```bash
# 특정 part 재시작
ps aux | grep extract_frames  # 확인
pkill -f "extract_frames.py.*part3"  # 중단

# 재시작 (이미 추출된 비디오는 자동 스킵)
nohup python3 scripts/extract_frames.py \
  --egodex-root /workspace/data/egodex \
  --split part3 \
  --output-dir /workspace/data/egodex_frames \
  --num-workers 12 \
  > /workspace/data/logs/extract_part3.log 2>&1 &
```

### 디스크 부족
```bash
# EBS 확장 (AWS Console)
# 1. EC2 → Volumes → Modify Volume → 크기 증가
# 2. 인스턴스에서:
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
```

## 참고 문서
- `/tmp/full_extraction_plan.md` - 전체 계획 상세
- `scripts/extract_frames.py` - 프레임 추출 스크립트
- `docs/RESEARCH_PLAN.md` - 전체 연구 일정
