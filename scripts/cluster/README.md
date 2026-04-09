# IBS Cluster (olaf) Quickstart

이 디렉토리는 **IBS Slurm 클러스터** 전용 launcher 모음입니다.
로컬 워크스테이션용 launcher는 `scripts/local/`에 있습니다. Python 코드 (`scripts/pretrain.py`,
`scripts/data/`, `scripts/eval/`)는 환경 무관이며 두 환경 모두에서 그대로 사용합니다.

## 클러스터 환경 요약

| 항목 | 값 |
|------|-----|
| Slurm version | 24.11.7 |
| GPU partition | `AIP` (3일), `AIP_long` (14일, 우선순위 낮음) |
| GPU 노드 | olaf-g[001-005,012] — 노드당 H100 × 4 |
| GPU 가용 | 최대 2 노드 × 4 = **8 H100** |
| CPU partition | `core_s` (2h, 우선순위 200), `normal_cpu` (3일), `large_cpu` (3일, 8노드+) |
| CPU 노드 | 노드당 144 CPU, 257 GB RAM |
| 컨테이너 | `apptainer/1.4.3` 모듈 (Docker 미사용) |
| Python | `miniforge3/24.11.3-2` 모듈 (conda/mamba) |
| 영구 저장소 | `/proj/external_group/mrg/` (GPFS, ~16 GB/s, mrg 그룹) |
| Local scratch | `/scratch` (NVMe, GPU 노드 olaf-g 전용, 잡 종료 시 휘발) |

**저장소 / Scratch 정책** (관리자 확인, 2026-04-09)

| 저장소 | 종류 | Read | Write | 위치 | 보존 |
|--------|------|------|-------|------|------|
| `/proj` | GPFS (공유 병렬) | ~16 GB/s | ~16 GB/s | 모든 노드 | 영구 |
| `/scratch` | 로컬 NVMe SSD | ~6.8 GB/s | ~2 GB/s | **GPU 노드 (olaf-g) 에만** | **잡 종료 시 보장 안 됨** |

- 이 클러스터는 GPFS(`/proj`)가 절대 throughput 면에서 scratch보다 빠름. **scratch는 필수가 아닌 보조 저장소**.
- scratch가 유리한 경우: 작은 파일 많은 워크로드, metadata-heavy random I/O, GPFS 부하 높을 때.
- **CPU 노드 (olaf-c)에는 scratch 없음** → 프레임 추출 같은 CPU 잡은 무조건 `/proj` 직접 사용.
- **GPU 학습 시**: 수십만 JPG random read는 scratch가 유리할 가능성 있음. 단, 먼저 GPFS 직접으로 측정 후 병목 시 stage-in 도입 (premature optimization 방지).
- **체크포인트**: scratch에 쓰면 잡 종료 직전 반드시 `/proj`로 stage-out 필요.

권장 stage-in/out 패턴 (학습 잡 작성 시 참고):
```bash
# Stage in (잡 시작 시)
SCRATCH_DATA=/scratch/$USER/$SLURM_JOB_ID/egodex
mkdir -p "$SCRATCH_DATA"
rsync -a /proj/external_group/mrg/datasets/egodex/frames/ "$SCRATCH_DATA/"

# 학습 (--egodex-root $SCRATCH_DATA)
python scripts/pretrain.py --egodex-root "$SCRATCH_DATA" ...

# Stage out (잡 종료 직전, trap으로 안전하게)
trap 'rsync -a "$SCRATCH_CKPT/" /proj/external_group/mrg/checkpoints/' EXIT
```

## 데이터 경로 규약

```
/proj/external_group/mrg/
├── datasets/
│   └── egodex/
│       ├── zips/         # part1.zip ~ part5.zip (CDN 다운로드 원본, ~1.5 TB)
│       ├── raw/          # part1/ ~ part5/ (압축 해제된 mp4, ~1.5 TB)
│       └── frames/       # part1/ ~ part5/ (256x256 jpg, ~500 GB)
├── conda_envs/
│   └── aavrl-extract/    # 프레임 추출용 경량 env (cv2, tqdm)
├── checkpoints/          # (예정) 학습 체크포인트
└── logs/                 # sbatch 로그 (.out, .err)
```

## EgoDex 다운로드 → 추출 워크플로우

### 1. 다운로드 (로그인 노드, 백그라운드)

CDN 다운로드는 네트워크 I/O만 사용 (CPU/GPU 거의 없음) → 로그인 노드에서 nohup 직접 실행.
sbatch 잡 불필요. 측정된 sustained 속도 ~60 MB/s, part당 ~85분, 5 parts 합계 약 7시간.

```bash
nohup bash scripts/cluster/download_egodex.sh part1 part2 part3 part4 part5 \
    > /proj/external_group/mrg/logs/download.log 2>&1 &
disown

# 진행 상황 확인
tail -f /proj/external_group/mrg/logs/download.log
```

`curl -C -` 옵션으로 중간에 끊겨도 같은 명령으로 이어받기 가능.

### 2. 프레임 추출 (sbatch, 144 CPU 병렬)

다운로드 완료된 part부터 즉시 제출 가능. 각 part가 독립 잡이라 일부만 먼저 처리 가능.

```bash
# 단일 part
sbatch --export=PART=part1 scripts/cluster/extract_egodex.sbatch

# 모든 part 한 번에 (sbatch array)
sbatch --array=1-5 scripts/cluster/extract_egodex.sbatch

# 진행 상황
squeue --me
tail -f /proj/external_group/mrg/logs/extract_egodex_*.out
```

스크립트가 자동으로 수행:
1. zip → `raw/$PART/` 압축 해제 (이미 풀려 있으면 스킵)
2. `extract_frames.py` 144 worker로 병렬 추출
3. 비디오 수 vs 프레임 디렉토리 수 검증

## 학습 (예정)

`scripts/cluster/pretrain.sbatch` — 2 노드 × 4 H100 DDP. 작성 예정.
DataParallel→DistributedDataParallel 전환 + multi-node `torchrun` 설정 필요.

## sbatch 일반 팁

```bash
# 잡 확인
squeue --me
squeue --me -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"

# 잡 취소
scancel <JOBID>
scancel -u $USER  # 본인 모든 잡

# 노드 상태
sinfo -p AIP -o "%n %G %T"
sinfo -p normal_cpu -o "%n %T %C"  # CPU alloc/idle/other/total

# 잡 자세히 보기
scontrol show job <JOBID>

# 끝난 잡 회계
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS
```

## 환경 디버그 (이슈 발생 시)

```bash
# 로그인 노드에서 conda env 테스트
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /proj/external_group/mrg/conda_envs/aavrl-extract
python -c "import cv2; print(cv2.__version__)"

# 짧은 인터랙티브 잡
salloc -p core_s -t 00:30:00 -N 1 --exclusive
# (할당 후 노드에서 직접 명령 실행)
```
