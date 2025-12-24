# 설치 가이드 - venv 문제 해결

## 문제: python3-venv 패키지 미설치

시스템에 python3-venv 패키지가 없어서 가상환경 생성이 안 되는 경우입니다.

## 해결 방법

### 방법 1: python3-venv 설치 (권장 - 관리자 권한 필요)
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10-venv

# 또는 Python 버전에 따라
sudo apt install python3-venv
```

### 방법 2: virtualenv 사용 (관리자 권한 불필요)
```bash
# virtualenv 설치
pip3 install --user virtualenv

# 가상환경 생성
python3 -m virtualenv venv

# 활성화
source venv/bin/activate
```

### 방법 3: pip 직접 사용 (가상환경 없이)
```bash
# 사용자 디렉토리에 설치
pip3 install --user numpy==1.24.4

# SIMPLER 설치
cd third_party/SimplerEnv/ManiSkill2_real2sim
pip3 install --user -e .
cd ../
pip3 install --user -e .
cd ../../

# 프로젝트 패키지 설치
pip3 install --user -r requirements.txt
```

## 가상환경 활성화 확인

가상환경이 활성화되면 프롬프트가 변경됩니다:
```bash
# 활성화 전
bys@ETRI:~/action-agnostic-visual-rl$

# 활성화 후
(venv) bys@ETRI:~/action-agnostic-visual-rl$
```

## 활성화 명령어

```bash
# venv 사용시
source venv/bin/activate

# 비활성화
deactivate
```