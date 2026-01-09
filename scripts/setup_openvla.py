#!/usr/bin/env python
"""
OpenVLA 모델 설치 스크립트
- 의존성 패키지 설치
- ManiSkill assets 다운로드
- OpenVLA 모델 다운로드
"""

import os
import sys
import subprocess
from pathlib import Path
import json
from datetime import datetime
import argparse

def check_docker():
    """Docker 환경인지 확인"""
    return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')

def install_dependencies():
    """OpenVLA 실행에 필요한 Python 패키지 설치"""
    print("="*60)
    print("Installing OpenVLA Dependencies")
    print("="*60)
    
    packages = {
        "transformers": "4.45.0",
        "timm": "0.9.16",
        "accelerate": "0.30.0",  # Fixed version for stability
        "einops": None,
        "safetensors": None,
        "pillow": None,
        "sentencepiece": None,
    }
    
    for package, version in packages.items():
        if version:
            if ">=" in version or "<=" in version:
                cmd = f"pip install '{package}{version}'"
            else:
                cmd = f"pip install {package}=={version}"
        else:
            cmd = f"pip install {package}"
        
        print(f"  Installing {package}...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ✗ Failed to install {package}")
            print(f"    {result.stderr}")
            return False
    
    print("✓ All dependencies installed")
    return True

def download_maniskill_assets():
    """SimplerEnv 평가에 필요한 ManiSkill assets 다운로드"""
    print("\n" + "="*60)
    print("Downloading ManiSkill Assets")
    print("="*60)
    
    assets = [
        "bridge_v2_real2sim",
        "widowx250s",  # Robot asset for SimplerEnv
    ]
    
    for asset in assets:
        print(f"  Downloading {asset}...")
        cmd = f"echo 'y' | python -m mani_skill.utils.download_asset {asset} 2>/dev/null"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if "successfully downloaded" in result.stdout or "exists" in result.stdout:
            print(f"  ✓ {asset} ready")
        else:
            print(f"  ⚠ {asset} may need manual download")
    
    print("✓ Assets download complete")
    return True

def download_openvla_model(model_id="openvla/openvla-7b", force=False):
    """OpenVLA 모델 다운로드 - download_model.py 재사용"""
    print("\n" + "="*60)
    print("Downloading OpenVLA Model")
    print("="*60)
    
    # download_model.py의 기능을 재사용
    sys.path.insert(0, str(Path(__file__).parent))
    import download_model
    
    # 먼저 체크
    exists, path = download_model.check_model(model_id)
    if exists and not force:
        print(f"✓ Model already exists at: {path}")
        return path
    
    # 다운로드 실행
    path = download_model.download_model(model_id, model_type="openvla", force=force)
    if path:
        print(f"✓ Model downloaded successfully")
        return path
    else:
        print(f"✗ Failed to download model")
        return None

def verify_installation():
    """설치 확인"""
    print("\n" + "="*60)
    print("Verifying Installation")
    print("="*60)
    
    checks = []
    
    # Python 패키지 확인
    try:
        import transformers
        import timm
        import accelerate
        checks.append(("Python packages", True))
    except ImportError as e:
        checks.append(("Python packages", False))
    
    # 모델 파일 확인
    model_path = Path("data/checkpoints/openvla/openvla-7b")
    if model_path.exists() and any(model_path.iterdir()):
        size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024**3)
        checks.append((f"OpenVLA model ({size:.2f}GB)", True))
    else:
        checks.append(("OpenVLA model", False))
    
    # ManiSkill assets 확인
    asset_path = Path.home() / ".maniskill/data/tasks/bridge_v2_real2sim_dataset"
    if asset_path.exists():
        checks.append(("ManiSkill assets", True))
    else:
        checks.append(("ManiSkill assets", False))
    
    # 결과 출력
    all_good = True
    for name, status in checks:
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {name}")
        if not status:
            all_good = False
    
    if all_good:
        print("\n✓ OpenVLA is ready to use!")
        print("\nTest with:")
        print("  docker exec simpler-dev python src/eval_simpler.py --model openvla/openvla-7b --n-episodes 1 --max-steps 10")
    else:
        print("\n⚠ Some components are missing. Please check the errors above.")
    
    return all_good

def main():
    parser = argparse.ArgumentParser(description="OpenVLA setup and installation")
    parser.add_argument("--model", default="openvla/openvla-7b", help="Model ID to download")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-assets", action="store_true", help="Skip ManiSkill assets download")
    parser.add_argument("--verify-only", action="store_true", help="Only verify installation")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_installation()
        return
    
    print("="*60)
    print("OpenVLA Setup")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Environment: {'Docker' if check_docker() else 'Host'}")
    print()
    
    # 의존성 설치
    if not args.skip_deps:
        if not install_dependencies():
            print("Failed to install dependencies")
            sys.exit(1)
    
    # Assets 다운로드
    if not args.skip_assets:
        download_maniskill_assets()
    
    # 모델 다운로드
    path = download_openvla_model(args.model, args.force)
    if not path:
        print("Failed to download model")
        sys.exit(1)
    
    # 설치 확인
    verify_installation()

if __name__ == "__main__":
    main()