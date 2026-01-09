#!/usr/bin/env python
"""
모델 다운로드 및 체크포인트 관리 스크립트
HuggingFace 모델을 로컬에 저장하고 관리합니다.
"""
import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
import json
from datetime import datetime

def download_model(model_id: str, save_dir: str = "./data/checkpoints", model_type: str = "openvla"):
    """
    HuggingFace 모델을 로컬에 다운로드
    
    Args:
        model_id: HuggingFace 모델 ID (예: "openvla/openvla-7b")
        save_dir: 저장할 디렉토리
        model_type: 모델 타입 (openvla, lapa, custom 등)
    """
    save_path = Path(save_dir) / model_type
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 모델 이름 추출 (예: openvla-7b)
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    local_dir = save_path / model_name
    
    # 이미 존재하는지 확인
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"Model already exists at: {local_dir}")
        print("Use --force to re-download")
        return str(local_dir)
    
    print(f"Downloading {model_id} to {local_dir}...")
    
    try:
        # HuggingFace에서 다운로드
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # 실제 파일 복사
            resume_download=True,  # 중단된 다운로드 재개
            ignore_patterns=["*.bin", "*.safetensors"] if model_type == "test" else None
        )
        
        # 메타데이터 저장
        metadata = {
            "model_id": model_id,
            "model_type": model_type,
            "download_date": datetime.now().isoformat(),
            "local_path": str(local_dir),
            "source": "huggingface"
        }
        
        metadata_file = local_dir / "model_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model downloaded successfully to: {local_dir}")
        
        # 체크포인트 레지스트리 업데이트
        update_registry(model_name, str(local_dir), model_type)
        
        return str(local_dir)
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None


def update_registry(model_name: str, local_path: str, model_type: str):
    """체크포인트 레지스트리 파일 업데이트"""
    registry_file = Path("./data/checkpoints/registry.json")
    
    # 기존 레지스트리 로드
    if registry_file.exists():
        with open(registry_file, "r") as f:
            registry = json.load(f)
    else:
        registry = {"models": {}}
    
    # 새 엔트리 추가
    if model_type not in registry["models"]:
        registry["models"][model_type] = {}
    
    registry["models"][model_type][model_name] = {
        "path": local_path,
        "added": datetime.now().isoformat()
    }
    
    # 저장
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_file, "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"✓ Registry updated: {registry_file}")


def list_local_models():
    """로컬에 저장된 모든 모델 나열"""
    registry_file = Path("./data/checkpoints/registry.json")
    
    if not registry_file.exists():
        print("No models found in registry")
        return
    
    with open(registry_file, "r") as f:
        registry = json.load(f)
    
    print("\n" + "="*60)
    print("Local Model Checkpoints")
    print("="*60)
    
    for model_type, models in registry["models"].items():
        print(f"\n{model_type.upper()}:")
        for name, info in models.items():
            print(f"  • {name}")
            print(f"    Path: {info['path']}")
            print(f"    Added: {info['added'][:10]}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Model download and checkpoint management")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model_id", help="HuggingFace model ID (e.g., openvla/openvla-7b)")
    download_parser.add_argument("--type", default="openvla", help="Model type (openvla, lapa, custom)")
    download_parser.add_argument("--save-dir", default="./data/checkpoints", help="Save directory")
    download_parser.add_argument("--force", action="store_true", help="Force re-download")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List local models")
    
    args = parser.parse_args()
    
    if args.command == "download":
        download_model(args.model_id, args.save_dir, args.type)
    elif args.command == "list":
        list_local_models()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()