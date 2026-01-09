#!/usr/bin/env python
"""
SimplerEnv 평가 스크립트 - 다중 모델/체크포인트 비교 지원
"""

import os
# Disable Vulkan for ManiSkill3 (use software rendering)
os.environ["SAPIEN_DISABLE_VULKAN"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import json
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *


def load_policy(model_type: str, checkpoint_path: str):
    """
    정책 로드
    
    Args:
        model_type: 모델 타입 (openvla, lapa, custom)
        checkpoint_path: 체크포인트 경로 또는 모델 ID
    """
    import sys
    import os
    
    # Add src directory to path for import
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    if model_type == "openvla":
        from policies.openvla import OpenVLAPolicy
        print(f"Loading OpenVLA from: {checkpoint_path}")
        return OpenVLAPolicy(model_path=checkpoint_path)
    
    elif model_type == "lapa":
        # LAPA 모델 플레이스홀더
        try:
            from policies.lapa import LAPAPolicy
            print(f"Loading LAPA from: {checkpoint_path}")
            return LAPAPolicy(model_path=checkpoint_path)
        except ImportError:
            print("Warning: LAPA not yet implemented, using mock policy")
            return MockPolicy()
    
    elif model_type == "custom":
        # 사용자 정의 모델 플레이스홀더
        try:
            from policies.custom import CustomPolicy
            print(f"Loading Custom model from: {checkpoint_path}")
            return CustomPolicy(model_path=checkpoint_path)
        except ImportError:
            print("Warning: Custom model not yet implemented, using mock policy")
            return MockPolicy()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class MockPolicy:
    """테스트용 Mock Policy"""
    def reset(self, instruction):
        pass
    
    def step(self, image, instruction):
        action = np.zeros(7, dtype=np.float32)
        return action, {"world_vector": action, "terminate_episode": np.array([0])}


def evaluate_single_model(
    policy,
    task_name: str,
    n_episodes: int = 24,
    max_steps: int = 300,
    verbose: bool = True
) -> Dict[str, Any]:
    """단일 task에서 모델 평가"""
    env = gym.make(task_name, obs_mode="rgb+segmentation", num_envs=1)
    
    episode_results = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        instruction = env.unwrapped.get_language_instruction()[0]
        
        # Policy 초기화
        if hasattr(policy, 'reset'):
            policy.reset(instruction)
        
        episode_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # 이미지 추출
            image = get_image_from_maniskill3_obs_dict(env, obs)
            
            # 액션 생성
            if hasattr(policy, 'step'):
                raw_action, action = policy.step(image, instruction)
                if action.get("terminate_episode", [0])[0] > 0:
                    break
                env_action = action["world_vector"]
            else:
                env_action = np.zeros(7, dtype=np.float32)
            
            # 환경 스텝
            obs, reward, terminated, truncated, info = env.step(env_action)
            episode_reward += float(reward) if hasattr(reward, '__float__') else reward
            steps = step + 1
            
            if terminated or truncated:
                break
        
        # 성공 여부 확인
        success = info.get("success", [False])[0] if isinstance(info.get("success"), list) else info.get("success", False)
        
        episode_results.append({
            "success": float(success),
            "reward": episode_reward,
            "steps": steps
        })
        
        if verbose:
            print(f"    Episode {episode+1}/{n_episodes}: {'✓' if success else '✗'} (reward: {episode_reward:.2f}, steps: {steps})")
    
    env.close()
    
    # 통계 계산
    successes = [r["success"] for r in episode_results]
    rewards = [r["reward"] for r in episode_results]
    steps_list = [r["steps"] for r in episode_results]
    
    return {
        "success_rate": np.mean(successes),
        "success_std": np.std(successes),
        "avg_reward": np.mean(rewards),
        "avg_steps": np.mean(steps_list),
        "episodes": episode_results
    }


def evaluate_models(
    models: List[Dict[str, str]],
    tasks: List[str],
    n_episodes: int = 24,
    max_steps: int = 300,
    output_dir: str = "./data/results"
) -> pd.DataFrame:
    """
    여러 모델을 평가하고 결과 비교
    
    Args:
        models: [{"name": "model_name", "type": "openvla", "checkpoint": "path/to/checkpoint"}, ...]
        tasks: 평가할 task 목록
        n_episodes: task당 에피소드 수
        max_steps: 에피소드당 최대 스텝
        output_dir: 결과 저장 디렉토리
    
    Returns:
        결과 DataFrame
    """
    results = []
    
    for model_config in models:
        model_name = model_config["name"]
        model_type = model_config["type"]
        checkpoint = model_config["checkpoint"]
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"  Type: {model_type}")
        print(f"  Checkpoint: {checkpoint}")
        print('='*60)
        
        # 정책 로드
        try:
            policy = load_policy(model_type, checkpoint)
        except Exception as e:
            print(f"Error loading model: {e}")
            continue
        
        model_results = {
            "model": model_name,
            "type": model_type,
            "checkpoint": checkpoint
        }
        
        # 각 task 평가
        for task in tasks:
            print(f"\n  Task: {task}")
            task_result = evaluate_single_model(
                policy, task, n_episodes, max_steps, verbose=False
            )
            
            # 결과 저장
            model_results[f"{task}_success"] = task_result["success_rate"]
            model_results[f"{task}_reward"] = task_result["avg_reward"]
            model_results[f"{task}_steps"] = task_result["avg_steps"]
            
            print(f"    Success rate: {task_result['success_rate']:.2%} (±{task_result['success_std']:.2%})")
            print(f"    Avg reward: {task_result['avg_reward']:.2f}")
        
        # 평균 성공률 계산
        success_rates = [model_results[f"{task}_success"] for task in tasks]
        model_results["avg_success"] = np.mean(success_rates)
        
        results.append(model_results)
        
        # 개별 모델 결과 저장
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = output_path / f"{model_name}_{timestamp}.json"
        
        with open(model_file, 'w') as f:
            json.dump(model_results, f, indent=2)
        print(f"\n  Results saved to {model_file}")
    
    # DataFrame으로 변환
    if results:
        df = pd.DataFrame(results)
        
        # 비교 표 출력
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        
        # 성공률만 보여주는 간단한 표
        success_cols = ["model", "avg_success"] + [f"{task}_success" for task in tasks]
        if not df.empty and all(col in df.columns for col in success_cols):
            success_df = df[success_cols].copy()
            
            # 퍼센트로 변환
            for col in success_cols[1:]:
                success_df[col] = success_df[col].apply(lambda x: f"{x:.1%}")
            
            print(success_df.to_string(index=False))
        else:
            print("No successful model evaluations to compare.")
        
        # 전체 결과 CSV 저장
        if not df.empty:
            csv_file = Path(output_dir) / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_file, index=False)
            print(f"\nFull results saved to {csv_file}")
    else:
        print("\n" + "="*80)
        print("No models were successfully evaluated.")
        print("="*80)
        df = pd.DataFrame()
    
    return df


def main():
    parser = argparse.ArgumentParser(description="SimplerEnv Multi-Model Evaluation")
    
    # 단일 모델 평가 옵션 (하위 호환성)
    parser.add_argument("--model", type=str, help="Single model evaluation (legacy mode)")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for single model")
    
    # 다중 모델 평가 옵션
    parser.add_argument("--config", type=str, help="JSON config file with model list")
    parser.add_argument("--models", nargs="+", help="Model configs as 'name:type:checkpoint'")
    
    # 공통 옵션
    parser.add_argument("--tasks", nargs="+", help="Tasks to evaluate (default: all 4)")
    parser.add_argument("--n-episodes", type=int, default=24, help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--output", type=str, default="./data/results", help="Output directory")
    
    args = parser.parse_args()
    
    # 기본 task 목록
    default_tasks = [
        "PutSpoonOnTableClothInScene-v1",
        "PutCarrotOnPlateInScene-v1",
        "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
        "PutEggplantInBasketScene-v1"
    ]
    tasks = args.tasks if args.tasks else default_tasks
    
    # 모델 설정 파싱
    models = []
    
    if args.config:
        # JSON 설정 파일에서 로드
        with open(args.config, 'r') as f:
            config = json.load(f)
            models = config.get("models", [])
    
    elif args.models:
        # 커맨드라인에서 파싱
        for model_str in args.models:
            parts = model_str.split(":")
            if len(parts) == 3:
                models.append({
                    "name": parts[0],
                    "type": parts[1],
                    "checkpoint": parts[2]
                })
    
    elif args.model:
        # 레거시 단일 모델 모드
        model_type = "openvla"  # 기본값
        if "lapa" in args.model.lower():
            model_type = "lapa"
        elif "custom" in args.model.lower():
            model_type = "custom"
        
        checkpoint = args.checkpoint if args.checkpoint else args.model
        model_name = Path(checkpoint).stem if "/" in checkpoint else checkpoint
        
        models = [{
            "name": model_name,
            "type": model_type,
            "checkpoint": checkpoint
        }]
    
    else:
        # 기본: OpenVLA 테스트
        models = [{
            "name": "openvla-7b",
            "type": "openvla",
            "checkpoint": "openvla/openvla-7b"
        }]
    
    # 평가 실행
    print("="*80)
    print(f"SimplerEnv Multi-Model Evaluation")
    print(f"Models: {len(models)}")
    print(f"Tasks: {len(tasks)}")
    print(f"Episodes per task: {args.n_episodes}")
    print("="*80)
    
    df = evaluate_models(
        models=models,
        tasks=tasks,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        output_dir=args.output
    )
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == "__main__":
    main()