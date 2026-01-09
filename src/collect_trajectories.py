#!/usr/bin/env python
"""
Trajectory 수집 스크립트
모델 로드 → 성공 trajectory 수집 → 저장
"""

import os
# Disable Vulkan for ManiSkill3 (use software rendering)
os.environ["SAPIEN_DISABLE_VULKAN"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import pickle
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from datetime import datetime
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
            print("Warning: LAPA not yet implemented")
            return None
    
    elif model_type == "custom":
        # 사용자 정의 모델 플레이스홀더
        try:
            from policies.custom import CustomPolicy
            print(f"Loading Custom model from: {checkpoint_path}")
            return CustomPolicy(model_path=checkpoint_path)
        except ImportError:
            print("Warning: Custom model not yet implemented")
            return None
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def collect_successful_trajectories(policy, task_name, n_target=25, max_steps=300):
    """성공한 trajectory만 수집"""
    env = gym.make(task_name, obs_mode="rgb+segmentation", num_envs=1)
    trajectories = []
    attempts = 0
    max_attempts = n_target * 20  # 충분한 시도 횟수
    
    while len(trajectories) < n_target and attempts < max_attempts:
        trajectory = {
            "task": task_name,
            "observations": [],
            "actions": [],
            "rewards": [],
            "language_instruction": env.unwrapped.get_language_instruction()[0]
        }
        
        obs, _ = env.reset()
        instruction = trajectory["language_instruction"]
        
        # Policy 초기화
        if hasattr(policy, 'reset'):
            policy.reset(instruction)
        elif hasattr(policy, 'phase'):  # SimplePolicy
            policy.phase = 0
            policy.phase_steps = 0
        
        for step in range(max_steps):
            # 이미지 저장 (numpy array로 변환)
            img = get_image_from_maniskill3_obs_dict(env, obs)
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            trajectory["observations"].append(img)
            
            # 액션 생성
            if hasattr(policy, 'step'):  # RT-1, OpenVLA 모델
                raw_action, action = policy.step(img, instruction)
                # terminate_episode 체크
                if action.get("terminate_episode", [0])[0] > 0:
                    break
                env_action = action["world_vector"]
            else:  # SimplePolicy
                env_action = policy.get_action(obs)
            
            trajectory["actions"].append(env_action)
            
            # 환경 스텝
            obs, reward, terminated, truncated, info = env.step(env_action)
            
            # 보상 저장 (스칼라로 변환)
            if hasattr(reward, 'item'):
                reward = reward.item()
            elif isinstance(reward, np.ndarray):
                reward = float(reward[0]) if reward.shape else float(reward)
            trajectory["rewards"].append(reward)
            
            if terminated or truncated:
                break
        
        # 성공 여부 확인
        if hasattr(info, '__iter__'):
            success = info.get("success", [False])[0] if isinstance(info.get("success"), list) else info.get("success", False)
        else:
            success = False
        
        # 성공한 경우만 저장
        if success:
            trajectory["success"] = True
            trajectory["total_reward"] = sum(trajectory["rewards"])
            trajectories.append(trajectory)
            print(f"  Collected {len(trajectories)}/{n_target} (attempt {attempts+1})")
        
        attempts += 1
    
    env.close()
    
    if len(trajectories) < n_target:
        print(f"  Warning: Only collected {len(trajectories)}/{n_target} after {attempts} attempts")
    
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Trajectory Collection for SimplerEnv")
    
    # 모델 설정
    parser.add_argument("--type", type=str, default="openvla",
                       help="Model type (openvla, lapa, custom)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Model checkpoint path or ID")
    parser.add_argument("--name", type=str,
                       help="Model name for output file (optional)")
    
    # 수집 설정
    parser.add_argument("--output", type=str, default="./data/trajectories",
                       help="Trajectory save directory")
    parser.add_argument("--n-per-task", type=int, default=25,
                       help="Trajectories to collect per task")
    parser.add_argument("--max-steps", type=int, default=300,
                       help="Max steps per episode")
    parser.add_argument("--tasks", nargs="+",
                       help="Tasks to collect from (default: all 4)")
    
    args = parser.parse_args()
    
    # 기본 task 목록
    default_tasks = [
        "PutSpoonOnTableClothInScene-v1",
        "PutCarrotOnPlateInScene-v1",
        "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
        "PutEggplantInBasketScene-v1"
    ]
    tasks = args.tasks if args.tasks else default_tasks
    
    # 정책 로드
    print(f"Loading {args.type} model from: {args.checkpoint}")
    policy = load_policy(args.type, args.checkpoint)
    
    if policy is None:
        print("Failed to load policy!")
        return
    
    # 모델 이름 결정
    if args.name:
        model_name = args.name
    else:
        model_name = f"{args.type}_{Path(args.checkpoint).stem}"
    
    # 수집 실행
    all_trajectories = []
    print("\n" + "="*60)
    print(f"Collecting {args.n_per_task} trajectories per task")
    print(f"Model: {model_name}")
    print(f"Tasks: {len(tasks)}")
    print("="*60)
    
    for task in tasks:
        print(f"\nCollecting from {task}...")
        trajs = collect_successful_trajectories(
            policy, task, args.n_per_task, args.max_steps
        )
        all_trajectories.extend(trajs)
        print(f"  Completed: {len(trajs)} trajectories collected")
    
    print("\n" + "="*60)
    print(f"Total collected: {len(all_trajectories)} trajectories")
    print("="*60)
    
    # 저장
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{model_name}_{len(all_trajectories)}trajs_{timestamp}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    print(f"\nSaved to {output_file}")
    
    # 간단한 통계 출력
    print("\nTrajectory Statistics:")
    for task in tasks:
        task_trajs = [t for t in all_trajectories if t["task"] == task]
        if task_trajs:
            avg_reward = np.mean([t["total_reward"] for t in task_trajs])
            print(f"  {task}: {len(task_trajs)} trajs, avg reward: {avg_reward:.2f}")


if __name__ == "__main__":
    main()