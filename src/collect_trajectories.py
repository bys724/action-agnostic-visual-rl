#!/usr/bin/env python
"""
Trajectory 수집 스크립트
모델 로드 → 성공 trajectory 수집 → 저장
"""

import argparse
import pickle
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *


class SimplePolicy:
    """테스트용 Simple Policy (test_simpler_demo.py에서 가져옴)"""
    
    def __init__(self):
        self.step_count = 0
        self.phase = 0
        self.phase_steps = 0
        
    def get_action(self, obs):
        """Generate simple action sequence for demonstration"""
        sequences = [
            (20, np.array([0, 0, -0.01, 0, 0, 0, 1.0])),
            (10, np.array([0, 0, 0, 0, 0, 0, -1.0])),
            (20, np.array([0, 0, 0.01, 0, 0, 0, -1.0])),
            (30, np.array([0.005, 0.005, 0, 0, 0, 0, -1.0])),
            (20, np.array([0, 0, -0.005, 0, 0, 0, -1.0])),
            (10, np.array([0, 0, 0, 0, 0, 0, 1.0])),
        ]
        
        if self.phase >= len(sequences):
            self.phase = 0
            self.phase_steps = 0
            
        duration, action = sequences[self.phase]
        self.phase_steps += 1
        
        if self.phase_steps >= duration:
            self.phase += 1
            self.phase_steps = 0
            
        self.step_count += 1
        return action.astype(np.float32)
    
    def reset(self, instruction=None):
        """Reset policy state (instruction ignored for SimplePolicy)"""
        self.phase = 0
        self.phase_steps = 0
        self.step_count = 0


def load_policy(model_path):
    """정책 로드"""
    if model_path == "simple":
        return SimplePolicy()
    
    # SimplerEnv 베이스라인 모델 로드
    policy_setup = "widowx_bridge"  # WidowX 로봇 사용
    
    if "octo" in model_path.lower():
        from simpler_env.policies.octo.octo_model import OctoInference
        model_type = "octo-small" if "small" in model_path else "octo-base"
        print(f"Loading Octo model: {model_type}")
        return OctoInference(model_type=model_type, policy_setup=policy_setup, init_rng=0)
    
    elif "rt1" in model_path.lower() or "rt_1" in model_path.lower():
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        print(f"Loading RT-1 model from: {model_path}")
        return RT1Inference(saved_model_path=model_path, policy_setup=policy_setup)
    
    else:
        print(f"Warning: Unknown model type, using SimplePolicy")
        return SimplePolicy()


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
            if hasattr(policy, 'step'):  # RT-1, Octo 모델
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="simple",
                       help="모델 체크포인트 경로 또는 'simple' for test")
    parser.add_argument("--output", type=str, default="./data/trajectories",
                       help="Trajectory 저장 디렉토리")
    parser.add_argument("--n-per-task", type=int, default=25,
                       help="Task당 수집할 trajectory 수")
    parser.add_argument("--max-steps", type=int, default=300)
    
    args = parser.parse_args()
    
    # SimplerEnv 4개 task
    tasks = [
        "PutSpoonOnTableClothInScene-v1",
        "PutCarrotOnPlateInScene-v1",
        "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
        "PutEggplantInBasketScene-v1"
    ]
    
    # 정책 로드
    print(f"Loading policy: {args.model}")
    policy = load_policy(args.model)
    
    # 수집 실행
    all_trajectories = []
    print("\n" + "="*60)
    print(f"Collecting {args.n_per_task} trajectories per task")
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
    
    model_name = Path(args.model).stem if args.model != "simple" else "simple"
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