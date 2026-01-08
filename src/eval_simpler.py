#!/usr/bin/env python
"""
SimplerEnv 평가 스크립트
모델 로드 → 평가 → 결과 저장
"""

import argparse
import json
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


def load_policy(model_path):
    """정책 로드"""
    if model_path == "simple":
        return SimplePolicy()
    
    # SimplerEnv 베이스라인 모델 로드
    policy_setup = "widowx_bridge"  # WidowX 로봇 사용
    
    if "octo" in model_path.lower():
        from simpler_env.policies.octo.octo_model import OctoInference
        # octo-base 또는 octo-small
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


def evaluate_model(policy, task_name, n_episodes=24, max_steps=300):
    """단일 task에서 모델 평가"""
    env = gym.make(task_name, obs_mode="rgb+segmentation", num_envs=1)
    successes = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        instruction = env.unwrapped.get_language_instruction()[0]
        
        # Policy 초기화
        if hasattr(policy, 'reset'):
            policy.reset(instruction)
        elif hasattr(policy, 'phase'):  # SimplePolicy
            policy.phase = 0
            policy.phase_steps = 0
        
        for step in range(max_steps):
            # 이미지 추출
            image = get_image_from_maniskill3_obs_dict(env, obs)
            
            # 액션 생성
            if hasattr(policy, 'step'):  # RT-1, Octo 모델
                raw_action, action = policy.step(image, instruction)
                # terminate_episode 체크
                if action.get("terminate_episode", [0])[0] > 0:
                    break
                # 실제 환경 액션 추출
                env_action = action["world_vector"]
            else:  # SimplePolicy
                env_action = policy.get_action(obs)
            
            # 환경 스텝
            obs, reward, terminated, truncated, info = env.step(env_action)
            
            if terminated or truncated:
                break
        
        # 성공 여부 확인
        if hasattr(info, '__iter__'):
            success = info.get("success", [False])[0] if isinstance(info.get("success"), list) else info.get("success", False)
        else:
            success = False
        
        successes.append(float(success))
        print(f"  Episode {episode+1}/{n_episodes}: {'Success' if success else 'Fail'}")
    
    env.close()
    return np.mean(successes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="simple",
                       help="모델 체크포인트 경로 또는 'simple' for test")
    parser.add_argument("--output", type=str, default="./data/results",
                       help="결과 저장 디렉토리")
    parser.add_argument("--n-episodes", type=int, default=24)
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
    
    # 평가 실행
    results = {}
    print("\n" + "="*60)
    for task in tasks:
        print(f"\nEvaluating {task}...")
        success_rate = evaluate_model(policy, task, args.n_episodes, args.max_steps)
        results[task] = success_rate
        print(f"  Success rate: {success_rate:.2%}")
    
    # 평균 계산
    results["average"] = np.mean(list(results.values()))
    
    print("\n" + "="*60)
    print(f"Overall Average: {results['average']:.2%}")
    print("="*60)
    
    # 결과 저장
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(args.model).stem if args.model != "simple" else "simple"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{model_name}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()