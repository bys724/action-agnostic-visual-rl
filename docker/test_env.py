#!/usr/bin/env python
"""
Docker 환경 테스트 스크립트
컨테이너 내에서 실행하여 SIMPLER가 정상 작동하는지 확인
"""

import os
# Vulkan 렌더링 비활성화, CPU 렌더링 사용
os.environ['SAPIEN_DISABLE_VULKAN'] = '1'
os.environ['SAPIEN_RENDERER'] = 'offscreen'

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import numpy as np

def test_simpler_env():
    print("="*50)
    print("SIMPLER Docker Environment Test")
    print("="*50)
    
    # 환경 생성
    env_name = 'google_robot_pick_coke_can'
    print(f"\n1. Creating environment: {env_name}")
    env = simpler_env.make(env_name, max_episode_steps=50)
    print("   ✓ Environment created")
    
    # 리셋
    print("\n2. Resetting environment")
    obs, info = env.reset()
    print("   ✓ Reset successful")
    
    # 작업 정보
    instruction = env.unwrapped.get_language_instruction()
    print(f"\n3. Task: {instruction}")
    
    # 이미지 추출
    print("\n4. Extracting image")
    image = get_image_from_maniskill2_obs_dict(env, obs)
    print(f"   ✓ Image shape: {image.shape}")
    
    # 에피소드 실행
    print("\n5. Running episode (10 steps)")
    total_reward = 0
    
    for i in range(10):
        # 랜덤 액션
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 3 == 0:
            print(f"   Step {i+1}: reward={reward:.3f}, total={total_reward:.3f}")
        
        if done or truncated:
            print(f"   Episode ended at step {i+1}")
            break
    
    # 결과
    success = info.get('success', False)
    print(f"\n6. Results:")
    print(f"   - Total reward: {total_reward:.3f}")
    print(f"   - Success: {'✓' if success else '✗'}")
    
    env.close()
    print("\n" + "="*50)
    print("✅ All tests passed! Docker environment is working.")
    print("="*50)

if __name__ == "__main__":
    try:
        test_simpler_env()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nDocker environment test failed.")
        exit(1)