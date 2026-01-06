#!/usr/bin/env python
"""
간단한 SIMPLER 환경 테스트 (렌더링 없이)
"""

import os
# 렌더링 관련 설정
os.environ['SAPIEN_DISABLE_VULKAN'] = '1'

import simpler_env
import numpy as np

def test_simpler_basic():
    print("="*50)
    print("SIMPLER Basic Test (No Rendering)")
    print("="*50)
    
    # 환경 생성
    env_name = 'google_robot_pick_coke_can'
    print(f"\n1. Creating environment: {env_name}")
    
    try:
        # obs_mode를 state로 설정하여 이미지 렌더링 회피
        env = simpler_env.make(env_name, obs_mode='state', render_mode=None)
        print("   ✓ Environment created")
    except Exception as e:
        print(f"   ✗ Failed to create environment: {e}")
        return False
    
    # 리셋
    print("\n2. Resetting environment")
    try:
        obs, info = env.reset()
        print("   ✓ Reset successful")
        print(f"   - Observation type: {type(obs)}")
    except Exception as e:
        print(f"   ✗ Reset failed: {e}")
        env.close()
        return False
    
    # 작업 정보
    try:
        instruction = env.unwrapped.get_language_instruction()
        print(f"\n3. Task: {instruction}")
    except:
        print("\n3. Task info not available")
    
    # 액션 공간
    print(f"\n4. Action space: {env.action_space.shape}")
    
    # 간단한 스텝 실행
    print("\n5. Running 5 test steps")
    for i in range(5):
        action = np.zeros(env.action_space.shape[0])  # 정지 액션
        try:
            obs, reward, done, truncated, info = env.step(action)
            print(f"   Step {i+1}: reward={reward:.3f}")
            if done or truncated:
                break
        except Exception as e:
            print(f"   Step {i+1} failed: {e}")
            break
    
    env.close()
    print("\n" + "="*50)
    print("✅ Basic test completed!")
    print("="*50)
    return True

if __name__ == "__main__":
    try:
        success = test_simpler_basic()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)