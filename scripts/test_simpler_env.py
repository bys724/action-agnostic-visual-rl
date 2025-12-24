"""
SIMPLER 환경 테스트 스크립트
기본 환경 로드, 리셋, 랜덤 액션 실행을 통해 설치 확인
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'SimplerEnv'))

import argparse
import numpy as np
import time

try:
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    print("✓ SIMPLER 임포트 성공")
except ImportError as e:
    print(f"✗ SIMPLER 임포트 실패: {e}")
    print("설치 가이드를 따라 SIMPLER를 설치하세요:")
    print("  cd third_party/SimplerEnv/ManiSkill2_real2sim && pip install -e .")
    print("  cd ../ && pip install -e .")
    sys.exit(1)

def test_environment(env_name='google_robot_pick_coke_can', n_steps=50):
    """환경 테스트 함수"""
    
    print(f"\n{'='*50}")
    print(f"환경 테스트: {env_name}")
    print(f"{'='*50}")
    
    # 환경 생성
    try:
        env = simpler_env.make(env_name)
        print(f"✓ 환경 생성 성공: {env_name}")
    except Exception as e:
        print(f"✗ 환경 생성 실패: {e}")
        return False
    
    # 환경 리셋
    try:
        obs, reset_info = env.reset()
        instruction = env.get_language_instruction()
        print(f"✓ 환경 리셋 성공")
        print(f"  - 지시사항: {instruction}")
        print(f"  - 리셋 정보: {reset_info}")
    except Exception as e:
        print(f"✗ 환경 리셋 실패: {e}")
        return False
    
    # 행동 공간 확인
    print(f"\n행동 공간 정보:")
    print(f"  - Shape: {env.action_space.shape}")
    print(f"  - Low: {env.action_space.low}")
    print(f"  - High: {env.action_space.high}")
    
    # 이미지 추출 테스트
    try:
        image = get_image_from_maniskill2_obs_dict(env, obs)
        print(f"\n✓ 이미지 추출 성공")
        print(f"  - Shape: {image.shape}")
        print(f"  - dtype: {image.dtype}")
        print(f"  - 범위: [{image.min()}, {image.max()}]")
    except Exception as e:
        print(f"✗ 이미지 추출 실패: {e}")
        return False
    
    # 랜덤 액션 실행
    print(f"\n{n_steps}스텝 랜덤 액션 실행...")
    
    success_count = 0
    done, truncated = False, False
    step = 0
    
    start_time = time.time()
    
    while step < n_steps and not (done or truncated):
        # 랜덤 액션 생성
        action = env.action_space.sample()
        
        try:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            
            # 매 10스텝마다 상태 출력
            if step % 10 == 0:
                print(f"  스텝 {step}: reward={reward:.3f}, done={done}, truncated={truncated}")
            
            # 성공 체크
            if done and info.get('success', False):
                success_count += 1
                
        except Exception as e:
            print(f"✗ 스텝 {step}에서 오류: {e}")
            return False
    
    elapsed = time.time() - start_time
    fps = step / elapsed
    
    print(f"\n실행 완료:")
    print(f"  - 총 스텝: {step}")
    print(f"  - 소요 시간: {elapsed:.2f}초")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - 성공: {success_count > 0}")
    
    # 에피소드 통계
    if 'episode_stats' in info:
        print(f"\n에피소드 통계:")
        for key, value in info['episode_stats'].items():
            print(f"  - {key}: {value}")
    
    return True

def test_multiple_environments():
    """여러 환경 테스트"""
    
    test_envs = [
        'google_robot_pick_coke_can',
        'google_robot_pick_horizontal_coke_can',
        'widowx_spoon_on_towel',
        'widowx_carrot_on_plate',
    ]
    
    available_envs = []
    
    print("\n사용 가능한 환경 확인:")
    for env_name in test_envs:
        try:
            env = simpler_env.make(env_name)
            env.close()
            available_envs.append(env_name)
            print(f"  ✓ {env_name}")
        except Exception as e:
            print(f"  ✗ {env_name}: {e}")
    
    return available_envs

def main():
    parser = argparse.ArgumentParser(description='SIMPLER 환경 테스트')
    parser.add_argument('--env', type=str, default='google_robot_pick_coke_can',
                        help='테스트할 환경 이름')
    parser.add_argument('--steps', type=int, default=50,
                        help='실행할 스텝 수')
    parser.add_argument('--list-envs', action='store_true',
                        help='사용 가능한 환경 목록 확인')
    args = parser.parse_args()
    
    print("SIMPLER 환경 테스트 시작...")
    print(f"Python: {sys.version}")
    print(f"Numpy: {np.__version__}")
    
    # GPU 확인
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch 미설치")
    
    if args.list_envs:
        # 환경 목록 확인
        available = test_multiple_environments()
        print(f"\n총 {len(available)}개 환경 사용 가능")
        return
    
    # 단일 환경 테스트
    success = test_environment(args.env, args.steps)
    
    if success:
        print("\n✓ 테스트 완료! SIMPLER 환경이 정상적으로 작동합니다.")
    else:
        print("\n✗ 테스트 실패. 설치를 확인하세요.")
        sys.exit(1)

if __name__ == "__main__":
    main()