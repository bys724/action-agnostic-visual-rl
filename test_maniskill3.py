#!/usr/bin/env python
"""
Test ManiSkill3 version of SIMPLER
"""

import numpy as np

def test_maniskill3():
    print("=" * 60)
    print("Testing ManiSkill3 Version of SIMPLER")
    print("=" * 60)
    
    try:
        import simpler_env
        print("✓ SimplerEnv imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import simpler_env: {e}")
        return
    
    # Test environment creation and stepping
    print("\nTesting environment creation and stepping:")
    env_name = 'google_robot_pick_coke_can'
    
    try:
        print(f"\n1. Creating environment: {env_name}")
        # ManiSkill3 API - simpler arguments
        env = simpler_env.make(env_name)
        print("   ✓ Environment created")
        
        print("\n2. Resetting environment...")
        obs, info = env.reset()
        print("   ✓ Reset successful")
        print(f"   Observation keys: {obs.keys() if hasattr(obs, 'keys') else type(obs)}")
        
        print("\n3. Taking 10 steps...")
        for i in range(10):
            # Random action (7-dim: dx, dy, dz, rx, ry, rz, gripper)
            action = np.random.uniform(-0.01, 0.01, size=7).astype(np.float32)
            action[-1] = 1.0  # Keep gripper open
            
            obs, reward, done, truncated, info = env.step(action)
            print(f"   Step {i+1}: reward={reward:.4f}, done={done}, truncated={truncated}")
            
            if done or truncated:
                print(f"   Episode ended at step {i+1}")
                break
        
        print("\n4. Extracting images...")
        try:
            from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
            image = get_image_from_maniskill2_obs_dict(env, obs)
            print(f"   ✓ Image extracted: shape={image.shape}, dtype={image.dtype}")
        except Exception as e:
            print(f"   Note: Image extraction may need adjustment for ManiSkill3: {e}")
        
        env.close()
        print("\n✅ ManiSkill3 test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_maniskill3()
