#!/usr/bin/env python
"""
Simple test for ManiSkill3 environments that are available
"""

import numpy as np
import gymnasium as gym
import mani_skill.envs

def test_available_env():
    print("=" * 60)
    print("Testing Available ManiSkill3 Environments")
    print("=" * 60)
    
    # Test environments that we know exist from the check
    test_envs = [
        "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
        "PutCarrotOnPlateInScene-v1", 
        "PutSpoonOnTableClothInScene-v1",
        "PutEggplantInBasketScene-v1",
        "StackCube-v1"
    ]
    
    for env_name in test_envs:
        print(f"\nTesting: {env_name}")
        print("-" * 40)
        
        try:
            # Create environment directly with gym.make
            env = gym.make(env_name, obs_mode="rgbd", render_mode="rgb_array")
            print("✓ Environment created")
            
            # Reset
            obs, info = env.reset()
            print("✓ Reset successful")
            
            # Check observation
            if isinstance(obs, dict):
                print(f"  Observation keys: {list(obs.keys())}")
                if 'image' in obs:
                    print(f"  Image shape: {obs['image'].shape if hasattr(obs['image'], 'shape') else type(obs['image'])}")
            
            # Take a few steps
            for i in range(3):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                print(f"  Step {i+1}: reward={reward:.4f}")
                
                if done or truncated:
                    break
            
            env.close()
            print("✓ Test completed")
            
            # If we found a working environment, use it for SimplerEnv
            print(f"\n✅ Found working environment: {env_name}")
            return env_name
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    print("\n❌ No working environments found")
    return None

def test_simpler_env():
    """Test SimplerEnv with ManiSkill3 backend"""
    print("\n" + "=" * 60)
    print("Testing SimplerEnv with ManiSkill3")
    print("=" * 60)
    
    try:
        import simpler_env
        
        # Try WidowX environments which seem to be available
        test_tasks = [
            "widowx_stack_cube",
            "widowx_carrot_on_plate",
            "widowx_spoon_on_towel",
            "widowx_put_eggplant_in_basket"
        ]
        
        for task_name in test_tasks:
            print(f"\nTrying SimplerEnv task: {task_name}")
            try:
                env = simpler_env.make(task_name)
                print(f"✓ Created {task_name}")
                
                obs, info = env.reset()
                print("✓ Reset successful")
                
                # Take one step
                action = np.zeros(7, dtype=np.float32)
                obs, reward, done, truncated, info = env.step(action)
                print(f"✓ Step successful: reward={reward:.4f}")
                
                env.close()
                print(f"✅ {task_name} works!")
                return task_name
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                continue
                
    except ImportError as e:
        print(f"✗ SimplerEnv not available: {e}")
    
    return None

if __name__ == "__main__":
    # First test raw ManiSkill3 environments
    working_env = test_available_env()
    
    # Then test SimplerEnv wrapper
    working_task = test_simpler_env()
    
    if working_task:
        print(f"\n✅ Success! You can use: simpler_env.make('{working_task}')")
    elif working_env:
        print(f"\n✅ Success! You can use: gym.make('{working_env}', obs_mode='rgbd')")
    else:
        print("\n❌ No working environments found. Check installation.")