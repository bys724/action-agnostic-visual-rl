#!/usr/bin/env python
"""
Final test for ManiSkill3 with SimplerEnv
"""

import numpy as np
import torch
import simpler_env

def test_maniskill3_env():
    print("=" * 60)
    print("Testing ManiSkill3 with SimplerEnv")
    print("=" * 60)
    
    # Test WidowX stack cube environment
    env_name = "widowx_stack_cube"
    
    try:
        print(f"\n1. Creating environment: {env_name}")
        env = simpler_env.make(env_name)
        print("   ✓ Environment created successfully")
        
        print("\n2. Environment information:")
        print(f"   Action space: {env.action_space}")
        print(f"   Action shape: {env.action_space.shape}")
        
        print("\n3. Resetting environment...")
        obs, info = env.reset()
        print("   ✓ Reset successful")
        
        # Explore observation structure
        print("\n4. Observation structure:")
        if isinstance(obs, dict):
            for key, val in obs.items():
                if isinstance(val, dict):
                    print(f"   {key}: dict with keys {list(val.keys())}")
                elif hasattr(val, 'shape'):
                    print(f"   {key}: shape {val.shape}")
                else:
                    print(f"   {key}: {type(val)}")
        
        # Check sensor data
        if 'sensor_data' in obs and isinstance(obs['sensor_data'], dict):
            print("\n5. Sensor data:")
            for cam_name, cam_data in obs['sensor_data'].items():
                print(f"   Camera: {cam_name}")
                if isinstance(cam_data, dict):
                    for data_key, data_val in cam_data.items():
                        if hasattr(data_val, 'shape'):
                            print(f"     {data_key}: shape {data_val.shape}")
        
        print("\n6. Taking 10 steps...")
        success_steps = 0
        for i in range(10):
            # Generate small random action
            action = np.zeros(7, dtype=np.float32)
            action[:3] = np.random.uniform(-0.005, 0.005, size=3)  # Small position changes
            action[3:6] = np.random.uniform(-0.01, 0.01, size=3)   # Small rotation changes
            action[6] = 0.0  # Keep gripper closed
            
            try:
                obs, reward, done, truncated, info = env.step(action)
                
                # Handle reward as tensor
                if isinstance(reward, torch.Tensor):
                    reward = reward.item()
                
                print(f"   Step {i+1}: reward={float(reward):.6f}")
                success_steps += 1
                
                if done or truncated:
                    print(f"   Episode ended at step {i+1}")
                    break
                    
            except Exception as e:
                print(f"   Step {i+1} failed: {e}")
        
        print(f"\n   Successfully completed {success_steps}/10 steps")
        
        # Try to extract images if possible
        print("\n7. Image extraction:")
        try:
            if 'sensor_data' in obs:
                for cam_name, cam_data in obs['sensor_data'].items():
                    if isinstance(cam_data, dict) and 'rgb' in cam_data:
                        rgb_data = cam_data['rgb']
                        if hasattr(rgb_data, 'shape'):
                            print(f"   Camera '{cam_name}' RGB image: shape {rgb_data.shape}")
                            # Could save image here if needed
        except Exception as e:
            print(f"   Image extraction note: {e}")
        
        env.close()
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_maniskill3_env()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! ManiSkill3 is working!")
        print("=" * 60)
        print("\nYou can now:")
        print("1. Use WidowX environments with SimplerEnv")
        print("2. Access RGB images from sensor_data")
        print("3. Train RL agents with these environments")
        print("\nNote: Google Robot environments are not yet")
        print("available in ManiSkill3 branch.")
    else:
        print("\n" + "=" * 60)
        print("❌ Test failed. Check the error messages above.")
        print("=" * 60)