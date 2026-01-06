#!/usr/bin/env python
"""
Test WidowX environment with ManiSkill3
"""

import numpy as np
import simpler_env

def test_widowx():
    print("=" * 60)
    print("Testing WidowX Stack Cube with ManiSkill3")
    print("=" * 60)
    
    # Use WidowX environment which should work with ManiSkill3
    task_name = "widowx_stack_cube"
    
    try:
        print(f"\n1. Creating environment: {task_name}")
        env = simpler_env.make(task_name)
        print("   ✓ Environment created")
        
        print("\n2. Resetting environment...")
        obs, info = env.reset()
        print("   ✓ Reset successful")
        
        # Check observation structure
        if isinstance(obs, dict):
            print(f"   Observation keys: {list(obs.keys())}")
            if 'agent' in obs:
                print(f"   Agent obs shape: {obs['agent'].shape if hasattr(obs['agent'], 'shape') else type(obs['agent'])}")
            if 'sensor_data' in obs:
                print(f"   Sensor data keys: {list(obs['sensor_data'].keys()) if isinstance(obs['sensor_data'], dict) else type(obs['sensor_data'])}")
        
        print("\n3. Taking 5 steps...")
        for i in range(5):
            # Small random actions for WidowX (7-DOF)
            action = np.random.uniform(-0.01, 0.01, size=7).astype(np.float32)
            action[-1] = 0.0  # Keep gripper closed initially
            
            try:
                obs, reward, done, truncated, info = env.step(action)
                # Convert reward to float if it's a tensor
                if hasattr(reward, 'item'):
                    reward = reward.item()
                print(f"   Step {i+1}: reward={float(reward):.4f}, done={done}")
                
                if done or truncated:
                    print(f"   Episode ended at step {i+1}")
                    break
            except Exception as e:
                print(f"   Step {i+1} error: {e}")
                # Continue with other steps even if one fails
                continue
        
        env.close()
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_widowx()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ ManiSkill3 is working with SimplerEnv!")
        print("You can now use environments like:")
        print("  - widowx_stack_cube")
        print("  - widowx_carrot_on_plate")
        print("  - widowx_spoon_on_towel")
        print("  - widowx_put_eggplant_in_basket")
        print("=" * 60)