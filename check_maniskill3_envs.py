#!/usr/bin/env python
"""
Check available environments in ManiSkill3
"""

import gymnasium as gym

def check_environments():
    print("=" * 60)
    print("Checking ManiSkill3 Registered Environments")
    print("=" * 60)
    
    # Import ManiSkill3 to register environments
    try:
        import mani_skill
        import mani_skill.envs
        print("✓ ManiSkill3 imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import mani_skill: {e}")
        return
    
    # List all registered environments
    print("\nRegistered Gymnasium environments:")
    all_envs = gym.envs.registry.keys()
    
    # Filter ManiSkill related environments
    maniskill_envs = [env for env in all_envs if 'Scene' in env or 'Grasp' in env or 'Drawer' in env or 'Move' in env or 'Place' in env or 'Stack' in env or 'Put' in env]
    
    if maniskill_envs:
        print(f"\nFound {len(maniskill_envs)} ManiSkill-related environments:")
        for env in sorted(maniskill_envs)[:20]:  # Show first 20
            print(f"  - {env}")
        if len(maniskill_envs) > 20:
            print(f"  ... and {len(maniskill_envs) - 20} more")
    else:
        print("\nNo ManiSkill Scene environments found. Showing all environments:")
        for i, env in enumerate(sorted(all_envs)[:30]):
            print(f"  {i+1}. {env}")
        print(f"  ... total {len(all_envs)} environments")
    
    # Try to check SimplerEnv specifically
    print("\n" + "=" * 60)
    print("Checking SimplerEnv:")
    try:
        import simpler_env
        print("✓ SimplerEnv imported")
        print(f"Available tasks: {simpler_env.ENVIRONMENTS[:5]} ...")
    except ImportError as e:
        print(f"✗ SimplerEnv import failed: {e}")

if __name__ == "__main__":
    check_environments()