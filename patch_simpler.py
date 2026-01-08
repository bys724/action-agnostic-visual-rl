#!/usr/bin/env python
"""
Patch script to fix SimplerEnv's import issues with ManiSkill2
"""

import os
import sys

def patch_simpler_init():
    """Patch SimplerEnv __init__.py to use mani_skill2 instead of mani_skill"""
    
    init_file = "/tmp/SimplerEnv/simpler_env/__init__.py"
    
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            content = f.read()
        
        # Replace mani_skill with mani_skill2
        content = content.replace("import mani_skill.envs", "import mani_skill2.envs")
        
        with open(init_file, 'w') as f:
            f.write(content)
        
        print(f"✓ Patched {init_file}")
        return True
    else:
        print(f"✗ File not found: {init_file}")
        return False

if __name__ == "__main__":
    success = patch_simpler_init()
    sys.exit(0 if success else 1)