#!/usr/bin/env python3
"""
Fix JAX compatibility issue in octo package
jax.random.KeyArray -> jax.Array (for JAX >= 0.4.16)
"""

import os

def fix_octo_typing():
    """Fix the typing.py file in octo package"""
    typing_file = "/usr/local/lib/python3.10/dist-packages/octo/utils/typing.py"
    
    if not os.path.exists(typing_file):
        print(f"Error: {typing_file} not found")
        return False
    
    # Read the file
    with open(typing_file, 'r') as f:
        content = f.read()
    
    # Replace the problematic line
    old_line = "PRNGKey = jax.random.KeyArray"
    new_line = "PRNGKey = jax.Array  # Fixed for JAX >= 0.4.16"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write back
        with open(typing_file, 'w') as f:
            f.write(content)
        
        print(f"✓ Fixed {typing_file}")
        return True
    else:
        print(f"Already fixed or different content in {typing_file}")
        return False

if __name__ == "__main__":
    fix_octo_typing()