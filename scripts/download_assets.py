#!/usr/bin/env python
"""
Download all required ManiSkill assets for SimplerEnv evaluation
"""

import os
import sys

def download_assets():
    """Download all required assets for SimplerEnv"""
    
    print("Downloading required ManiSkill assets...")
    
    # Set auto-download environment variable
    os.environ["MS_ASSET_DIR"] = "/root/.maniskill"
    
    # Assets to download
    assets = [
        "bridge_v2_real2sim",
        # Add more assets as needed
    ]
    
    for asset in assets:
        print(f"\nDownloading {asset}...")
        cmd = f"echo 'y' | python -m mani_skill.utils.download_asset {asset}"
        result = os.system(cmd)
        if result != 0:
            print(f"Warning: Could not download {asset}")
    
    print("\nAsset download complete!")
    print("Note: Additional robot assets will be auto-downloaded on first use.")

if __name__ == "__main__":
    download_assets()