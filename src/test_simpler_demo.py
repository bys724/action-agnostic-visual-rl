#!/usr/bin/env python
"""
SimplerEnv demo with pre-trained model
Tests the environment setup and optionally displays GUI visualization
"""

import os
# Disable Vulkan for ManiSkill3 (use software rendering)
os.environ["SAPIEN_DISABLE_VULKAN"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import numpy as np
import torch
import cv2
import gymnasium as gym
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *
from pathlib import Path
from datetime import datetime
import time


class SimplePolicy:
    """Simple policy for demonstration - can be replaced with trained model"""
    
    def __init__(self):
        self.step_count = 0
        self.phase = 0
        self.phase_steps = 0
        
    def get_action(self, obs):
        """Generate simple action sequence for demonstration"""
        # Simple scripted policy for demonstration
        # In real use, load your trained model here
        
        sequences = [
            # Move down to object
            (20, np.array([0, 0, -0.01, 0, 0, 0, 1.0])),
            # Close gripper
            (10, np.array([0, 0, 0, 0, 0, 0, -1.0])),
            # Lift up
            (20, np.array([0, 0, 0.01, 0, 0, 0, -1.0])),
            # Move to target
            (30, np.array([0.005, 0.005, 0, 0, 0, 0, -1.0])),
            # Lower
            (20, np.array([0, 0, -0.005, 0, 0, 0, -1.0])),
            # Release
            (10, np.array([0, 0, 0, 0, 0, 0, 1.0])),
        ]
        
        if self.phase >= len(sequences):
            self.phase = 0
            self.phase_steps = 0
            
        duration, action = sequences[self.phase]
        self.phase_steps += 1
        
        if self.phase_steps >= duration:
            self.phase += 1
            self.phase_steps = 0
            
        self.step_count += 1
        return action.astype(np.float32)


def run_demo(env_name="PutSpoonOnTableClothInScene-v1", gui=False, max_steps=300, save_video=None):
    """
    Run SimplerEnv demo

    Args:
        env_name: SimplerEnv environment name
        gui: Whether to show GUI visualization
        max_steps: Maximum number of steps to run
        save_video: Path to save video file (e.g., "output.mp4")
    """
    
    print("="*60)
    print(f"SimplerEnv Demo - Environment Test")
    print(f"Environment: {env_name}")
    print(f"GUI: {'Enabled' if gui else 'Disabled'}")
    print(f"Save Video: {save_video if save_video else 'Disabled'}")
    print("="*60)
    
    try:
        # Create environment
        print(f"\nInitializing environment...")
        env = gym.make(
            env_name,
            obs_mode="rgb+segmentation",
            num_envs=1,  # Single env for demo
        )
        obs, info = env.reset()
        print(f"✓ Environment created successfully")
        
        # Get language instruction
        instruction = env.unwrapped.get_language_instruction()
        print(f"✓ Task instruction: {instruction[0]}")
        
        # Initialize policy (replace with your trained model)
        policy = SimplePolicy()
        print(f"✓ Policy initialized")
        
        # Setup GUI if enabled
        if gui:
            cv2.namedWindow('SimplerEnv Demo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('SimplerEnv Demo', 800, 600)
            print(f"✓ GUI window created")

        # Setup video writer if enabled
        video_writer = None
        if save_video:
            video_dir = Path(save_video).parent
            if video_dir and not video_dir.exists():
                video_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Video will be saved to: {save_video}")

        # Run simulation
        print(f"\nRunning simulation for {max_steps} steps...")
        print("Press 'q' to quit early (GUI mode only)\n")
        
        total_reward = 0.0
        for step in range(max_steps):
            # Get action from policy
            action = policy.get_action(obs)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Process reward
            if hasattr(reward, 'item'):
                reward = reward.item()
            else:
                reward = float(reward)
            total_reward += reward
            
            # Display progress
            if step % 10 == 0:
                print(f"Step {step:3d}/{max_steps} | Reward: {reward:7.4f} | Total: {total_reward:7.4f}")
            
            # GUI visualization and/or video recording
            if gui or save_video:
                # Extract image from observation using ManiSkill3 utility
                rgb_image = get_image_from_maniskill3_obs_dict(env, obs)

                # rgb_image shape: (num_envs, H, W, 3)
                if len(rgb_image.shape) == 4:
                    rgb_image = rgb_image[0]  # Take first env

                # Convert from torch to numpy if needed
                if isinstance(rgb_image, torch.Tensor):
                    rgb_image = rgb_image.cpu().numpy()

                # Ensure uint8
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)

                # Convert RGB to BGR for OpenCV
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                # Add info overlay
                cv2.putText(bgr_image, f"Step: {step}/{max_steps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(bgr_image, f"Reward: {reward:.4f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(bgr_image, f"Total: {total_reward:.4f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Initialize video writer on first frame (to get correct dimensions)
                if save_video and video_writer is None:
                    h, w = bgr_image.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(save_video, fourcc, 30, (w, h))

                # Write frame to video
                if video_writer is not None:
                    video_writer.write(bgr_image)

                # Show image if GUI enabled
                if gui:
                    cv2.imshow('SimplerEnv Demo', bgr_image)

                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nQuitting early...")
                        break
            
            # Reset if episode ends
            if done or truncated:
                print(f"\nEpisode ended at step {step}")
                obs, info = env.reset()
                total_reward = 0.0
                policy = SimplePolicy()  # Reset policy state
        
        print(f"\n✓ Demo completed successfully!")
        print(f"Final total reward: {total_reward:.4f}")

        # Cleanup
        if video_writer is not None:
            video_writer.release()
            print(f"✓ Video saved to: {save_video}")
        if gui:
            cv2.destroyAllWindows()
        env.close()

        return True
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()

        if 'video_writer' in locals() and video_writer is not None:
            video_writer.release()
        if gui:
            cv2.destroyAllWindows()

        return False


def main():
    parser = argparse.ArgumentParser(description="SimplerEnv Demo - Test environment setup")
    parser.add_argument('--env', type=str, default='PutSpoonOnTableClothInScene-v1',
                       help='Environment name (default: PutSpoonOnTableClothInScene-v1)')
    parser.add_argument('--gui', action='store_true',
                       help='Enable GUI visualization')
    parser.add_argument('--steps', type=int, default=300,
                       help='Number of steps to run (default: 300)')
    parser.add_argument('--save-video', nargs='?', const='auto', default=None,
                       help='Save video to data/results/ (optional: specify filename)')

    args = parser.parse_args()

    # Process save_video path
    save_video = None
    if args.save_video is not None:
        results_dir = Path('data/results')
        results_dir.mkdir(parents=True, exist_ok=True)

        if args.save_video == 'auto':
            # Auto-generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_video = str(results_dir / f'demo_{timestamp}.mp4')
        elif '/' in args.save_video or '\\' in args.save_video:
            # Full path provided
            save_video = args.save_video
        else:
            # Filename only - save to data/results/
            save_video = str(results_dir / args.save_video)

    # Run demo
    success = run_demo(env_name=args.env, gui=args.gui, max_steps=args.steps, save_video=save_video)
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()