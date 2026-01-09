"""
OpenVLA policy wrapper for SimplerEnv evaluation
Based on: https://github.com/DelinQu/SimplerEnv-OpenVLA
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from transforms3d.euler import euler2axangle
import os
import json


class OpenVLAPolicy:
    """OpenVLA policy for SimplerEnv evaluation"""
    
    def __init__(
        self, 
        model_path: str = "openvla/openvla-7b",
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
        policy_setup: str = "widowx_bridge",
        use_local: bool = True
    ):
        """
        Initialize OpenVLA model
        
        Args:
            model_path: Path to model checkpoint or HuggingFace model ID
            device: Device to run model on
            torch_dtype: Data type for model weights
            policy_setup: Robot setup (widowx_bridge or google_robot)
            use_local: Try to use local checkpoint first if available
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.policy_setup = policy_setup
        
        # Disable tokenizer parallelism warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Check for local checkpoint first
        actual_model_path = self._resolve_model_path(model_path, use_local)
        
        print(f"Loading OpenVLA model from {actual_model_path}...")
        if actual_model_path != model_path:
            print(f"  (Using local checkpoint)")
        
        # Setup unnorm key based on policy
        if policy_setup == "widowx_bridge":
            self.unnorm_key = "bridge_orig"
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            self.unnorm_key = "fractal20220817_data"
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(f"Policy setup {policy_setup} not supported")
        
        # Load processor and model
        try:
            self.processor = AutoProcessor.from_pretrained(
                actual_model_path, 
                trust_remote_code=True,
                local_files_only=os.path.exists(actual_model_path)  # Use local if exists
            )
            
            # Use AutoModelForVision2Seq for OpenVLA
            self.model = AutoModelForVision2Seq.from_pretrained(
                actual_model_path,
                # attn_implementation="flash_attention_2",  # Optional, requires flash_attn
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                local_files_only=os.path.exists(actual_model_path)  # Use local if exists
            ).to(device)
            
            self.model.eval()
            print("OpenVLA model loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load actual model, using mock. Error: {e}")
            # Use a mock model for testing without GPU
            self.model = None
            self.processor = None
        
        # Action space parameters for SimplerEnv
        self.action_dim = 7  # [dx, dy, dz, rx, ry, rz, gripper]
        # Increase action scale for better movement
        self.action_scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # Cache for current instruction
        self.current_instruction = None
        
        # Sticky gripper state for google_robot
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
    
    def _resolve_model_path(self, model_path: str, use_local: bool) -> str:
        """
        Resolve model path - check local first if enabled
        
        Args:
            model_path: Original model path or HuggingFace ID
            use_local: Whether to check local checkpoints first
            
        Returns:
            Actual path to use for loading
        """
        if not use_local:
            return model_path
        
        # Check if it's already a local path
        if os.path.exists(model_path):
            return model_path
        
        # Check in data/checkpoints directory
        from pathlib import Path
        import json
        
        # Try to load from registry
        registry_file = Path("./data/checkpoints/registry.json")
        if registry_file.exists():
            with open(registry_file, "r") as f:
                registry = json.load(f)
            
            # Extract model name from HuggingFace ID
            if "/" in model_path:
                model_name = model_path.split("/")[-1]
                
                # Check all model types
                for model_type in registry.get("models", {}):
                    if model_name in registry["models"][model_type]:
                        local_path = registry["models"][model_type][model_name]["path"]
                        if os.path.exists(local_path):
                            return local_path
        
        # Check default locations
        default_paths = [
            f"./data/checkpoints/openvla/{model_path.split('/')[-1]}",
            f"./data/checkpoints/{model_path.replace('/', '_')}",
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                return path
        
        # Fallback to original path (will download from HuggingFace)
        return model_path
    
    def reset(self, instruction: str):
        """Reset policy with new instruction"""
        self.current_instruction = instruction
    
    def step(self, image: np.ndarray, instruction: Optional[str] = None) -> tuple:
        """
        Generate action from image observation
        
        Args:
            image: RGB image observation (H, W, 3) as numpy array or torch tensor
            instruction: Language instruction (optional, uses cached if not provided)
            
        Returns:
            tuple: (raw_action, action_dict)
                - raw_action: 7D action array
                - action_dict: Dictionary with 'world_vector' and 'terminate_episode' keys
        """
        # Use provided instruction or cached one
        if instruction is not None:
            self.current_instruction = instruction
        
        if self.current_instruction is None:
            raise ValueError("No instruction provided")
        
        # Convert to PIL Image
        # Handle torch tensors (from ManiSkill3)
        if torch.is_tensor(image):
            # Convert torch tensor to numpy
            image = image.cpu().numpy()
            
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            # Remove batch dimensions if present
            while len(image.shape) > 3:
                image = image.squeeze(0)
            
            # Ensure correct shape and dtype
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Convert from [0, 1] to [0, 255]
                image = (image * 255).astype(np.uint8)
            elif image.max() <= 1.0 and image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Handle different channel orders
            if len(image.shape) == 3:
                if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                    image = np.transpose(image, (1, 2, 0))
                elif image.shape[0] == 480 and image.shape[2] == 3:
                    # Already in (H, W, C) format
                    pass
            
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Check if model is loaded
        if self.model is None or self.processor is None:
            # Return mock action for testing
            action = np.random.randn(7) * 0.001
            scaled_action = action * self.action_scale
            action_dict = {
                "world_vector": scaled_action.astype(np.float32),
                "terminate_episode": np.array([0])
            }
            return action, action_dict
        
        # Prepare inputs - OpenVLA expects just the task description
        prompt = self.current_instruction
        inputs = self.processor(prompt, pil_image).to(self.device, dtype=self.torch_dtype)
        
        # Predict action using OpenVLA's predict_action method
        with torch.no_grad():
            # OpenVLA directly predicts 7-DOF actions
            raw_actions = self.model.predict_action(
                **inputs, 
                unnorm_key=self.unnorm_key,
                do_sample=False
            )
        
        # Convert to numpy and extract action components
        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.cpu().numpy()
        
        # Parse the 7-DOF action [x, y, z, roll, pitch, yaw, gripper]
        action = raw_actions.squeeze()[:7]  # Ensure we get 7 dimensions
        
        # Scale action
        scaled_action = action * self.action_scale
        
        # Process action for SimplerEnv format
        # SimplerEnv expects a flat 7D action: [x, y, z, rx, ry, rz, gripper]
        # where positions are in meters and rotations are in radians
        
        # Scale the action appropriately
        scaled_action[6] = 2.0 * (action[6] > 0.5) - 1.0  # Convert gripper to -1/1 format
        
        # For ManiSkill3, we need to return the full 7D action
        # Create action dictionary for compatibility
        action_dict = {
            "world_vector": scaled_action.astype(np.float32),
            "terminate_episode": np.array([0], dtype=np.float32)
        }
        
        return action, action_dict
    
    
    def get_action(self, obs: Any) -> np.ndarray:
        """
        Alternative interface for getting action (SimplePolicy style)
        
        Args:
            obs: Observation (image or full observation dict)
            
        Returns:
            7D action array
        """
        # Extract image if obs is a dict
        if isinstance(obs, dict):
            if "rgb" in obs:
                image = obs["rgb"]
            elif "image" in obs:
                image = obs["image"]
            else:
                raise ValueError("No image found in observation")
        else:
            image = obs
        
        raw_action, action_dict = self.step(image)
        return action_dict["world_vector"]


class OpenVLAInference:
    """
    Wrapper class for compatibility with SimplerEnv's inference scripts
    Mimics the interface of RT1Inference and OctoInference
    """
    
    def __init__(
        self,
        model_id: str = "openvla/openvla-7b",
        policy_setup: str = "widowx_bridge",
        device: str = "cuda"
    ):
        """
        Initialize OpenVLA for SimplerEnv inference
        
        Args:
            model_id: HuggingFace model ID or local path
            policy_setup: Robot setup (currently only supports widowx_bridge)
            device: Device to run model on
        """
        self.policy_setup = policy_setup
        self.policy = OpenVLAPolicy(model_path=model_id, device=device)
    
    def reset(self, instruction: str):
        """Reset with new instruction"""
        self.policy.reset(instruction)
    
    def step(self, image: np.ndarray, instruction: str) -> tuple:
        """Step policy"""
        return self.policy.step(image, instruction)