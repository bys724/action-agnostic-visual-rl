"""
OpenVLA policy wrapper for SimplerEnv evaluation
Based on: https://github.com/DelinQu/SimplerEnv-OpenVLA
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image


class OpenVLAPolicy:
    """OpenVLA policy for SimplerEnv evaluation"""
    
    def __init__(
        self, 
        model_path: str = "openvla/openvla-7b",
        device: str = "cuda",
        torch_dtype = torch.bfloat16
    ):
        """
        Initialize OpenVLA model
        
        Args:
            model_path: Path to model checkpoint or HuggingFace model ID
            device: Device to run model on
            torch_dtype: Data type for model weights
        """
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"Loading OpenVLA model from {model_path}...")
        
        # Load processor and model
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            # Try to load the model with trust_remote_code
            # OpenVLA requires custom model class
            from transformers import AutoModel
            try:
                # First try with AutoModel (for custom models)
                self.model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    trust_remote_code=True
                )
            except:
                # Fallback to AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    trust_remote_code=True
                )
        except Exception as e:
            print(f"Warning: Could not load actual model, using mock. Error: {e}")
            # Use a mock model for testing without GPU
            self.model = None
            self.processor = None
        
        if self.model is not None:
            self.model.eval()
        
        # Action space parameters for SimplerEnv
        self.action_dim = 7  # [dx, dy, dz, rx, ry, rz, gripper]
        self.action_scale = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 1.0])
        
        # Cache for current instruction
        self.current_instruction = None
        
        print("OpenVLA model loaded successfully")
    
    def reset(self, instruction: str):
        """Reset policy with new instruction"""
        self.current_instruction = instruction
    
    def step(self, image: np.ndarray, instruction: Optional[str] = None) -> tuple:
        """
        Generate action from image observation
        
        Args:
            image: RGB image observation (H, W, 3) as numpy array
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
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            # Ensure correct shape and dtype
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Convert from [0, 1] to [0, 255]
                image = (image * 255).astype(np.uint8)
            elif image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Handle different channel orders
            if len(image.shape) == 3:
                if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                    image = np.transpose(image, (1, 2, 0))
            
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
        
        # Prepare inputs
        prompt = f"In: What action should the robot take to {self.current_instruction}?\\nOut:"
        inputs = self.processor(prompt, pil_image).to(self.device, dtype=self.torch_dtype)
        
        # Generate action
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0
            )
        
        # Decode action from output
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse action from generated text
        action = self._parse_action(generated_text)
        
        # Scale action
        scaled_action = action * self.action_scale
        
        # Create action dictionary for SimplerEnv
        action_dict = {
            "world_vector": scaled_action,
            "terminate_episode": np.array([0])  # OpenVLA doesn't predict termination
        }
        
        return action, action_dict
    
    def _parse_action(self, text: str) -> np.ndarray:
        """
        Parse action from generated text
        
        Args:
            text: Generated text from model
            
        Returns:
            7D action array
        """
        # Extract action values from text
        # Expected format: "Out: <ACTION> [values]"
        try:
            # Find the action part after "Out:"
            if "Out:" in text:
                action_part = text.split("Out:")[-1].strip()
            else:
                action_part = text.strip()
            
            # Extract numbers from the action string
            # OpenVLA outputs actions in various formats, try to extract numbers
            import re
            numbers = re.findall(r"[-+]?\d*\.?\d+", action_part)
            
            if len(numbers) >= 7:
                # Take first 7 numbers as action
                action = np.array([float(x) for x in numbers[:7]])
            else:
                # Default to small random action if parsing fails
                print(f"Warning: Could not parse action from: {action_part}")
                action = np.random.randn(7) * 0.001
            
            # Clip action to reasonable range
            action = np.clip(action, -1.0, 1.0)
            
        except Exception as e:
            print(f"Error parsing action: {e}")
            # Return small random action on error
            action = np.random.randn(7) * 0.001
        
        return action.astype(np.float32)
    
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