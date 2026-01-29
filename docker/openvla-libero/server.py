"""
OpenVLA-LIBERO Policy Server

LIBERO 벤치마크용 OpenVLA 모델 추론을 제공합니다.
"""

import os
import io
import json
import base64
from typing import Optional

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForVision2Seq, AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class StepRequest(BaseModel):
    image: str  # base64 encoded PNG
    instruction: str


class OpenVLALIBEROServer:
    def __init__(
        self,
        model_path: str = "openvla/openvla-7b-finetuned-libero-10",
        unnorm_key: str = "libero_spatial",
        device: str = "cuda",
    ):
        self.device = device
        self.current_instruction = None
        self.unnorm_key = unnorm_key

        print(f"Loading OpenVLA-LIBERO model from {model_path}...")

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(device)
            self.model.eval()

            # Load normalization statistics
            self.norm_stats = None
            stats_path = os.path.join(model_path, "dataset_statistics.json")
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    all_stats = json.load(f)
                # Try exact key, then with _no_noops suffix
                if unnorm_key in all_stats:
                    self.norm_stats = all_stats[unnorm_key]
                elif f"{unnorm_key}_no_noops" in all_stats:
                    self.norm_stats = all_stats[f"{unnorm_key}_no_noops"]
                else:
                    # Use first available key
                    self.norm_stats = list(all_stats.values())[0]
                print(f"Loaded normalization stats for: {unnorm_key}")

            print("OpenVLA-LIBERO model loaded successfully")
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Running in mock mode")
            self.model = None
            self.processor = None
            self.model_loaded = False
            self.norm_stats = None

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Unnormalize action using dataset statistics."""
        if self.norm_stats is None:
            return action

        mean = np.array(self.norm_stats["action"]["mean"])
        std = np.array(self.norm_stats["action"]["std"])

        # Unnormalize: action = normalized * std + mean
        unnorm_action = action * std + mean
        return unnorm_action

    def predict(self, image: np.ndarray, instruction: str) -> dict:
        """Predict action from image and instruction."""
        self.current_instruction = instruction

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        pil_image = Image.fromarray(image)

        if not self.model_loaded:
            return {
                "action": [0.0] * 7,
                "terminate": False,
                "mock": True
            }

        with torch.no_grad():
            inputs = self.processor(instruction, pil_image).to(self.device, dtype=torch.bfloat16)
            raw_actions = self.model.predict_action(
                **inputs,
                do_sample=False,
                unnorm_key=self.unnorm_key,
            )

        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.cpu().numpy()

        action = raw_actions.squeeze()[:7].astype(float)

        # Note: predict_action with unnorm_key should handle unnormalization
        # But we apply it manually if norm_stats are available and unnorm_key wasn't used
        # action = self.unnormalize_action(action)

        # Gripper processing:
        # OpenVLA outputs gripper in [0, 1] where 0=close, 1=open
        # LIBERO expects gripper in [-1, 1] where -1=open, +1=close
        # So we need to: (1) binarize, (2) invert sign
        gripper_value = action[6]
        if gripper_value > 0.5:
            action[6] = -1.0  # open
        else:
            action[6] = 1.0   # close

        return {
            "action": action.tolist(),
            "terminate": False
        }


app = FastAPI(title="OpenVLA-LIBERO Policy Server")
server: Optional[OpenVLALIBEROServer] = None


@app.on_event("startup")
async def startup():
    global server
    model_path = os.environ.get("MODEL_PATH", "openvla/openvla-7b-finetuned-libero-10")
    unnorm_key = os.environ.get("UNNORM_KEY", "libero_spatial")
    server = OpenVLALIBEROServer(model_path=model_path, unnorm_key=unnorm_key)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "openvla-libero",
        "loaded": server.model_loaded if server else False
    }


@app.post("/step")
def step(request: StepRequest):
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        image_data = base64.b64decode(request.image)
        image = np.array(Image.open(io.BytesIO(image_data)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    return server.predict(image, request.instruction)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 18010))
    uvicorn.run(app, host="0.0.0.0", port=port)
