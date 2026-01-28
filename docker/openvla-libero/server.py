"""
OpenVLA-LIBERO Policy Server

LIBERO 벤치마크용 OpenVLA 모델 추론을 제공합니다.
"""

import os
import io
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
        device: str = "cuda",
    ):
        self.device = device
        self.current_instruction = None

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
            print("OpenVLA-LIBERO model loaded successfully")
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Running in mock mode")
            self.model = None
            self.processor = None
            self.model_loaded = False

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
                do_sample=False
            )

        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.cpu().numpy()

        action = raw_actions.squeeze()[:7].astype(float)

        # Gripper binarization
        action[6] = 1.0 if action[6] > 0.5 else -1.0

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
    server = OpenVLALIBEROServer(model_path=model_path)


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
