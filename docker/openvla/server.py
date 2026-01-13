"""
OpenVLA Policy Server

REST API로 OpenVLA 모델 추론을 제공합니다.
평가 환경에서 HTTP 요청으로 액션을 받아갈 수 있습니다.
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
from transforms3d.euler import euler2axangle

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ResetRequest(BaseModel):
    instruction: str


class StepRequest(BaseModel):
    image: str  # base64 encoded PNG
    instruction: str


class OpenVLAServer:
    def __init__(
        self,
        model_path: str = "openvla/openvla-7b",
        device: str = "cuda",
        policy_setup: str = "widowx_bridge"
    ):
        self.device = device
        self.policy_setup = policy_setup
        self.current_instruction = None

        # Unnorm key for action denormalization
        if policy_setup == "widowx_bridge":
            self.unnorm_key = "bridge_orig"
        elif policy_setup == "google_robot":
            self.unnorm_key = "fractal20220817_data"
        else:
            self.unnorm_key = "bridge_orig"

        print(f"Loading OpenVLA model from {model_path}...")

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
            print("OpenVLA model loaded successfully")
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Running in mock mode")
            self.model = None
            self.processor = None
            self.model_loaded = False

    def reset(self, instruction: str):
        self.current_instruction = instruction
        return {"status": "ok", "instruction": instruction}

    def predict(self, image: np.ndarray, instruction: str) -> dict:
        """이미지에서 액션 예측"""
        self.current_instruction = instruction

        # PIL 이미지로 변환
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        pil_image = Image.fromarray(image)

        if not self.model_loaded:
            # Mock 모드
            return {
                "action": [0.0] * 7,
                "terminate": False,
                "mock": True
            }

        # OpenVLA 추론
        with torch.no_grad():
            inputs = self.processor(instruction, pil_image).to(self.device, dtype=torch.bfloat16)
            raw_actions = self.model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False
            )

        # numpy로 변환
        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.cpu().numpy()

        action = raw_actions.squeeze()[:7].astype(float)

        # Gripper 이진화
        action[6] = 1.0 if action[6] > 0.5 else -1.0

        return {
            "action": action.tolist(),
            "terminate": False
        }


# FastAPI 앱 생성
app = FastAPI(title="OpenVLA Policy Server")
server: Optional[OpenVLAServer] = None


@app.on_event("startup")
async def startup():
    global server
    model_path = os.environ.get("MODEL_PATH", "openvla/openvla-7b")
    policy_setup = os.environ.get("POLICY_SETUP", "widowx_bridge")
    server = OpenVLAServer(model_path=model_path, policy_setup=policy_setup)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "openvla",
        "loaded": server.model_loaded if server else False
    }


@app.post("/reset")
def reset(request: ResetRequest):
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return server.reset(request.instruction)


@app.post("/step")
def step(request: StepRequest):
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # base64 디코딩
    try:
        image_data = base64.b64decode(request.image)
        image = np.array(Image.open(io.BytesIO(image_data)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    return server.predict(image, request.instruction)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
