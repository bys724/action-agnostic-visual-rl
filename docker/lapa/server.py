"""
LAPA Policy Server

REST API로 LAPA 모델 추론을 제공합니다.
JAX/Flax 기반의 LAPA 모델을 위한 독립 환경입니다.
"""

import os
import sys
import io
import base64
import csv
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transforms3d.euler import euler2axangle

# Add LAPA to path
sys.path.insert(0, "/app/LAPA")


class ResetRequest(BaseModel):
    instruction: str


class StepRequest(BaseModel):
    image: str  # base64 encoded PNG
    instruction: str


class LAPAServer:
    def __init__(
        self,
        checkpoint_path: str,
        action_scale_file: str,
        vqgan_checkpoint: str = "/app/LAPA/lapa_checkpoints/vqgan",
        vocab_file: str = "/app/LAPA/lapa_checkpoints/tokenizer.model",
        policy_setup: str = "widowx_bridge",
        tokens_per_delta: int = 4,
        tokens_per_action: int = 7,
    ):
        self.policy_setup = policy_setup
        self.current_instruction = None
        self.action_scale = 1.0

        print(f"Loading LAPA model from {checkpoint_path}...")

        try:
            # Import LAPA dependencies
            from tux import JaxDistributedConfig, set_random_seed
            from latent_pretraining.delta_llama import VideoLLaMAConfig

            # Initialize JAX
            jax_config = JaxDistributedConfig.get_default_config()
            JaxDistributedConfig.initialize(jax_config)
            set_random_seed(1234)

            # Setup tokenizer and LLaMA config
            tokenizer = VideoLLaMAConfig.get_tokenizer_config()
            llama = VideoLLaMAConfig.get_default_config()
            tokenizer.vocab_file = vocab_file

            # Import sampler
            if tokens_per_delta > 0:
                from latent_pretraining.sampler_latent_action_pretrain import DeltaActionSampler as Sampler
            else:
                from latent_pretraining.sampler_action_pretrain import ActionSampler as Sampler

            # Create FLAGS-like config
            class FLAGSClass:
                def __init__(self, flag_dict):
                    for key, value in flag_dict.items():
                        setattr(self, key, value)

            update_llama_config = (
                "dict(action_vocab_size=256,delta_vocab_size=8,sample_mode='text',"
                "theta=50000000,max_sequence_length=32768,scan_attention=False,"
                "scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,"
                "scan_mlp_chunk_size=8192,scan_layers=True)"
            )

            kwargs = {
                "vqgan_checkpoint": vqgan_checkpoint,
                "seed": 1234,
                "mesh_dim": "1,-1,1,1",
                "dtype": "bf16",
                "load_llama_config": "7b",
                "update_llama_config": update_llama_config,
                "tokens_per_delta": tokens_per_delta,
                "tokens_per_action": tokens_per_action,
                "vocab_file": vocab_file,
                "multi_image": 1,
                "jax_distributed": jax_config,
                "action_scale_file": action_scale_file,
                "tokenizer": tokenizer,
                "llama": llama,
                "load_checkpoint": f"params::{checkpoint_path}",
                "image_aug": False,
            }

            flags = FLAGSClass(kwargs)
            self.model = Sampler(FLAGS=flags)

            # Load action scale file
            self.action_scale_list = []
            with open(action_scale_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    self.action_scale_list.append([float(v) for v in row if v.strip()])

            self.tokens_per_delta = tokens_per_delta
            self.model_loaded = True
            print("LAPA model loaded successfully")

        except Exception as e:
            print(f"Warning: Could not load LAPA model: {e}")
            print("Running in mock mode")
            self.model = None
            self.model_loaded = False
            self.action_scale_list = []

    def get_averaged_values(self, indices):
        """잠재 액션 인덱스를 실제 액션 값으로 변환"""
        averaged_values = []
        for row_idx, idx in enumerate(indices):
            try:
                value1 = self.action_scale_list[row_idx][idx]
                value2 = self.action_scale_list[row_idx][idx + 1]
                average = (value1 + value2) / 2
            except:
                average = 0.0
            averaged_values.append(average)
        return averaged_values

    def reset(self, instruction: str):
        self.current_instruction = instruction
        return {"status": "ok", "instruction": instruction}

    def predict(self, image: np.ndarray, instruction: str) -> dict:
        """이미지에서 액션 예측"""
        self.current_instruction = instruction

        if not self.model_loaded:
            # Mock 모드
            return {
                "action": [0.0] * 7,
                "terminate": False,
                "mock": True
            }

        # 이미지 전처리
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        pil_image = Image.fromarray(image)
        prompts = [{'image': [pil_image], 'question': instruction}]

        # LAPA 추론
        action_output = self.model(prompts)[0]
        raw_actions = self.get_averaged_values(action_output)

        # 액션 변환
        world_vector = np.array(raw_actions[:3]) * self.action_scale
        rotation_delta = np.array(raw_actions[3:6])
        gripper = raw_actions[6]

        # Rotation을 axis-angle로 변환
        roll, pitch, yaw = rotation_delta
        ax, angle = euler2axangle(roll, pitch, yaw)
        rot_axangle = ax * angle * self.action_scale

        # Gripper 이진화
        if self.policy_setup == "widowx_bridge":
            gripper_action = 1.0 if gripper > 0.5 else -1.0
        else:
            gripper_action = gripper

        # 7D 액션 조합
        action = np.concatenate([world_vector, rot_axangle, [gripper_action]])

        return {
            "action": action.tolist(),
            "terminate": False
        }


# FastAPI 앱 생성
app = FastAPI(title="LAPA Policy Server")
server: Optional[LAPAServer] = None


@app.on_event("startup")
async def startup():
    global server
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/lapa/params")
    action_scale_file = os.environ.get("ACTION_SCALE_FILE", "/app/checkpoints/lapa/action_scale.csv")
    vqgan_checkpoint = os.environ.get("VQGAN_CHECKPOINT", "/app/checkpoints/lapa/vqgan")
    vocab_file = os.environ.get("VOCAB_FILE", "/app/checkpoints/lapa/tokenizer.model")
    policy_setup = os.environ.get("POLICY_SETUP", "widowx_bridge")

    server = LAPAServer(
        checkpoint_path=checkpoint_path,
        action_scale_file=action_scale_file,
        vqgan_checkpoint=vqgan_checkpoint,
        vocab_file=vocab_file,
        policy_setup=policy_setup
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "lapa",
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
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
