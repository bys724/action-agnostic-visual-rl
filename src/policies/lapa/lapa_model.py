"""
LAPA policy wrapper for SimplerEnv evaluation

LAPA는 JAX/Flax 기반 모델로, 의존성 충돌 방지를 위해 API 모드 사용을 권장합니다.
로컬 모드는 mock policy로 동작합니다.

사용법:
  API 모드 (권장): python src/eval_simpler.py --model lapa --api-url http://localhost:8002
  서버 시작: docker compose up -d lapa

Reference: https://github.com/LatentActionPretraining/LAPA
"""

from typing import Optional, Any
import numpy as np


class LAPAPolicy:
    """
    LAPA policy wrapper for SimplerEnv evaluation

    LAPA는 JAX/Flax 환경이 필요하므로 별도 컨테이너(docker/lapa/)에서 실행됩니다.
    이 클래스는 API 모드가 아닌 경우 mock policy로 동작합니다.
    """

    def __init__(self, model_path: str = "", **kwargs):
        self.model_path = model_path
        self.current_instruction = None
        print("Warning: LAPA local mode not supported. Using mock policy.")
        print("Use API mode instead: --api-url http://localhost:8002")

    def reset(self, instruction: str):
        """Reset policy with new instruction"""
        self.current_instruction = instruction

    def step(self, image: np.ndarray, instruction: Optional[str] = None) -> tuple:
        """Generate mock action (로컬 모드에서는 zero action 반환)"""
        if instruction is not None:
            self.current_instruction = instruction

        raw_action = np.zeros(7, dtype=np.float32)
        action_dict = {
            "world_vector": raw_action,
            "terminate_episode": np.array([0.0])
        }
        return raw_action, action_dict

    def get_action(self, obs: Any) -> np.ndarray:
        """Alternative interface for getting action"""
        if isinstance(obs, dict):
            image = obs.get("rgb", obs.get("image"))
        else:
            image = obs
        _, action_dict = self.step(image)
        return action_dict["world_vector"]
