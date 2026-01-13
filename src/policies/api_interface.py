"""
Policy API Interface - 모델 서버 통신을 위한 공통 인터페이스

모든 모델 서버(OpenVLA, LAPA, Custom)는 이 인터페이스를 따릅니다.
평가 환경은 이 클라이언트를 통해 어떤 모델이든 동일하게 호출할 수 있습니다.
"""

import requests
import numpy as np
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import base64
import json
from PIL import Image
import io


class PolicyInterface(ABC):
    """모든 정책이 구현해야 하는 추상 인터페이스"""

    @abstractmethod
    def reset(self, instruction: str) -> None:
        """새 에피소드 시작 시 호출"""
        pass

    @abstractmethod
    def step(self, image: np.ndarray, instruction: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        이미지로부터 액션 생성

        Args:
            image: RGB 이미지 (H, W, 3), uint8
            instruction: 태스크 설명 (선택적)

        Returns:
            raw_action: 원본 모델 출력
            action_dict: {"world_vector": np.ndarray(7,), "terminate_episode": np.ndarray(1,)}
        """
        pass


class PolicyAPIClient(PolicyInterface):
    """
    원격 모델 서버와 통신하는 클라이언트

    모델 서버는 다음 엔드포인트를 제공해야 함:
    - POST /reset: {"instruction": str}
    - POST /step: {"image": base64, "instruction": str} -> {"action": [7 floats], "terminate": bool}
    - GET /health: 서버 상태 확인
    """

    def __init__(
        self,
        server_url: str,
        model_name: str = "unknown",
        timeout: float = 30.0
    ):
        """
        Args:
            server_url: 모델 서버 URL (예: "http://localhost:8001")
            model_name: 모델 이름 (로깅용)
            timeout: 요청 타임아웃 (초)
        """
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.current_instruction = None

    def _encode_image(self, image) -> str:
        """이미지를 base64로 인코딩 (numpy array 또는 torch tensor 지원)"""
        # PyTorch tensor인 경우 numpy로 변환
        if hasattr(image, 'cpu'):  # torch.Tensor check
            image = image.cpu().numpy()

        # numpy array로 변환 확인
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # 배치 차원 제거 (squeeze) - (1, 1, H, W, 3) -> (H, W, 3)
        while image.ndim > 3:
            image = image.squeeze(0)

        # (H, W, 3) 형태 확인
        if image.ndim == 3 and image.shape[2] == 3:
            pass  # 정상
        elif image.ndim == 3 and image.shape[0] == 3:
            # (3, H, W) -> (H, W, 3)
            image = np.transpose(image, (1, 2, 0))

        # uint8로 변환
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def health_check(self) -> bool:
        """서버 상태 확인"""
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=5.0
            )
            return response.status_code == 200
        except:
            return False

    def reset(self, instruction: str) -> None:
        """새 에피소드 시작"""
        self.current_instruction = instruction
        try:
            response = requests.post(
                f"{self.server_url}/reset",
                json={"instruction": instruction},
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Warning: Reset request failed for {self.model_name}: {e}")

    def step(self, image: np.ndarray, instruction: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """액션 생성"""
        if instruction is not None:
            self.current_instruction = instruction

        if self.current_instruction is None:
            raise ValueError("No instruction provided")

        # 이미지 인코딩 및 요청
        payload = {
            "image": self._encode_image(image),
            "instruction": self.current_instruction
        }

        try:
            response = requests.post(
                f"{self.server_url}/step",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            # 응답 파싱
            action = np.array(result["action"], dtype=np.float32)
            terminate = result.get("terminate", False)

            action_dict = {
                "world_vector": action,
                "terminate_episode": np.array([1.0 if terminate else 0.0], dtype=np.float32)
            }

            return action, action_dict

        except requests.exceptions.RequestException as e:
            print(f"Error calling {self.model_name} server: {e}")
            # 실패 시 제로 액션 반환
            action = np.zeros(7, dtype=np.float32)
            return action, {
                "world_vector": action,
                "terminate_episode": np.array([0.0], dtype=np.float32)
            }


class PolicyServerBase:
    """
    모델 서버 구현을 위한 베이스 클래스

    각 모델(OpenVLA, LAPA)은 이 클래스를 상속받아 구현
    """

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.current_instruction = None

    def reset(self, instruction: str) -> Dict[str, Any]:
        """에피소드 리셋 - 서브클래스에서 오버라이드 가능"""
        self.current_instruction = instruction
        return {"status": "ok"}

    @abstractmethod
    def predict(self, image: np.ndarray, instruction: str) -> Dict[str, Any]:
        """
        액션 예측 - 서브클래스에서 반드시 구현

        Returns:
            {"action": [7 floats], "terminate": bool}
        """
        pass

    def create_app(self):
        """FastAPI 앱 생성"""
        from fastapi import FastAPI
        from pydantic import BaseModel
        import base64

        app = FastAPI(title=f"{self.model_name} Policy Server")

        class ResetRequest(BaseModel):
            instruction: str

        class StepRequest(BaseModel):
            image: str  # base64 encoded
            instruction: str

        @app.get("/health")
        def health():
            return {"status": "healthy", "model": self.model_name}

        @app.post("/reset")
        def reset(request: ResetRequest):
            return self.reset(request.instruction)

        @app.post("/step")
        def step(request: StepRequest):
            # base64 디코딩
            image_data = base64.b64decode(request.image)
            image = np.array(Image.open(io.BytesIO(image_data)))

            return self.predict(image, request.instruction)

        return app

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """서버 실행"""
        import uvicorn
        app = self.create_app()
        uvicorn.run(app, host=host, port=port)
