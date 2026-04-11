#!/usr/bin/env python
"""
LIBERO Benchmark Evaluation Script

OpenVLA, Pi0, Custom encoder 모델을 LIBERO 벤치마크에서 평가합니다.
- OpenVLA: REST API 사용
- Pi0: WebSocket API 사용 (openpi 프로토콜)
- Custom: 로컬 encoder + action head (two-stream, single-stream, videomae)

=============================================================================
TODO: Phase 3B — OpenVLA vision backbone 교체 실험 (축소안)
=============================================================================
목적: "동일한 downstream 파이프라인에서 vision encoder 만 교체했을 때,
      어떤 encoder 가 더 나은가" 를 Llama 7B decoder 조건에서 검증.

핵심 논리: 절대 성능이 원본 OpenVLA 40% 를 재현하지 못해도, 모든 encoder 가
          같은 불리함을 겪으므로 **상대 순위** 는 유효하다. Phase 3 의 MLP
          decoder 결과와 결합해서 "encoder 우위가 decoder 용량과 무관함" 을
          주장한다.

실험 범위 (축소):
- Encoder 3개만: Two-Stream v4, V-JEPA-ours (축 1 유지), SigLIP (home advantage reference)
- Task suite 1개: libero_spatial (원본 OpenVLA 결과 존재)
- 모든 encoder frozen, 동일 LoRA rank/epoch/data/seed
- Trial 수: task 당 50 trial × 3 seed

학습 대상 (Phase 3B):
- Projection layer: encoder_feature_dim → Llama embed dim (신규 학습)
- Llama-2 7B LoRA adapter: rank 32, OpenVLA 논문 setup 준수
- Vision encoder: frozen (원본 OpenVLA 는 joint fine-tune 이지만, 우리는
  encoder 품질 비교가 목적이므로 의도적으로 frozen)

기존 인프라 (OpenVLA 공식 저장소 활용):
- https://github.com/openvla/openvla — fine-tune 스크립트 `vla-scripts/finetune.py`
- LIBERO RLDS 데이터: `~/.cache/openvla/datasets/modified_libero_rlds/` (9.6 GB)
- Docker 환경: `docker/openvla-libero/`
- Rollout 서버: 이 파일 (src/eval_libero.py) 의 OpenVLAClient

구현해야 할 것:
(1) OpenVLA vision_backbone 교체 경로
    - `prismatic/models/backbones/vision/` 구조 파악
    - 우리 encoder 를 PrismaticVisionBackbone 인터페이스에 맞춰 감싸는 어댑터
    - 출력 token 포맷 (shape, dtype, feature dim) 을 SigLIP 기준에 맞춰야 함
    - 2-frame 입력 → OpenVLA 가 기대하는 single feature tensor 로 reduction

(2) Projection layer
    - encoder_feature_dim → Llama embed dim (4096)
    - 단순 linear 로 시작. 필요 시 2-layer MLP 로 확장.

(3) fine-tune 파이프라인
    - `vla-scripts/finetune.py` 에 encoder 교체 경로 삽입
    - Vision encoder requires_grad=False 명시
    - LoRA 설정은 OpenVLA 논문 기본값 (rank 32, target llama q/k/v/o)

(4) Rollout 통합
    - 학습된 VLA 를 이 파일의 `OpenVLAClient` 경로로 rollout
    - 기존 OpenVLA REST API 가 vision encoder 교체 후에도 작동하는지 확인

공정성 명시 (논문에 반드시 기재):
    "The original OpenVLA jointly fine-tunes its SigLIP vision encoder.
    For controlled comparison of encoder quality, we freeze all vision
    encoders and train only the projection layer and Llama-2 LoRA adapter
    with the same hyperparameters. Absolute numbers are therefore not
    directly comparable to the original OpenVLA paper."

진입 조건:
- Two-Stream v4 Phase 1.5 학습 완료 + Phase 2 probing 완료
- Phase 3 (MLP decoder) 결과 확인 후 진입 여부 결정
- Phase 3 에서 Two-Stream 명확히 우위 → Phase 3B 는 "강화 증거"
- Phase 3 에서 박빙 → Phase 3B 결과가 판정 역할 (결과 뒤집힐 수 있음)

예상 비용 (Phase 2-3 완료 후 기준):
- OpenVLA vision_backbone 교체 + projection: ~1주
- 3 encoder × LoRA fine-tuning (Llama 7B): ~1주
- Rollout 평가 (3 encoder × libero_spatial × 3 seed): ~2-3일
- 총 ~2.5주

공식 저장소 검증 절차 (구현 전 필수):
1. OpenVLA 공식 finetune.py 를 SigLIP 그대로 LIBERO 에 재현 → 원본 40% 재현 여부 확인
2. SigLIP frozen 으로 돌려 "frozen 정책" 단독의 성능 손실 측정
3. 그 이후 Two-Stream 으로 교체 → 상대 비교

대안 (일정 안 맞으면): rebuttal 시점에 Two-Stream 1개만 붙여서 제시.
=============================================================================
"""

import argparse
import collections
import json
import logging
import math
import pathlib
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import imageio
import numpy as np
import requests
from PIL import Image
import io
import base64

# LIBERO imports (available in libero container)
try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    logging.warning("LIBERO not available. Install LIBERO or run in LIBERO container.")

# Custom encoder imports (optional)
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Task suite configurations
TASK_SUITE_CONFIG = {
    "libero_spatial": {"max_steps": 220},
    "libero_object": {"max_steps": 280},
    "libero_goal": {"max_steps": 300},
    "libero_10": {"max_steps": 520},
    "libero_90": {"max_steps": 400},
}

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


class OpenVLAClient:
    """OpenVLA REST API Client for LIBERO."""

    def __init__(self, host: str, port: int, resize_size: int = 224):
        self.base_url = f"http://{host}:{port}"
        self.resize_size = resize_size
        self._check_health()

    def _check_health(self):
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                logging.info(f"OpenVLA server healthy: {resp.json()}")
            else:
                logging.warning(f"OpenVLA server not healthy: {resp.status_code}")
        except Exception as e:
            logging.warning(f"Cannot connect to OpenVLA server: {e}")

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 PNG."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _resize_with_pad(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image with padding to preserve aspect ratio."""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)
        resized = np.array(pil_image)

        # Pad to target size
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        padded = np.zeros((target_size, target_size, 3), dtype=resized.dtype)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        return padded

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on observation."""
        # Preprocess image (rotate 180 degrees as per LIBERO convention)
        image = np.ascontiguousarray(obs["observation/image"][::-1, ::-1])
        image = self._resize_with_pad(image, self.resize_size)

        payload = {
            "image": self._encode_image(image),
            "instruction": obs["prompt"]
        }

        resp = requests.post(f"{self.base_url}/step", json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenVLA inference failed: {resp.text}")

        result = resp.json()
        action = np.array(result["action"], dtype=np.float32)

        return {"actions": action[np.newaxis, :]}  # Add action horizon dim


class CustomEncoderClient:
    """
    Custom encoder client for LIBERO evaluation.

    Loads a fine-tuned encoder + action head locally and runs inference
    without requiring a server.
    """

    def __init__(
        self,
        checkpoint_path: str,
        encoder_type: str = "two-stream",
        device: str = "cuda",
        img_size: int = 224,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install: pip install torch")

        self.device = device
        self.img_size = img_size
        self.encoder_type = encoder_type

        logging.info(f"Loading custom encoder from: {checkpoint_path}")
        logging.info(f"Encoder type: {encoder_type}, Device: {device}")

        # Load model
        sys.path.insert(0, "/workspace")
        from src.models.action_head import EncoderWithActionHead

        self.model = EncoderWithActionHead.from_checkpoint(
            checkpoint_path=checkpoint_path,
            encoder_type=encoder_type,
            device=device,
        )
        self.model.eval()
        logging.info("Custom encoder loaded successfully")

        # Track previous observation for frame pair
        self._prev_image = None

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image: resize, normalize, convert to tensor."""
        # img: [H, W, C] uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        img = img.astype(np.float32) / 255.0  # [0, 1]

        # Convert to tensor and resize
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return img_tensor

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on observation."""
        # Get current image (rotate 180 degrees as per LIBERO convention)
        curr_image = np.ascontiguousarray(obs["observation/image"][::-1, ::-1])

        # Use previous image or current if first frame
        if self._prev_image is None:
            prev_image = curr_image.copy()
        else:
            prev_image = self._prev_image

        # Store for next iteration
        self._prev_image = curr_image.copy()

        # Preprocess images
        prev_tensor = self._preprocess_image(prev_image)
        curr_tensor = self._preprocess_image(curr_image)

        # Stack as 6-channel input [prev, curr]
        pixel_values = torch.cat([prev_tensor, curr_tensor], dim=0)  # [6, H, W]
        pixel_values = pixel_values.unsqueeze(0).to(self.device)  # [1, 6, H, W]

        # Inference
        with torch.no_grad():
            action = self.model(pixel_values)  # [1, 7]

        action_np = action.cpu().numpy().squeeze(0)  # [7]

        return {"actions": action_np[np.newaxis, :]}  # Add action horizon dim

    def reset(self):
        """Reset internal state for new episode."""
        self._prev_image = None


class Pi0Client:
    """Pi0 WebSocket Client for LIBERO."""

    def __init__(self, host: str, port: int, resize_size: int = 224):
        self.host = host
        self.port = port
        self.resize_size = resize_size
        self._ws = None
        self._packer = None
        self._connect()

    def _connect(self):
        """Connect to Pi0 WebSocket server."""
        try:
            import functools
            import websockets.sync.client
            import msgpack

            # Custom msgpack numpy serialization (matching openpi format)
            def pack_array(obj):
                if isinstance(obj, np.ndarray):
                    return {
                        b"__ndarray__": True,
                        b"data": obj.tobytes(),
                        b"dtype": obj.dtype.str,
                        b"shape": obj.shape,
                    }
                if isinstance(obj, np.generic):
                    return {
                        b"__npgeneric__": True,
                        b"data": obj.item(),
                        b"dtype": obj.dtype.str,
                    }
                return obj

            def unpack_array(obj):
                if b"__ndarray__" in obj:
                    return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
                if b"__npgeneric__" in obj:
                    return np.dtype(obj[b"dtype"]).type(obj[b"data"])
                return obj

            uri = f"ws://{self.host}:{self.port}"
            logging.info(f"Connecting to Pi0 server at {uri}...")

            self._ws = websockets.sync.client.connect(
                uri, compression=None, max_size=None
            )
            # Receive server metadata
            metadata = msgpack.unpackb(self._ws.recv(), object_hook=unpack_array)
            logging.info(f"Pi0 server metadata: {metadata}")

            # Create packer and unpack function with numpy support
            self._packer = msgpack.Packer(default=pack_array)
            self._unpack_array = unpack_array
        except ImportError:
            logging.error("websockets or msgpack not installed. Install: pip install websockets msgpack")
            raise
        except Exception as e:
            logging.error(f"Cannot connect to Pi0 server: {e}")
            raise

    def _resize_with_pad(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image with padding."""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)
        resized = np.array(pil_image)

        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        padded = np.zeros((target_size, target_size, 3), dtype=resized.dtype)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        return padded

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on observation."""
        import msgpack

        # Preprocess images (rotate 180 degrees as per LIBERO convention)
        image = np.ascontiguousarray(obs["observation/image"][::-1, ::-1])
        wrist_image = np.ascontiguousarray(obs["observation/wrist_image"][::-1, ::-1])

        image = self._resize_with_pad(image, self.resize_size)
        wrist_image = self._resize_with_pad(wrist_image, self.resize_size)

        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if wrist_image.dtype != np.uint8:
            wrist_image = (wrist_image * 255).astype(np.uint8)

        # Prepare Pi0 observation format
        element = {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": obs["observation/state"].astype(np.float32),
            "prompt": obs["prompt"],
        }

        # Send request using custom msgpack numpy serialization
        data = self._packer.pack(element)
        self._ws.send(data)

        # Receive response
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Pi0 inference error: {response}")

        result = msgpack.unpackb(response, object_hook=self._unpack_array)
        return result


def quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_libero_env(task, resolution, seed):
    """Initialize LIBERO environment."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def evaluate_libero(
    client,
    task_suite_name: str = "libero_spatial",
    num_trials_per_task: int = 50,
    num_steps_wait: int = 10,
    replan_steps: int = 5,
    video_out_path: str = "data/libero/videos",
    seed: int = 7,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run LIBERO evaluation."""
    if not LIBERO_AVAILABLE:
        raise RuntimeError("LIBERO not available. Run in LIBERO container.")

    np.random.seed(seed)

    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = TASK_SUITE_CONFIG[task_suite_name]["max_steps"]

    logging.info(f"Task suite: {task_suite_name} ({num_tasks} tasks)")
    logging.info(f"Max steps: {max_steps}, Trials per task: {num_trials_per_task}")

    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    # Evaluation loop
    total_episodes, total_successes = 0, 0
    task_results = []

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)

        task_episodes, task_successes = 0, 0

        for episode_idx in range(num_trials_per_task):
            if verbose:
                logging.info(f"Task {task_id+1}/{num_tasks}, Episode {episode_idx+1}/{num_trials_per_task}: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            # Reset client state for new episode (for CustomEncoderClient)
            if hasattr(client, "reset"):
                client.reset()

            t = 0
            replay_images = []
            done = False

            while t < max_steps + num_steps_wait:
                try:
                    # Wait for objects to stabilize
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Preprocess observations
                    img = obs["agentview_image"]
                    wrist_img = obs["robot0_eye_in_hand_image"]
                    replay_images.append(img.copy())

                    if not action_plan:
                        # Prepare observation dict
                        state = np.concatenate([
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ])

                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state,
                            "prompt": str(task_description),
                        }

                        # Get action from policy
                        result = client.infer(element)
                        action_chunk = result["actions"]

                        # Handle different action formats
                        if action_chunk.ndim == 1:
                            action_chunk = action_chunk[np.newaxis, :]

                        assert len(action_chunk) >= replan_steps, \
                            f"Need {replan_steps} actions, got {len(action_chunk)}"
                        action_plan.extend(action_chunk[:replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Error during episode: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save video
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")[:50]
            video_path = pathlib.Path(video_out_path) / f"task{task_id}_{episode_idx}_{suffix}.mp4"
            try:
                imageio.mimwrite(str(video_path), replay_images, fps=10)
            except Exception as e:
                logging.warning(f"Could not save video: {e}")

            if verbose:
                logging.info(f"  Result: {'SUCCESS' if done else 'FAILURE'}")

        env.close()

        task_success_rate = float(task_successes) / float(task_episodes)
        task_results.append({
            "task_id": task_id,
            "task_description": task_description,
            "success_rate": task_success_rate,
            "successes": task_successes,
            "episodes": task_episodes,
        })

        logging.info(f"Task {task_id+1} ({task_description[:30]}...): {task_success_rate:.1%}")

    # Summary
    overall_success_rate = float(total_successes) / float(total_episodes)

    return {
        "task_suite": task_suite_name,
        "overall_success_rate": overall_success_rate,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "task_results": task_results,
    }


def main():
    parser = argparse.ArgumentParser(description="LIBERO Benchmark Evaluation")

    # Model selection
    parser.add_argument("--model", type=str, default="openvla",
                        choices=["openvla", "pi0", "custom"],
                        help="Model to evaluate")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Policy server host (for openvla/pi0)")
    parser.add_argument("--port", type=int, default=None,
                        help="Policy server port (default: 8001 for OpenVLA, 8000 for Pi0)")

    # Custom encoder options
    parser.add_argument("--encoder", type=str, default="two-stream",
                        choices=["two-stream", "single-stream", "videomae"],
                        help="Encoder type (for --model custom)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned checkpoint (for --model custom)")

    # Task configuration
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        choices=list(TASK_SUITE_CONFIG.keys()),
                        help="LIBERO task suite")
    parser.add_argument("--num-trials", type=int, default=50,
                        help="Number of trials per task")
    parser.add_argument("--replan-steps", type=int, default=None,
                        help="Replanning interval (default: 1 for OpenVLA, 5 for Pi0)")

    # Output
    parser.add_argument("--output-dir", type=str, default="data/libero/results",
                        help="Output directory for results")
    parser.add_argument("--video-dir", type=str, default="data/libero/videos",
                        help="Output directory for videos")

    # Other
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-episode logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Set model-specific defaults
    if args.port is None:
        args.port = 8001 if args.model == "openvla" else 8000
    if args.replan_steps is None:
        if args.model == "openvla":
            args.replan_steps = 1
        elif args.model == "pi0":
            args.replan_steps = 5
        else:  # custom
            args.replan_steps = 1

    # Create client
    if args.model == "openvla":
        client = OpenVLAClient(args.host, args.port)
    elif args.model == "pi0":
        client = Pi0Client(args.host, args.port)
    elif args.model == "custom":
        if args.checkpoint is None:
            print("Error: --checkpoint is required for --model custom")
            sys.exit(1)
        client = CustomEncoderClient(
            checkpoint_path=args.checkpoint,
            encoder_type=args.encoder,
            device="cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print("=" * 70)
    print(f"LIBERO Benchmark Evaluation")
    print(f"Model: {args.model}")
    if args.model == "custom":
        print(f"Encoder: {args.encoder}")
        print(f"Checkpoint: {args.checkpoint}")
    else:
        print(f"Server: {args.host}:{args.port}")
    print(f"Task Suite: {args.task_suite}")
    print(f"Trials per task: {args.num_trials}")
    print(f"Replan steps: {args.replan_steps}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # Run evaluation
    results = evaluate_libero(
        client=client,
        task_suite_name=args.task_suite,
        num_trials_per_task=args.num_trials,
        replan_steps=args.replan_steps,
        video_out_path=args.video_dir,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Add evaluation settings metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results["metadata"] = {
        "model": args.model,
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "num_trials_per_task": args.num_trials,
        "num_steps_wait": 10,
        "env_resolution": LIBERO_ENV_RESOLUTION,
        "max_steps": TASK_SUITE_CONFIG[args.task_suite]["max_steps"],
        "timestamp": timestamp,
    }
    if args.model == "custom":
        results["metadata"]["encoder"] = args.encoder
        results["metadata"]["checkpoint"] = args.checkpoint
    else:
        results["metadata"]["server"] = f"{args.host}:{args.port}"

    # Save results
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.model == "custom":
        result_file = output_path / f"custom_{args.encoder}_{args.task_suite}_{timestamp}.json"
    else:
        result_file = output_path / f"{args.model}_{args.task_suite}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Task Suite: {results['task_suite']}")
    print(f"Overall Success Rate: {results['overall_success_rate']:.1%}")
    print(f"Total: {results['total_successes']}/{results['total_episodes']} episodes")
    print(f"\nResults saved to: {result_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
