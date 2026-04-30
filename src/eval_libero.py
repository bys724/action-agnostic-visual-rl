#!/usr/bin/env python
"""LIBERO BC-Transformer policy rollout (closed-loop simulator evaluation).

학습 산출물(`scripts/eval/finetune_libero_bct.py`의 `best.pt`)을 LIBERO
시뮬레이터에서 closed-loop 평가. Encoder ckpt path가 cluster path로 박혀
있어도 무관 — `policy_state_dict`이 adapter weights를 모두 포함하므로 None
override 후 `load_state_dict`로 한 번에 덮어씀.

Usage (in libero-eval container):
    docker compose up -d libero
    docker exec libero-eval python src/eval_libero.py \
        --checkpoint /mnt/data/checkpoints/libero_bct/bct_videomae-ours_libero_spatial_seed0_best.pt \
        --task-suite libero_spatial \
        --num-trials 50
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import pathlib
import sys
from datetime import datetime
from typing import Any, Dict

import imageio
import numpy as np

# robomimic 0.x ↔ numpy 1.20+ 호환
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# LIBERO (libero-eval container 안에서만 import 가능)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.policies.bc_transformer_adapted import AdaptedBCTransformerPolicy


# ============================================================================
# Constants
# ============================================================================

TASK_SUITE_CONFIG = {
    "libero_spatial": {"max_steps": 220},
    "libero_object": {"max_steps": 280},
    "libero_goal": {"max_steps": 300},
    "libero_10": {"max_steps": 520},
    "libero_90": {"max_steps": 400},
}

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def libero_shape_meta() -> Dict[str, Any]:
    """LIBERO 표준 shape_meta — task suite 무관, BC-T 학습 시 robomimic이
    HDF5에서 추출하던 값을 hardcode (rollout 시 dataset 불필요).
    """
    return {
        "ac_dim": 7,
        "all_shapes": collections.OrderedDict([
            ("agentview_rgb", [3, 128, 128]),
            ("eye_in_hand_rgb", [3, 128, 128]),
            ("gripper_states", [2]),
        ]),
        "all_obs_keys": ["agentview_rgb", "eye_in_hand_rgb", "gripper_states"],
    }


# ============================================================================
# Utilities
# ============================================================================

def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_libero_env(task, resolution: int, seed: int):
    desc = task.language
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, desc


# ============================================================================
# BC-Transformer rollout client
# ============================================================================

class BCTransformerClient:
    """LIBERO BC-Transformer policy rollout client.

    학습 흐름과 동일하게 `spatial_encode`/`temporal_encode`/`policy_head`를
    직접 호출 (`get_action`의 `preprocess_input` 우회 — 입력 분포 일치).
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        logging.info(f"Loading BC-T ckpt: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(ckpt["config"])

        # Encoder ckpt path가 cluster-only일 수 있음 → random init 후 덮어씀
        cfg.encoder.checkpoint = None

        self.policy = AdaptedBCTransformerPolicy(cfg, libero_shape_meta()).to(device)
        missing, unexpected = self.policy.load_state_dict(
            ckpt["policy_state_dict"], strict=False,
        )
        if missing:
            logging.warning(f"Missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            logging.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
        self.policy.eval()
        self.img_size = self.policy.adapter.img_size

        # CLIP for task_emb (학습 시와 동일 setup: pooler_output 사용)
        from transformers import CLIPTokenizer, CLIPModel
        self.clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_model.eval()
        self._task_emb_cache: Dict[str, torch.Tensor] = {}

        eloss = ckpt.get("eval_loss")
        eloss_str = f"{eloss:.4f}" if eloss is not None else "n/a"
        logging.info(f"Loaded | epoch={ckpt.get('epoch')} eval_loss={eloss_str} img_size={self.img_size}")

    @torch.no_grad()
    def _task_emb(self, prompt: str) -> torch.Tensor:
        if prompt not in self._task_emb_cache:
            toks = self.clip_tok(
                [prompt], padding="max_length", max_length=25,
                truncation=True, return_tensors="pt",
            ).to(self.device)
            feats = self.clip_model.text_model(**toks).pooler_output  # (1, 512)
            self._task_emb_cache[prompt] = feats.squeeze(0)
        return self._task_emb_cache[prompt]

    def _img_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """LIBERO env image (H, W, 3) uint8 → (1, 1, 3, img_size, img_size) [0,1]."""
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        if x.shape[1] != self.img_size or x.shape[2] != self.img_size:
            x = F.interpolate(
                x.unsqueeze(0), size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return x.unsqueeze(0).unsqueeze(0).to(self.device)

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Single-step inference. Returns {"actions": np.ndarray (1, 7)}.

        학습 시 raw HDF5 (env-coord 그대로) 그대로 사용 → rollout도 회전 미적용.
        """
        agent = self._img_to_tensor(obs["observation/image"])
        wrist = self._img_to_tensor(obs["observation/wrist_image"])
        # gripper_states: state = [eef_pos(3), axisangle(3), gripper_qpos(2)]
        state = obs["observation/state"]
        gripper = torch.from_numpy(state[-2:]).float().view(1, 1, 2).to(self.device)
        task_emb = self._task_emb(str(obs["prompt"])).unsqueeze(0)  # (1, 512)

        data = {
            "obs": {
                "agentview_rgb": agent,
                "eye_in_hand_rgb": wrist,
                "gripper_states": gripper,
            },
            "task_emb": task_emb,
        }

        # 학습 forward 흐름 그대로 (preprocess_input 우회)
        x = self.policy.spatial_encode(data)  # (1, 1, num_mod, E)
        self.policy.latent_queue.append(x)
        if len(self.policy.latent_queue) > self.policy.max_seq_len:
            self.policy.latent_queue.pop(0)
        x_seq = torch.cat(self.policy.latent_queue, dim=1)  # (1, T_acc, num_mod, E)
        x_seq = self.policy.temporal_encode(x_seq)  # (1, T_acc, E)
        dist = self.policy.policy_head(x_seq[:, -1])  # GMM at last step

        action = dist.sample().squeeze(0).cpu().numpy()  # (7,)
        return {"actions": action[np.newaxis, :]}  # (1, 7)


# ============================================================================
# Rollout loop
# ============================================================================

def evaluate_libero(
    client: BCTransformerClient,
    task_suite_name: str,
    num_trials_per_task: int,
    task_ids: list = None,
    num_steps_wait: int = 10,
    replan_steps: int = 1,
    video_out_path: str = "data/libero/videos",
    seed: int = 7,
    verbose: bool = True,
) -> Dict[str, Any]:
    np.random.seed(seed)
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    n_tasks = task_suite.n_tasks
    max_steps = TASK_SUITE_CONFIG[task_suite_name]["max_steps"]
    target_ids = list(range(n_tasks)) if task_ids is None else list(task_ids)
    logging.info(f"{task_suite_name}: tasks={target_ids} × {num_trials_per_task} trials, max_steps={max_steps}")

    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    total_eps, total_succ = 0, 0
    task_results = []

    for task_id in target_ids:
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env, desc = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)

        task_eps, task_succ = 0, 0
        for ep in range(num_trials_per_task):
            if verbose:
                logging.info(f"[{task_id+1}/{n_tasks}] ep {ep+1}/{num_trials_per_task}: {desc}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(init_states[ep])
            client.reset()

            t = 0
            replay = []
            done = False
            while t < max_steps + num_steps_wait:
                try:
                    if t < num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = obs["agentview_image"]
                    wrist_img = obs["robot0_eye_in_hand_image"]
                    replay.append(img.copy())

                    if not action_plan:
                        state = np.concatenate([
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ])
                        result = client.infer({
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state,
                            "prompt": str(desc),
                        })
                        chunk = result["actions"]
                        if chunk.ndim == 1:
                            chunk = chunk[np.newaxis, :]
                        assert len(chunk) >= replan_steps
                        action_plan.extend(chunk[:replan_steps])

                    action = action_plan.popleft()
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        task_succ += 1
                        total_succ += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error(f"Episode error: {e}")
                    break

            task_eps += 1
            total_eps += 1
            suffix = "success" if done else "failure"
            video_p = pathlib.Path(video_out_path) / f"task{task_id}_ep{ep}_{suffix}.mp4"
            try:
                imageio.mimwrite(str(video_p), replay, fps=10)
            except Exception as e:
                logging.warning(f"Video save failed: {e}")
            if verbose:
                logging.info(f"  → {'SUCCESS' if done else 'FAILURE'}")

        env.close()
        sr = task_succ / task_eps if task_eps > 0 else 0.0
        task_results.append({
            "task_id": task_id,
            "task_description": desc,
            "success_rate": sr,
            "successes": task_succ,
            "episodes": task_eps,
        })
        logging.info(f"Task {task_id} ({desc[:40]}...): {sr:.1%}")

    return {
        "task_suite": task_suite_name,
        "overall_success_rate": total_succ / total_eps if total_eps > 0 else 0.0,
        "total_successes": total_succ,
        "total_episodes": total_eps,
        "task_results": task_results,
    }


def main():
    p = argparse.ArgumentParser(description="LIBERO BC-T rollout")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="BC-T best.pt (finetune_libero_bct.py 산출물)")
    p.add_argument("--task-suite", type=str, default="libero_spatial",
                   choices=list(TASK_SUITE_CONFIG.keys()))
    p.add_argument("--num-trials", type=int, default=50)
    p.add_argument("--task-ids", type=int, nargs="+", default=None,
                   help="task ID 부분집합 (sanity용). None=전체")
    p.add_argument("--replan-steps", type=int, default=1,
                   help="Re-plan interval (1 = inference every step)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", type=str, default="data/libero/results")
    p.add_argument("--video-dir", type=str, default="data/libero/videos")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    client = BCTransformerClient(args.checkpoint)

    print("=" * 70)
    print("LIBERO BC-T Rollout")
    print(f"  ckpt: {args.checkpoint}")
    print(f"  suite: {args.task_suite} | trials/task: {args.num_trials} | seed: {args.seed}")
    print("=" * 70)

    results = evaluate_libero(
        client, args.task_suite, args.num_trials,
        task_ids=args.task_ids,
        replan_steps=args.replan_steps,
        video_out_path=args.video_dir,
        seed=args.seed, verbose=not args.quiet,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_stem = pathlib.Path(args.checkpoint).stem
    results["metadata"] = {
        "checkpoint": args.checkpoint,
        "task_suite": args.task_suite,
        "num_trials_per_task": args.num_trials,
        "replan_steps": args.replan_steps,
        "seed": args.seed,
        "env_resolution": LIBERO_ENV_RESOLUTION,
        "max_steps": TASK_SUITE_CONFIG[args.task_suite]["max_steps"],
        "timestamp": ts,
    }

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"bct_{ckpt_stem}_{args.task_suite}_seed{args.seed}_{ts}.json"
    with open(out / fname, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Overall: {results['overall_success_rate']:.1%} "
          f"({results['total_successes']}/{results['total_episodes']})")
    print(f"Saved: {out / fname}")


if __name__ == "__main__":
    main()
