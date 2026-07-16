#!/usr/bin/env python
"""LIBERO BC-Transformer policy rollout (closed-loop simulator evaluation).

Evaluates the training artifact (`best.pt` from finetune_libero_bct.py) in the
LIBERO simulator, closed-loop. The encoder checkpoint path baked into the
config is irrelevant here -- `policy_state_dict` already contains all adapter
weights, so the encoder checkpoint is overridden to None and everything is
loaded at once via `load_state_dict`.

Usage:
    python eval/eval_libero.py \\
        --checkpoint <CHECKPOINT_DIR>/bct_comp-mae_libero_spatial_seed0_best.pt \\
        --task-suite libero_spatial \\
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

# robomimic 0.x <-> numpy 1.20+ compatibility
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# LIBERO (importable only inside the LIBERO eval environment)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Bootstrap: make the supplement root (containing model/ and eval/) importable.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # supplement root (contains model/, eval/)

from eval.policies.bc_transformer_adapted import AdaptedBCTransformerPolicy


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
# Same resolution as the HDF5 demos (`agentview_rgb`). At training time
# robomimic passes the 128x128 frames straight to the adapter (which resizes
# internally), so the rollout must render the env at the same resolution to
# match the training distribution.
LIBERO_ENV_RESOLUTION = 128


def libero_shape_meta() -> Dict[str, Any]:
    """LIBERO standard shape_meta -- task-suite-independent. Hardcodes the
    values robomimic extracted from HDF5 during BC-T training, so no dataset is
    needed at rollout time.

    Includes `joint_states` to match training with `use_joint=True` (LIBERO
    official default).
    """
    return {
        "ac_dim": 7,
        "all_shapes": collections.OrderedDict([
            ("agentview_rgb", [3, 128, 128]),
            ("eye_in_hand_rgb", [3, 128, 128]),
            ("gripper_states", [2]),
            ("joint_states", [7]),
        ]),
        "all_obs_keys": [
            "agentview_rgb", "eye_in_hand_rgb",
            "gripper_states", "joint_states",
        ],
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

    Calls `spatial_encode`/`temporal_encode`/`policy_head` directly, matching
    the training flow (bypassing `get_action`'s `preprocess_input` to keep the
    input distribution consistent).
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        logging.info(f"Loading BC-T ckpt: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(ckpt["config"])

        # The encoder checkpoint path may be unavailable here -> random-init the
        # encoder and overwrite it from the policy state dict below.
        cfg.encoder.checkpoint = None

        # The comp-mae adapter infers its architecture from the pretrain
        # ckpt, which is absent at rollout time -> infer the architecture from
        # policy_state_dict (self-contained, includes the encoder) and inject it.
        if str(cfg.encoder.type).lower().replace("_", "-") == "comp-mae":
            psd = ckpt["policy_state_dict"]
            ak = dict(cfg.encoder.get("adapter_kwargs", {}))
            ak["embed_dim"] = int(psd["adapter.model.pos_embed_p"].shape[-1])
            ak["m_depth"] = len({k.split("blocks_m.")[1].split(".")[0]
                                 for k in psd if "adapter.model.blocks_m." in k})
            ak["comp_mae"] = any(k.startswith("adapter.model.") and "m_recon" in k
                                 for k in psd)
            cfg.encoder.adapter_kwargs = ak

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
        self.encoder_type = str(cfg.encoder.type)
        self.train_epoch = ckpt.get("epoch")
        self.train_eval_loss = ckpt.get("eval_loss")

        # CLIP for task_emb (same setup as training: use pooler_output)
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
        """LIBERO env image (H, W, 3) uint8 -> (1, 3, img_size, img_size) [0,1]."""
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        if x.shape[1] != self.img_size or x.shape[2] != self.img_size:
            x = F.interpolate(
                x.unsqueeze(0), size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return x.unsqueeze(0).to(self.device)  # (1, 3, H, W)

    def reset(self):
        # Training runs on (B, T=seq_len, ...) sequences. The pair-based
        # adapters form their prev/curr pair only when T>1. At rollout the
        # adapter's internal prev_obs cache would be cross-camera contaminated,
        # because a single adapter instance is shared by both cameras
        # (agentview, wrist). So instead we keep a raw obs history here and call
        # spatial_encode on the full (B=1, T=T_acc) sequence each step (T>1
        # branch active); latent_queue is not used.
        self.policy.reset()
        self.obs_history: list = []  # list of dict per timestep
        self.max_seq_len = self.policy.max_seq_len  # same as training seq_len

    def observe(self, obs: Dict[str, Any]) -> None:
        """Accumulate obs into history without running inference. Called during
        the dummy-wait period so that at the first inference the adapter sees a
        real (prev=t-1, curr=t) motion pair. Motion-aware adapters collapse the
        motion feature on an (im, im) pair, so real motion is required from the
        first step.
        """
        agent = self._img_to_tensor(obs["observation/image"])
        wrist = self._img_to_tensor(obs["observation/wrist_image"])
        state = obs["observation/state"]
        gripper = torch.from_numpy(state[-2:]).float().view(1, 2).to(self.device)
        joint = torch.from_numpy(np.asarray(obs["observation/joint_pos"])
                                 ).float().view(1, 7).to(self.device)
        self.obs_history.append({
            "agentview_rgb": agent, "eye_in_hand_rgb": wrist,
            "gripper_states": gripper, "joint_states": joint,
        })
        if len(self.obs_history) > self.max_seq_len:
            self.obs_history.pop(0)

    @torch.no_grad()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Single-step inference. Returns {"actions": np.ndarray (1, 7)}.

        Calls spatial_encode on the same sequence distribution as the training
        forward.
        """
        agent = self._img_to_tensor(obs["observation/image"])      # (1, 3, H, W)
        wrist = self._img_to_tensor(obs["observation/wrist_image"])
        state = obs["observation/state"]
        gripper = torch.from_numpy(state[-2:]).float().view(1, 2).to(self.device)
        joint = torch.from_numpy(np.asarray(obs["observation/joint_pos"])
                                 ).float().view(1, 7).to(self.device)

        self.obs_history.append({
            "agentview_rgb": agent,
            "eye_in_hand_rgb": wrist,
            "gripper_states": gripper,
            "joint_states": joint,
        })
        if len(self.obs_history) > self.max_seq_len:
            self.obs_history.pop(0)

        # Build the same (B=1, T_acc, ...) sequence as training.
        T_acc = len(self.obs_history)
        data = {
            "obs": {
                k: torch.stack([h[k] for h in self.obs_history], dim=1)
                # (1, T_acc, ...) -- stack along time dim
                for k in self.obs_history[0]
            },
            "task_emb": self._task_emb(str(obs["prompt"])).unsqueeze(0),  # (1, 512)
        }

        x = self.policy.spatial_encode(data)   # (1, T_acc, num_mod, E)
        x = self.policy.temporal_encode(x)     # (1, T_acc, E)
        dist = self.policy.policy_head(x[:, -1])  # GMM at last step

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
    logging.info(f"{task_suite_name}: tasks={target_ids} x {num_trials_per_task} trials, max_steps={max_steps}")

    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    total_eps, total_succ = 0, 0
    task_results = []

    for task_id in target_ids:
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env, desc = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)

        task_eps, task_succ = 0, 0
        episode_records = []
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
            errored = False
            while t < max_steps + num_steps_wait:
                try:
                    if t < num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        # Accumulate the last max_seq_len steps of the dummy
                        # period into history, so the first inference sees a real
                        # motion-pair sequence of length T_acc=max_seq_len
                        # (avoids the step-0 collapse of motion-aware adapters).
                        if t >= num_steps_wait - client.max_seq_len:
                            state = np.concatenate([
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            ])
                            client.observe({
                                "observation/image": obs["agentview_image"],
                                "observation/wrist_image": obs["robot0_eye_in_hand_image"],
                                "observation/state": state,
                                "observation/joint_pos": obs["robot0_joint_pos"],
                            })
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
                            "observation/joint_pos": obs["robot0_joint_pos"],
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
                    errored = True
                    break

            task_eps += 1
            total_eps += 1
            suffix = "success" if done else "failure"
            video_p = pathlib.Path(video_out_path) / f"task{task_id}_ep{ep}_{suffix}.mp4"
            try:
                imageio.mimwrite(str(video_p), replay, fps=10)
            except Exception as e:
                logging.warning(f"Video save failed: {e}")
            episode_records.append({
                "ep_id": ep,
                "success": bool(done),
                "steps_to_done": int(t),
                "errored": bool(errored),
            })
            if verbose:
                logging.info(f"  -> {'SUCCESS' if done else 'FAILURE'}")

        env.close()
        sr = task_succ / task_eps if task_eps > 0 else 0.0
        task_results.append({
            "task_id": task_id,
            "task_description": desc,
            "success_rate": sr,
            "successes": task_succ,
            "episodes": task_eps,
            "episode_records": episode_records,
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
                   help="BC-T best.pt (produced by finetune_libero_bct.py)")
    p.add_argument("--task-suite", type=str, default="libero_spatial",
                   choices=list(TASK_SUITE_CONFIG.keys()))
    p.add_argument("--num-trials", type=int, default=50)
    p.add_argument("--task-ids", type=int, nargs="+", default=None,
                   help="Subset of task IDs (for sanity tests). None = all tasks")
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
        "encoder_type": client.encoder_type,
        "task_suite": args.task_suite,
        "num_trials_per_task": args.num_trials,
        "replan_steps": args.replan_steps,
        "seed": args.seed,
        "env_resolution": LIBERO_ENV_RESOLUTION,
        "max_steps": TASK_SUITE_CONFIG[args.task_suite]["max_steps"],
        "train_epoch": client.train_epoch,
        "train_eval_loss": (
            float(client.train_eval_loss)
            if client.train_eval_loss is not None else None
        ),
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
