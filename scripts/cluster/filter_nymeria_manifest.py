#!/usr/bin/env python3
"""Nymeria download manifest을 subset으로 필터링.

전체 manifest(51 TB)에서 우리가 실제 필요한 subset만 추출한 필터된 manifest를 생성.

포함:
  - video_main_rgb           (RGB preview mp4, 1.06 TB)
  - body_motion              (body xdata npz + glb, 1.09 TB)
  - narration_*              (atomic action / activity / motion CSV, <1 MB)
  - metadata_json, LICENSE   (무시할 만큼 작음)

제외 (우리 시각 표현 연구에 불필요):
  - recording_*_data_data_vrs  (33 TB, raw VRS 센서)
  - semidense_observations     (11 TB, SLAM feature obs)
  - recording_*/mps/*          (2.3 TB, SLAM trajectories / eye gaze)
  - body_xdata_mvnx            (1.96 TB, XSens mocap raw)

Usage:
  python scripts/cluster/filter_nymeria_manifest.py \\
      --input  /proj/external_group/mrg/datasets/nymeria/manifests/nymeria_v0.0_urls_full.json \\
      --output /proj/external_group/mrg/datasets/nymeria/manifests/nymeria_v0.0_urls_video_motion.json
"""

import argparse
import copy
import json
from pathlib import Path

# 유지할 data group 정확한 키
KEEP_GROUPS = {
    "video_main_rgb",
    "body_motion",
    "narration_atomic_action_csv",
    "narration_activity_summarization_csv",
    "narration_motion_narration_csv",
    "metadata_json",
    "LICENSE",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.input) as f:
        manifest = json.load(f)

    seqs = manifest["sequences"]
    print(f"Input: {len(seqs)} sequences")

    total_in = 0
    total_out = 0
    out_seqs = {}
    for sid, seq in seqs.items():
        kept = {}
        for g, info in seq.items():
            sz = info.get("file_size_bytes", 0) if isinstance(info, dict) else 0
            total_in += sz
            if g in KEEP_GROUPS:
                kept[g] = info
                total_out += sz
        out_seqs[sid] = kept

    out = copy.deepcopy(manifest)
    out["sequences"] = out_seqs

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)

    print(f"Output: {len(out_seqs)} sequences → {args.output}")
    print(f"Size before filter: {total_in / 1e12:.2f} TB")
    print(f"Size after  filter: {total_out / 1e12:.2f} TB ({100 * total_out / total_in:.1f}% of full)")


if __name__ == "__main__":
    main()
