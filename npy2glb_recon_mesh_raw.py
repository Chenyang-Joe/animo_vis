#!/usr/bin/env python3
"""npy2glb_recon_mesh without tiered non-tracked joint optimization.

Only the 30 AniMo-tracked joints receive animation deltas; all other joints
keep their rest-pose local rotation. Use this to compare against the optimized
version (npy2glb_recon_mesh.py) which adds slerp/twist/tail deltas for ~20
additional joints.

Usage:
    python npy2glb_recon_mesh_raw.py skeleton.npy reference.glb --output out.glb
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from core.glb_parser import parse_glb
from core.retarget import retarget
from core.glb_writer import build_animation_glb
from core.decoder import decode_joint_vecs


def main():
    parser = argparse.ArgumentParser(
        description="Convert AniMo skeleton .npy to mesh-animated GLB (raw, no tiered optimization)",
    )
    parser.add_argument("npy_file", help="Skeleton sequence .npy (T,30,3) or (T,359)")
    parser.add_argument("ref_glb", help="Reference GLB with mesh + skeleton")
    parser.add_argument("--mode", choices=["joints", "vecs"], default="joints",
                        help="Input format: 'joints' for (T,30,3), 'vecs' for (T,359)")
    parser.add_argument("--fps", type=float, default=30.0, help="Animation framerate")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Truncate animation to N frames")
    parser.add_argument("--output", type=str, default=None,
                        help="Output GLB path (default: <npy_stem>_mesh_raw.glb)")
    args = parser.parse_args()

    npy_path = Path(args.npy_file)
    ref_path = Path(args.ref_glb)

    if not npy_path.exists():
        print(f"Error: {npy_path} not found", file=sys.stderr)
        sys.exit(1)
    if not ref_path.exists():
        print(f"Error: {ref_path} not found", file=sys.stderr)
        sys.exit(1)

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = npy_path.with_name(npy_path.stem + "_mesh_raw.glb")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load skeleton positions
    if args.mode == "vecs":
        print(f"Decoding joint vecs from {npy_path} ...")
        positions = decode_joint_vecs(str(npy_path))  # (T, 30, 3)
    else:
        positions = np.load(str(npy_path))  # (T, 30, 3)
        if positions.ndim == 2 and positions.shape[1] == 359:
            print("Detected (T,359) input, switching to vecs mode.")
            positions = decode_joint_vecs(str(npy_path))

    if positions.ndim != 3 or positions.shape[1] != 30 or positions.shape[2] != 3:
        print(f"Error: Expected (T,30,3) positions, got {positions.shape}", file=sys.stderr)
        sys.exit(1)

    positions = positions.astype(np.float64)

    if args.max_frames:
        positions = positions[:args.max_frames]

    T = positions.shape[0]
    print(f"Loaded {T} frames, shape {positions.shape}")

    # Parse reference GLB
    print(f"Parsing reference GLB: {ref_path} ...")
    glb_data = parse_glb(str(ref_path))
    print(f"  {len(glb_data.joint_names)} joints, "
          f"{len(glb_data.animo_to_glb)}/30 AniMo joints mapped")

    if len(glb_data.animo_to_glb) < 30:
        missing = []
        from core.joint_config import JOINT_NAMES
        for i, name in enumerate(JOINT_NAMES):
            if i not in glb_data.animo_to_glb:
                missing.append(name)
        print(f"  Warning: unmapped joints: {missing}", file=sys.stderr)

    # Retarget (tiered=False → only 30 tracked joints, no slerp/twist/tail)
    print("Retargeting (raw, no tiered optimization) ...")
    local_rots, root_trans, tracked = retarget(positions, glb_data, tiered=False)
    print(f"  {len(tracked)} tracked joints, {local_rots.shape[1]} total joints")

    # Write output GLB
    print(f"Writing {out_path} ...")
    build_animation_glb(
        glb_data=glb_data,
        local_rotations_xyzw=local_rots,
        root_translations=root_trans,
        tracked_glb_indices=tracked,
        fps=args.fps,
        output_path=str(out_path),
    )
    print(f"Done! Output: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
