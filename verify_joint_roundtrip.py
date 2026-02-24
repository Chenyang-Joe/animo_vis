#!/usr/bin/env python3
"""Round-trip verification: input 30-joint positions → retarget → FK → 30-joint positions.

Checks that the 30 AniMo joint world trajectories are preserved through the
retarget pipeline (positions → local rotations → FK world positions).

Usage:
    python verify_joint_roundtrip.py skeleton.npy reference.glb
"""

import argparse
from pathlib import Path

import numpy as np

from core.glb_parser import parse_glb, _trs_to_mat4
from core.retarget import retarget, INVERSE_PERM, _build_animo_to_glb_chain
from core.decoder import decode_joint_vecs
from core.joint_config import JOINT_NAMES


def _fk_world_positions(local_rots_xyzw, root_trans, glb_data):
    """FK: local rotations → world joint positions for all GLB joints.

    Returns:
        world_pos: (T, N, 3)
    """
    T, N = local_rots_xyzw.shape[:2]
    parent_indices = glb_data.parent_indices

    # Build topo order
    visited = set()
    topo = []
    def _visit(si):
        if si in visited:
            return
        pi = parent_indices[si]
        if pi >= 0 and pi not in visited:
            _visit(pi)
        visited.add(si)
        topo.append(si)
    for si in range(N):
        _visit(si)

    # FK: build world matrices from local TRS
    world = np.zeros((T, N, 4, 4), dtype=np.float64)

    for si in topo:
        t_rest, r_rest, s_rest = glb_data.rest_local_trs[si]
        t_arr = np.tile(t_rest, (T, 1))  # (T, 3)
        s_arr = np.tile(s_rest, (T, 1))  # (T, 3)
        r_arr = local_rots_xyzw[:, si]    # (T, 4) xyzw

        # Override root translation with animated values
        if si == 0:
            t_arr = root_trans.copy()

        x, y, z, w = r_arr[:, 0], r_arr[:, 1], r_arr[:, 2], r_arr[:, 3]
        sx, sy, sz = s_arr[:, 0], s_arr[:, 1], s_arr[:, 2]

        local = np.zeros((T, 4, 4), dtype=np.float64)
        local[:, 3, 3] = 1.0
        local[:, 0, 0] = (1 - 2*(y*y + z*z)) * sx
        local[:, 0, 1] = (2*(x*y - z*w)) * sy
        local[:, 0, 2] = (2*(x*z + y*w)) * sz
        local[:, 1, 0] = (2*(x*y + z*w)) * sx
        local[:, 1, 1] = (1 - 2*(x*x + z*z)) * sy
        local[:, 1, 2] = (2*(y*z - x*w)) * sz
        local[:, 2, 0] = (2*(x*z - y*w)) * sx
        local[:, 2, 1] = (2*(y*z + x*w)) * sy
        local[:, 2, 2] = (1 - 2*(x*x + y*y)) * sz
        local[:, 0, 3] = t_arr[:, 0]
        local[:, 1, 3] = t_arr[:, 1]
        local[:, 2, 3] = t_arr[:, 2]

        pi = parent_indices[si]
        if pi < 0:
            world[:, si] = local
        else:
            world[:, si] = np.einsum('tij,tjk->tik', world[:, pi], local)

    return world[:, :, :3, 3]  # (T, N, 3)


def main():
    parser = argparse.ArgumentParser(
        description="Round-trip joint trajectory verification")
    parser.add_argument("npy_file", help="AniMo skeleton .npy")
    parser.add_argument("ref_glb", help="Reference GLB with mesh + skeleton")
    parser.add_argument("--mode", choices=["joints", "vecs"], default="joints")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    # ── Load inputs ──
    npy_path = Path(args.npy_file)
    ref_path = Path(args.ref_glb)

    print(f"Reference GLB: {ref_path}")
    glb_data = parse_glb(str(ref_path))
    n_joints = len(glb_data.joint_names)
    print(f"  {n_joints} GLB joints, {len(glb_data.animo_to_glb)}/30 AniMo mapped")

    if args.mode == "vecs":
        positions = decode_joint_vecs(str(npy_path))
    else:
        positions = np.load(str(npy_path))
        if positions.ndim == 2 and positions.shape[1] == 359:
            positions = decode_joint_vecs(str(npy_path))
    positions = positions.astype(np.float64)

    if args.max_frames:
        positions = positions[:args.max_frames]
    T = positions.shape[0]
    print(f"  {T} frames, input shape {positions.shape}")

    # ── Retarget ──
    print("Retargeting ...")
    local_rots_xyzw, root_trans, tracked = retarget(positions, glb_data)
    print(f"  {len(tracked)} tracked GLB joints")

    # ── FK: recover world joint positions ──
    print("Running FK ...")
    fk_world_pos = _fk_world_positions(local_rots_xyzw, root_trans, glb_data)
    # fk_world_pos: (T, N, 3)

    # ── Map AniMo input positions to GLB joint positions ──
    # chain_to_glb: AniMo_chain_idx → GLB_skin_idx
    chain_to_glb = _build_animo_to_glb_chain(glb_data)

    print(f"\n{'='*65}")
    print("ROUND-TRIP: input AniMo positions vs FK-recovered positions")
    print(f"{'='*65}")

    all_errors = []
    per_joint = {}

    for animo_idx in range(30):
        if animo_idx not in chain_to_glb:
            continue
        glb_idx = chain_to_glb[animo_idx]
        original_idx = INVERSE_PERM[animo_idx]
        joint_name = JOINT_NAMES[original_idx]
        glb_name = glb_data.joint_names[glb_idx]

        # Input position for this joint across all frames
        input_pos = positions[:, animo_idx]          # (T, 3)
        fk_pos = fk_world_pos[:, glb_idx]            # (T, 3)

        per_frame_err = np.sqrt(((input_pos - fk_pos) ** 2).sum(axis=-1))  # (T,)
        mean_err = per_frame_err.mean()
        max_err = per_frame_err.max()

        per_joint[animo_idx] = {
            'name': joint_name,
            'glb_name': glb_name,
            'mean': mean_err,
            'max': max_err,
        }
        all_errors.append(per_frame_err)

    all_errors = np.concatenate(all_errors)

    # Sort by mean error
    sorted_joints = sorted(per_joint.items(), key=lambda x: x[1]['mean'], reverse=True)

    print(f"\n  Per-joint L2 error (input pos vs FK pos):")
    print(f"  {'Joint':<35s} {'GLB Name':<40s} {'Mean':>8s} {'Max':>8s}")
    print(f"  {'-'*35} {'-'*40} {'-'*8} {'-'*8}")
    for animo_idx, info in sorted_joints:
        print(f"  {info['name']:<35s} {info['glb_name']:<40s} {info['mean']:8.5f} {info['max']:8.5f}")

    print(f"\n  Overall (all 30 joints x {T} frames):")
    print(f"    Mean error:   {all_errors.mean():.6f}")
    print(f"    Median error: {np.median(all_errors):.6f}")
    print(f"    P95 error:    {np.percentile(all_errors, 95):.6f}")
    print(f"    Max error:    {all_errors.max():.6f}")

    # ── Root position check (should be exact) ──
    root_glb = chain_to_glb[0]
    root_input = positions[:, 0]
    root_fk = fk_world_pos[:, root_glb]
    root_err = np.sqrt(((root_input - root_fk) ** 2).sum(axis=-1))
    print(f"\n  Root joint (should be exact):")
    print(f"    Mean: {root_err.mean():.10f}  Max: {root_err.max():.10f}")

    # ── Direction check: bone directions should match ──
    # For each joint pair (parent, child) in AniMo chains, compare
    # the bone direction between input and FK
    print(f"\n{'='*65}")
    print("BONE DIRECTION ERROR (cosine distance)")
    print(f"{'='*65}")

    from core.joint_config import KINEMATIC_CHAIN
    dir_errors = []
    for chain in KINEMATIC_CHAIN:
        for i in range(len(chain) - 1):
            parent_ai = chain[i]
            child_ai = chain[i + 1]
            if parent_ai not in chain_to_glb or child_ai not in chain_to_glb:
                continue

            parent_glb = chain_to_glb[parent_ai]
            child_glb = chain_to_glb[child_ai]

            # Input bone direction
            in_dir = positions[:, child_ai] - positions[:, parent_ai]
            in_len = np.linalg.norm(in_dir, axis=-1, keepdims=True)
            in_dir = in_dir / np.maximum(in_len, 1e-10)

            # FK bone direction
            fk_dir = fk_world_pos[:, child_glb] - fk_world_pos[:, parent_glb]
            fk_len = np.linalg.norm(fk_dir, axis=-1, keepdims=True)
            fk_dir = fk_dir / np.maximum(fk_len, 1e-10)

            # Cosine similarity
            cos_sim = (in_dir * fk_dir).sum(axis=-1)
            cos_dist = 1.0 - cos_sim  # 0 = perfect match

            # Bone length ratio
            bl_ratio = fk_len.squeeze() / np.maximum(in_len.squeeze(), 1e-10)

            parent_name = JOINT_NAMES[INVERSE_PERM[parent_ai]]
            child_name = JOINT_NAMES[INVERSE_PERM[child_ai]]

            dir_errors.append({
                'label': f"{parent_name}→{child_name}",
                'cos_dist_mean': cos_dist.mean(),
                'cos_dist_max': cos_dist.max(),
                'bl_ratio_mean': bl_ratio.mean(),
                'bl_ratio_std': bl_ratio.std(),
            })

    dir_errors.sort(key=lambda x: x['cos_dist_mean'], reverse=True)

    print(f"\n  {'Bone':<40s} {'CosDist':>8s} {'BLRatio':>8s} {'BLStd':>8s}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    for d in dir_errors:
        print(f"  {d['label']:<40s} {d['cos_dist_mean']:8.5f} {d['bl_ratio_mean']:8.4f} {d['bl_ratio_std']:8.4f}")

    all_cos = [d['cos_dist_mean'] for d in dir_errors]
    all_bl = [d['bl_ratio_mean'] for d in dir_errors]
    print(f"\n  Direction: mean cos_dist = {np.mean(all_cos):.6f} (0 = perfect)")
    print(f"  Bone length ratio: mean = {np.mean(all_bl):.4f} ± {np.std(all_bl):.4f}")

    print(f"\n{'='*65}")
    print("INTERPRETATION")
    print(f"{'='*65}")
    print(f"  Position error comes from TWO sources:")
    print(f"    1. Bone DIRECTION error (cos_dist): retarget loses some direction accuracy")
    print(f"    2. Bone LENGTH mismatch: retarget uses GLB rest-pose bone lengths,")
    print(f"       not AniMo predicted lengths. Same direction + different length = offset.")
    print(f"  If bone directions are near-zero and bone length ratios are ~1.0,")
    print(f"  the round-trip is perfect. If bone length ratios differ, the GLB skeleton")
    print(f"  has different proportions from AniMo's implicit skeleton.")

    print(f"\n{'='*65}")
    print("DONE")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
