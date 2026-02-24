#!/usr/bin/env python3
"""Compare per-vertex skinned positions between reference GLB animation
and our reconstructed GLB to verify scale equivalence and overall fidelity.

Usage:
    python verify_vertex_error.py \
        skeleton.npy \
        reference.glb \
        [--max-frames N] [--mode joints|vecs]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

from core.glb_parser import parse_glb, _read_accessor, _trs_to_mat4
from core.retarget import retarget
from core.decoder import decode_joint_vecs


# ── GLB constants ──────────────────────────────────────────────────────────
GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN  = 0x004E4942

_COMP = {
    5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
    5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4),
}
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


# ── Mesh data extraction ──────────────────────────────────────────────────

def _extract_mesh_data(gltf, buf):
    """Extract all vertex positions, joint indices, and weights from all mesh primitives.

    Returns:
        positions: (V, 3) rest-pose vertex positions
        joints:    (V, 4) joint indices (skin.joints indices)
        weights:   (V, 4) blend weights
    """
    all_pos = []
    all_joints = []
    all_weights = []

    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs or "JOINTS_0" not in attrs:
                continue
            pos = _read_accessor(gltf, buf, attrs["POSITION"])       # (N, 3)
            jts = _read_accessor(gltf, buf, attrs["JOINTS_0"])       # (N, 4)
            wts = _read_accessor(gltf, buf, attrs["WEIGHTS_0"])      # (N, 4)
            all_pos.append(pos)
            all_joints.append(jts.astype(np.int32))
            all_weights.append(wts)

    return (np.concatenate(all_pos, axis=0),
            np.concatenate(all_joints, axis=0),
            np.concatenate(all_weights, axis=0))


# ── Reference animation extraction ────────────────────────────────────────

def _extract_ref_animation(gltf, buf, n_joints, joint_node_indices, rest_local_trs):
    """Extract animation from reference GLB → per-frame local TRS for all joints.

    Returns:
        T:     number of frames
        translations: (T, n_joints, 3)
        rotations:    (T, n_joints, 4) xyzw
        scales:       (T, n_joints, 3)
    """
    anim = gltf["animations"][0]
    node_to_sjidx = {ni: si for si, ni in enumerate(joint_node_indices)}

    # Determine max frame count across all samplers
    T = 0
    for s in anim["samplers"]:
        cnt = gltf["accessors"][s["input"]]["count"]
        T = max(T, cnt)

    # Find the time array from a sampler with T keyframes (full animation)
    master_times = None
    for s in anim["samplers"]:
        cnt = gltf["accessors"][s["input"]]["count"]
        if cnt == T:
            master_times = _read_accessor(gltf, buf, s["input"]).ravel()
            break

    # Initialize with rest pose
    translations = np.zeros((T, n_joints, 3), dtype=np.float64)
    rotations = np.zeros((T, n_joints, 4), dtype=np.float64)
    scales = np.ones((T, n_joints, 3), dtype=np.float64)

    for si in range(n_joints):
        t, r_xyzw, s = rest_local_trs[si]
        translations[:, si] = t
        rotations[:, si] = r_xyzw
        scales[:, si] = s

    # Overwrite with animated channels, interpolating if needed
    for ch in anim["channels"]:
        node_idx = ch["target"]["node"]
        if node_idx not in node_to_sjidx:
            continue
        skin_idx = node_to_sjidx[node_idx]
        path = ch["target"]["path"]
        sampler = anim["samplers"][ch["sampler"]]
        data = _read_accessor(gltf, buf, sampler["output"])
        n_keys = data.shape[0]

        if n_keys == T:
            # Full keyframes — use directly
            values = data
        elif n_keys == 1:
            # Constant
            values = np.tile(data[0], (T, 1))
        elif n_keys == 2 and master_times is not None:
            # 2 keyframes (likely constant or linear start/end) — interpolate
            times = _read_accessor(gltf, buf, sampler["input"]).ravel()
            t0, t1 = times[0], times[1]
            dt = t1 - t0
            if abs(dt) < 1e-10:
                values = np.tile(data[0], (T, 1))
            else:
                alpha = ((master_times - t0) / dt).clip(0, 1)[:, np.newaxis]
                values = (1 - alpha) * data[0] + alpha * data[1]
        else:
            # General case: linear interpolation over time axis
            times = _read_accessor(gltf, buf, sampler["input"]).ravel()
            values = np.zeros((T, data.shape[1]), dtype=np.float64)
            for ci in range(data.shape[1]):
                values[:, ci] = np.interp(master_times, times, data[:, ci])

        if path == "translation":
            translations[:, skin_idx] = values
        elif path == "rotation":
            rotations[:, skin_idx] = values
        elif path == "scale":
            scales[:, skin_idx] = values

    return T, translations, rotations, scales


# ── Forward kinematics ────────────────────────────────────────────────────

def _fk_world_matrices(translations, rotations_xyzw, scales,
                       parent_indices, topo_order):
    """Compute world matrices from local TRS for all joints across all frames.

    Args:
        translations:   (T, N, 3)
        rotations_xyzw: (T, N, 4)
        scales:         (T, N, 3)
        parent_indices: list of ints, -1 for root
        topo_order:     topological order of joints

    Returns:
        world_matrices: (T, N, 4, 4)
    """
    T, N = translations.shape[:2]
    world = np.zeros((T, N, 4, 4), dtype=np.float64)

    for si in topo_order:
        for t in range(T):
            tr = translations[t, si]
            r_xyzw = rotations_xyzw[t, si]
            sc = scales[t, si]
            local = _trs_to_mat4(tr, r_xyzw, sc)
            pi = parent_indices[si]
            if pi < 0:
                world[t, si] = local
            else:
                world[t, si] = world[t, pi] @ local

    return world


def _fk_world_matrices_fast(translations, rotations_xyzw, scales,
                            parent_indices, topo_order):
    """Vectorized version: compute world matrices for all frames at once."""
    T, N = translations.shape[:2]
    world = np.zeros((T, N, 4, 4), dtype=np.float64)

    # Precompute all local matrices in bulk
    for si in topo_order:
        # Build local matrix for all T frames at once
        t_all = translations[:, si]          # (T, 3)
        r_all = rotations_xyzw[:, si]        # (T, 4) xyzw
        s_all = scales[:, si]                # (T, 3)

        x, y, z, w = r_all[:, 0], r_all[:, 1], r_all[:, 2], r_all[:, 3]
        sx, sy, sz = s_all[:, 0], s_all[:, 1], s_all[:, 2]

        local = np.zeros((T, 4, 4), dtype=np.float64)
        local[:, 3, 3] = 1.0
        local[:, 0, 0] = (1 - 2*(y*y + z*z)) * sx
        local[:, 0, 1] = (2*(x*y - z*w))     * sy
        local[:, 0, 2] = (2*(x*z + y*w))     * sz
        local[:, 1, 0] = (2*(x*y + z*w))     * sx
        local[:, 1, 1] = (1 - 2*(x*x + z*z)) * sy
        local[:, 1, 2] = (2*(y*z - x*w))     * sz
        local[:, 2, 0] = (2*(x*z - y*w))     * sx
        local[:, 2, 1] = (2*(y*z + x*w))     * sy
        local[:, 2, 2] = (1 - 2*(x*x + y*y)) * sz
        local[:, 0, 3] = t_all[:, 0]
        local[:, 1, 3] = t_all[:, 1]
        local[:, 2, 3] = t_all[:, 2]

        pi = parent_indices[si]
        if pi < 0:
            world[:, si] = local
        else:
            # Batched matrix multiply: (T,4,4) @ (T,4,4)
            world[:, si] = np.einsum('tij,tjk->tik', world[:, pi], local)

    return world


# ── Linear Blend Skinning ─────────────────────────────────────────────────

def _lbs(rest_positions, joint_indices, weights,
         world_matrices, ibms):
    """Linear Blend Skinning: compute skinned vertex positions.

    Args:
        rest_positions:  (V, 3) rest-pose vertices
        joint_indices:   (V, 4) joint indices per vertex
        weights:         (V, 4) blend weights per vertex
        world_matrices:  (T, N, 4, 4) animated joint world matrices
        ibms:            (N, 4, 4) inverse bind matrices

    Returns:
        skinned: (T, V, 3) skinned vertex positions
    """
    T = world_matrices.shape[0]
    V = rest_positions.shape[0]

    # Precompute skinning matrices: skin_mat[t, j] = world[t, j] @ ibm[j]
    # (T, N, 4, 4) @ (N, 4, 4) → (T, N, 4, 4)
    N = world_matrices.shape[1]
    skin_mats = np.einsum('tnij,njk->tnik', world_matrices, ibms)

    # Rest positions in homogeneous coords: (V, 4)
    rest_homo = np.ones((V, 4), dtype=np.float64)
    rest_homo[:, :3] = rest_positions

    skinned = np.zeros((T, V, 3), dtype=np.float64)

    for k in range(4):  # 4 influences per vertex
        jidx = joint_indices[:, k]  # (V,)
        w = weights[:, k]            # (V,)
        # Gather skin matrices for each vertex's k-th influence: (T, V, 4, 4)
        mats = skin_mats[:, jidx]    # (T, V, 4, 4)
        # Transform rest positions: (T, V, 4, 4) @ (V, 4) → (T, V, 4)
        transformed = np.einsum('tvij,vj->tvi', mats, rest_homo)
        skinned += w[np.newaxis, :, np.newaxis] * transformed[:, :, :3]

    return skinned


# ── Reconstruction: our pipeline → world matrices ─────────────────────────

def _recon_world_matrices(local_rots_xyzw, root_trans, glb_data, topo_order):
    """Build world matrices from our retarget output.

    Our pipeline outputs:
      - local_rotations_xyzw (T, N, 4) for all joints
      - root_translations    (T, 3)
    Non-root joints keep rest-pose translation and scale.
    """
    T, N = local_rots_xyzw.shape[:2]
    parent_indices = glb_data.parent_indices

    translations = np.zeros((T, N, 3), dtype=np.float64)
    scales = np.ones((T, N, 3), dtype=np.float64)

    # Fill rest-pose translation and scale
    for si in range(N):
        t_rest, r_rest, s_rest = glb_data.rest_local_trs[si]
        translations[:, si] = t_rest
        scales[:, si] = s_rest

    # Override root translation with animated values
    root_si = 0  # first skin joint = root
    translations[:, root_si] = root_trans

    return _fk_world_matrices_fast(translations, local_rots_xyzw, scales,
                                   parent_indices, topo_order)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify per-vertex error between reference and reconstructed GLB")
    parser.add_argument("npy_file", help="AniMo skeleton .npy")
    parser.add_argument("ref_glb", help="Reference GLB with mesh + animation")
    parser.add_argument("--mode", choices=["joints", "vecs"], default="joints")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sample-frames", type=int, default=None,
                        help="Subsample every N-th frame for speed")
    args = parser.parse_args()

    ref_path = Path(args.ref_glb)
    npy_path = Path(args.npy_file)

    # ── 1. Parse reference GLB ──
    print(f"Parsing reference GLB: {ref_path}")
    glb_data = parse_glb(str(ref_path))
    n_joints = len(glb_data.joint_names)
    print(f"  {n_joints} joints, {len(glb_data.animo_to_glb)}/30 AniMo mapped")

    # Re-read raw GLB for mesh + animation data
    with open(str(ref_path), "rb") as f:
        raw = f.read()
    json_len = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20:20 + json_len])
    bin_offset = 20 + json_len
    bin_len = struct.unpack_from("<I", raw, bin_offset)[0]
    buf = raw[bin_offset + 8: bin_offset + 8 + bin_len]

    # ── 2. Extract mesh data ──
    rest_pos, joint_idx, weights = _extract_mesh_data(gltf, buf)
    V = rest_pos.shape[0]
    print(f"  {V} vertices across {len(gltf['meshes'])} meshes")

    # ── 3. Extract reference animation ──
    joint_node_indices = gltf["skins"][0]["joints"]
    T_ref, ref_trans, ref_rots, ref_scales = _extract_ref_animation(
        gltf, buf, n_joints, joint_node_indices, glb_data.rest_local_trs)
    print(f"  Reference animation: {T_ref} frames")

    # ── 4. Load AniMo skeleton + retarget ──
    if args.mode == "vecs":
        positions = decode_joint_vecs(str(npy_path))
    else:
        positions = np.load(str(npy_path))
        if positions.ndim == 2 and positions.shape[1] == 359:
            positions = decode_joint_vecs(str(npy_path))
    positions = positions.astype(np.float64)
    T_npy = positions.shape[0]
    print(f"  AniMo input: {T_npy} frames")

    # Match frame counts
    T = min(T_ref, T_npy)
    if args.max_frames:
        T = min(T, args.max_frames)
    print(f"  Using {T} frames for comparison")

    positions = positions[:T]

    print("Retargeting ...")
    local_rots_xyzw, root_trans, tracked = retarget(positions, glb_data)
    print(f"  {len(tracked)} tracked joints")

    # ── 5. Topo order ──
    topo_order = []
    visited = set()
    def _topo(si):
        if si in visited:
            return
        pi = glb_data.parent_indices[si]
        if pi >= 0 and pi not in visited:
            _topo(pi)
        visited.add(si)
        topo_order.append(si)
    for si in range(n_joints):
        _topo(si)

    # ── 6. Subsample for speed if requested ──
    frame_indices = list(range(T))
    if args.sample_frames and args.sample_frames > 1:
        frame_indices = frame_indices[::args.sample_frames]
        print(f"  Subsampled to {len(frame_indices)} frames (every {args.sample_frames})")

    ref_trans_sub = ref_trans[frame_indices]
    ref_rots_sub = ref_rots[frame_indices]
    ref_scales_sub = ref_scales[frame_indices]
    local_rots_sub = local_rots_xyzw[frame_indices]
    root_trans_sub = root_trans[frame_indices]

    # ── 7. Compute world matrices ──
    print("Computing reference world matrices ...")
    ref_world = _fk_world_matrices_fast(
        ref_trans_sub, ref_rots_sub, ref_scales_sub,
        glb_data.parent_indices, topo_order)

    print("Computing reconstruction world matrices ...")
    recon_world = _recon_world_matrices(
        local_rots_sub, root_trans_sub, glb_data, topo_order)

    # ── 8. LBS ──
    ibms = glb_data.inverse_bind_matrices
    print(f"Computing LBS for {len(frame_indices)} frames x {V} vertices ...")
    ref_skinned = _lbs(rest_pos, joint_idx, weights, ref_world, ibms)
    recon_skinned = _lbs(rest_pos, joint_idx, weights, recon_world, ibms)

    # ── 8b. Rest-pose bone length comparison (true scale check) ──
    # Both use the same GLB rest pose, so bone lengths MUST match exactly.
    # Any difference indicates a bug in FK computation, not a scale issue.
    rest_world_mats = glb_data.rest_world_matrices
    rest_joint_pos = rest_world_mats[:, :3, 3]  # (N, 3)

    rest_bone_lengths = []
    bone_labels = []
    for si in range(n_joints):
        pi = glb_data.parent_indices[si]
        if pi < 0:
            continue
        bl = np.linalg.norm(rest_joint_pos[si] - rest_joint_pos[pi])
        if bl > 0.001:  # skip degenerate zero-length bones (twist helpers)
            rest_bone_lengths.append(bl)
            bone_labels.append(f"{glb_data.joint_names[pi]}→{glb_data.joint_names[si]}")

    rest_bl_arr = np.array(rest_bone_lengths)
    total_skeleton_length = rest_bl_arr.sum()

    # Animated bone lengths at frame 0 (affected by pose differences)
    ref_joint_pos_f0 = ref_world[0, :, :3, 3]
    recon_joint_pos_f0 = recon_world[0, :, :3, 3]

    ref_anim_bl = []
    recon_anim_bl = []
    anim_labels = []
    for si in range(n_joints):
        pi = glb_data.parent_indices[si]
        if pi < 0:
            continue
        r_bl = np.linalg.norm(ref_joint_pos_f0[si] - ref_joint_pos_f0[pi])
        rc_bl = np.linalg.norm(recon_joint_pos_f0[si] - recon_joint_pos_f0[pi])
        if r_bl > 0.001:  # skip degenerate bones
            ref_anim_bl.append(r_bl)
            recon_anim_bl.append(rc_bl)
            anim_labels.append(f"{glb_data.joint_names[pi]}→{glb_data.joint_names[si]}")

    ref_abl = np.array(ref_anim_bl)
    recon_abl = np.array(recon_anim_bl)
    abl_ratio = recon_abl / np.maximum(ref_abl, 1e-10)

    print(f"\n{'='*60}")
    print("REST-POSE SKELETON (shared by both ref and recon)")
    print(f"{'='*60}")
    print(f"  {len(rest_bone_lengths)} non-degenerate bones")
    print(f"  Total skeleton length: {total_skeleton_length:.4f}")
    print(f"  Bone lengths: min={rest_bl_arr.min():.4f}  max={rest_bl_arr.max():.4f}  mean={rest_bl_arr.mean():.4f}")

    print(f"\n{'='*60}")
    print("ANIMATED BONE LENGTH RATIO (frame 0, ref vs recon)")
    print(f"{'='*60}")
    print(f"  {len(ref_anim_bl)} bones (>1mm)")
    print(f"  Ratio: mean={abl_ratio.mean():.6f}  std={abl_ratio.std():.6f}")
    print(f"  Ratio: min={abl_ratio.min():.6f}  max={abl_ratio.max():.6f}")
    # Show largest deviations
    deviations = np.abs(abl_ratio - 1.0)
    worst_idx = np.argsort(deviations)[::-1][:5]
    print(f"  Top deviations:")
    for idx in worst_idx:
        print(f"    {anim_labels[idx]:50s}  ref={ref_abl[idx]:.4f}  recon={recon_abl[idx]:.4f}  ratio={abl_ratio[idx]:.4f}")

    # ── 9. Scale check (orientation-invariant) ──
    # Compare sorted bounding box extents — rotation may swap X/Z
    ref_extent = ref_skinned[0].max(axis=0) - ref_skinned[0].min(axis=0)
    recon_extent = recon_skinned[0].max(axis=0) - recon_skinned[0].min(axis=0)
    ref_diag = np.linalg.norm(ref_extent)
    recon_diag = np.linalg.norm(recon_extent)

    ref_sorted = np.sort(ref_extent)[::-1]
    recon_sorted = np.sort(recon_extent)[::-1]
    sorted_ratio = recon_sorted / np.maximum(ref_sorted, 1e-10)

    print(f"\n{'='*60}")
    print("SCALE CHECK (bounding box at frame 0)")
    print(f"{'='*60}")
    print(f"  Reference extent:  X={ref_extent[0]:.4f}  Y={ref_extent[1]:.4f}  Z={ref_extent[2]:.4f}  diag={ref_diag:.4f}")
    print(f"  Recon extent:      X={recon_extent[0]:.4f}  Y={recon_extent[1]:.4f}  Z={recon_extent[2]:.4f}  diag={recon_diag:.4f}")
    print(f"  Diagonal ratio:    {recon_diag / ref_diag:.6f}")
    print(f"  Sorted extents (large→small):")
    print(f"    Reference: {ref_sorted[0]:.4f}  {ref_sorted[1]:.4f}  {ref_sorted[2]:.4f}")
    print(f"    Recon:     {recon_sorted[0]:.4f}  {recon_sorted[1]:.4f}  {recon_sorted[2]:.4f}")
    print(f"    Ratio:     {sorted_ratio[0]:.4f}  {sorted_ratio[1]:.4f}  {sorted_ratio[2]:.4f}")

    # Average over all frames
    ref_extents_all = ref_skinned.max(axis=1) - ref_skinned.min(axis=1)   # (T, 3)
    recon_extents_all = recon_skinned.max(axis=1) - recon_skinned.min(axis=1)
    ref_diag_all = np.linalg.norm(ref_extents_all, axis=1)
    recon_diag_all = np.linalg.norm(recon_extents_all, axis=1)
    diag_ratio_all = recon_diag_all / np.maximum(ref_diag_all, 1e-10)
    print(f"  Diagonal ratio (all frames): mean={diag_ratio_all.mean():.6f}  std={diag_ratio_all.std():.6f}")

    # ── 10. Raw per-vertex error ──
    diff = recon_skinned - ref_skinned
    per_vertex_l2 = np.sqrt((diff ** 2).sum(axis=-1))

    print(f"\n{'='*60}")
    print("RAW PER-VERTEX ERROR (includes global position/orientation offset)")
    print(f"{'='*60}")
    print(f"  Mean:   {per_vertex_l2.mean():.6f}")
    print(f"  Median: {np.median(per_vertex_l2):.6f}")
    print(f"  P95:    {np.percentile(per_vertex_l2, 95):.6f}")
    print(f"  Max:    {per_vertex_l2.max():.6f}")

    # ── 11. Centroid-aligned error (removes global translation) ──
    ref_centroid = ref_skinned.mean(axis=1, keepdims=True)       # (T, 1, 3)
    recon_centroid = recon_skinned.mean(axis=1, keepdims=True)
    ref_centered = ref_skinned - ref_centroid
    recon_centered = recon_skinned - recon_centroid

    diff_centered = recon_centered - ref_centered
    pv_centered = np.sqrt((diff_centered ** 2).sum(axis=-1))

    print(f"\n{'='*60}")
    print("CENTROID-ALIGNED PER-VERTEX ERROR (removes global translation)")
    print(f"{'='*60}")
    print(f"  Mean:   {pv_centered.mean():.6f}  ({pv_centered.mean()/ref_diag*100:.2f}% of diag)")
    print(f"  Median: {np.median(pv_centered):.6f}")
    print(f"  P95:    {np.percentile(pv_centered, 95):.6f}")
    print(f"  Max:    {pv_centered.max():.6f}")

    per_frame_centered = pv_centered.mean(axis=1)
    print(f"  Per-frame mean: min={per_frame_centered.min():.6f}  max={per_frame_centered.max():.6f}")

    # ── 12. Rigid-aligned error (removes translation + rotation only) ──
    # Per-frame Kabsch alignment to isolate pose error from global rigid
    # transform differences. NO scale correction — we want to detect
    # scale mismatches, not hide them.
    pv_rigid = np.zeros_like(pv_centered)
    for fi in range(len(frame_indices)):
        A = ref_centered[fi]       # (V, 3)
        B = recon_centered[fi]     # (V, 3)
        # Kabsch: find R that minimizes ||A - B @ R||
        H = A.T @ B
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(U @ Vt)
        D = np.diag([1, 1, d])  # ensure proper rotation (det=+1)
        R = U @ D @ Vt
        B_aligned = B @ R.T
        pv_rigid[fi] = np.sqrt(((A - B_aligned) ** 2).sum(axis=-1))

    print(f"\n{'='*60}")
    print("RIGID-ALIGNED ERROR (removes translation + rotation, keeps scale)")
    print(f"{'='*60}")
    print(f"  Mean:   {pv_rigid.mean():.6f}  ({pv_rigid.mean()/ref_diag*100:.2f}% of diag)")
    print(f"  Median: {np.median(pv_rigid):.6f}")
    print(f"  P95:    {np.percentile(pv_rigid, 95):.6f}")
    print(f"  Max:    {pv_rigid.max():.6f}")

    per_frame_rigid = pv_rigid.mean(axis=1)
    print(f"  Per-frame mean: min={per_frame_rigid.min():.6f}  max={per_frame_rigid.max():.6f}")

    # ── 13. Centroid displacement ──
    centroid_diff = np.sqrt(((ref_centroid.squeeze(1) - recon_centroid.squeeze(1)) ** 2).sum(axis=-1))
    print(f"\n{'='*60}")
    print("GLOBAL CENTROID DISPLACEMENT")
    print(f"{'='*60}")
    print(f"  Mean:   {centroid_diff.mean():.6f}")
    print(f"  Max:    {centroid_diff.max():.6f}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    # Our retarget only applies rotations (no bone scaling/translation except root),
    # so rest-pose bone lengths are preserved by construction. The animated bone
    # length ratio shows how close the poses are.
    abl_mean = abl_ratio.mean()
    abl_scale_ok = abs(abl_mean - 1.0) < 0.05
    print(f"  Scale equivalent:      {'YES' if abl_scale_ok else 'NO'} (anim bone length ratio = {abl_mean:.4f} ± {abl_ratio.std():.4f})")
    print(f"  Rest skeleton length:  {total_skeleton_length:.4f} (identical for ref and recon)")
    print(f"  Centroid-aligned err:  {pv_centered.mean():.4f} ({pv_centered.mean()/ref_diag*100:.1f}% of diag)")
    print(f"  Rigid-aligned err:     {pv_rigid.mean():.4f} ({pv_rigid.mean()/ref_diag*100:.1f}% of diag)")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
