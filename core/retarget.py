"""Retarget AniMo joint positions → GLB local rotations.

Pipeline (delta-from-rest approach):
  1. NO coordinate undo (AniMo new_joints data is already Y-up = GLB space)
  2. Compute root heading delta from forward direction comparison
  3. Compute per-bone delta: qbetween(GLB_rest_dir, actual_dir)
  4. Apply delta: world_anim = delta * rest_world
  5. Compute local rotations from world rotations via GLB hierarchy
  6. Convert wxyz → xyzw for glTF
"""

import numpy as np

from core.joint_config import KINEMATIC_CHAIN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Face joint indices (AniMo reordered ordering): r_hip, l_hip, sdr_r, sdr_l
FACE_JOINT_IDX = [25, 20, 8, 7]

# Inverse permutation: AniMo reordered → original Blender/GLB ordering.
# Forward (cell 7): dst[i] gets src[i]
#   dst = [9,10,11,...,29]
#   src = [10,9,11,12,13,15,14,16,17,19,18,20,24,21,23,22,25,29,26,28,27]
# INVERSE_PERM[animo_idx] = original_idx (which AniMo slot holds original_idx)
INVERSE_PERM = [
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    10, 9, 11, 12, 13, 15, 14, 16, 17, 19, 18, 20,
    24, 21, 23, 22,
    25,
    29, 26, 28, 27,
]


# ---------------------------------------------------------------------------
# Pure-numpy quaternion helpers (wxyz convention, w at index 0)
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vectors along last axis, handling zero-length."""
    norms = np.sqrt((v ** 2).sum(axis=-1, keepdims=True))
    norms = np.maximum(norms, 1e-10)
    return v / norms


def qmul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Multiply quaternions q * r.  Shape (..., 4), wxyz convention."""
    w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ], axis=-1)


def qinv(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (inverse for unit quaternions). wxyz."""
    qi = q.copy()
    qi[..., 1:] *= -1
    return qi


def qbetween(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """Find quaternion that rotates v0 to v1. Returns wxyz."""
    cross = np.cross(v0, v1)
    dot = (v0 * v1).sum(axis=-1, keepdims=True)
    w = np.sqrt((v0**2).sum(axis=-1, keepdims=True) *
                (v1**2).sum(axis=-1, keepdims=True)) + dot
    q = np.concatenate([w, cross], axis=-1)
    return _normalize(q)


def qslerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between q0 and q1. wxyz, batched (T, 4)."""
    dot = (q0 * q1).sum(axis=-1, keepdims=True)
    # Ensure shortest path
    sign = np.where(dot < 0, -1.0, 1.0)
    q1 = q1 * sign
    dot = dot * sign
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    # Fallback to lerp when angle is very small
    small = (sin_theta.squeeze(-1) < 1e-6)
    s0 = np.sin((1 - t) * theta) / np.maximum(sin_theta, 1e-10)
    s1 = np.sin(t * theta) / np.maximum(sin_theta, 1e-10)
    result = s0 * q0 + s1 * q1
    # lerp fallback
    if np.any(small):
        lerp = (1 - t) * q0 + t * q1
        result[small] = lerp[small]
    return _normalize(result)


def _twist_decompose(q: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Extract the twist component of quaternion q around axis. wxyz, batched (T,4).

    Returns the twist quaternion (rotation around axis only).
    """
    # Project imaginary part onto axis
    # axis should be (3,) unit vector
    proj = (q[..., 1:] * axis).sum(axis=-1, keepdims=True) * axis  # (T, 3)
    twist = np.concatenate([q[..., :1], proj], axis=-1)  # (T, 4)
    norms = np.sqrt((twist ** 2).sum(axis=-1, keepdims=True))
    norms = np.maximum(norms, 1e-10)
    twist = twist / norms
    # Ensure w > 0 for consistent sign
    sign = np.where(twist[..., :1] < 0, -1.0, 1.0)
    return twist * sign


def _mat4_to_quat_wxyz(mat4: np.ndarray) -> np.ndarray:
    """Extract rotation quaternion (wxyz) from a 4x4 matrix."""
    m = mat4[:3, :3].copy()
    # Remove scale
    for i in range(3):
        n = np.linalg.norm(m[:, i])
        if n > 1e-10:
            m[:, i] /= n
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _build_animo_to_glb_chain(glb_data) -> dict:
    """Map AniMo chain index → GLB joint index.

    AniMo chain uses reordered indices. We go through INVERSE_PERM
    to get original ordering, then use animo_to_glb (which is keyed
    by original index).
    """
    mapping = {}
    for animo_idx in range(30):
        original_idx = INVERSE_PERM[animo_idx]
        if original_idx in glb_data.animo_to_glb:
            mapping[animo_idx] = glb_data.animo_to_glb[original_idx]
    return mapping


def _build_glb_topo_order(glb_data) -> list:
    """Topological order for GLB joints (parents before children)."""
    n = len(glb_data.joint_names)
    parent_indices = glb_data.parent_indices
    visited = set()
    order = []

    def _visit(si):
        if si in visited:
            return
        pi = parent_indices[si]
        if pi >= 0 and pi not in visited:
            _visit(pi)
        visited.add(si)
        order.append(si)

    for si in range(n):
        _visit(si)
    return order


# ---------------------------------------------------------------------------
# Compute forward direction from hip/shoulder joints
# ---------------------------------------------------------------------------

def _compute_forward(positions, face_idx):
    """Compute forward direction from hip + shoulder positions.

    Args:
        positions: (..., 3) positions for [r_hip, l_hip, sdr_r, sdr_l]
                   or indexable array
        face_idx: [r_hip, l_hip, sdr_r, sdr_l] indices

    Returns:
        forward: (..., 3) normalized forward direction
    """
    r_hip, l_hip, sdr_r, sdr_l = face_idx
    across = ((positions[:, r_hip] - positions[:, l_hip]) +
              (positions[:, sdr_r] - positions[:, sdr_l]))
    across = _normalize(across)
    forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    return _normalize(forward)


def _compute_forward_single(positions, face_idx):
    """Same as _compute_forward but for a single frame (no batch dim)."""
    r_hip, l_hip, sdr_r, sdr_l = face_idx
    across = ((positions[r_hip] - positions[l_hip]) +
              (positions[sdr_r] - positions[sdr_l]))
    n = np.linalg.norm(across)
    if n > 1e-10:
        across = across / n
    forward = np.cross([0, 1, 0], across)
    n = np.linalg.norm(forward)
    if n > 1e-10:
        forward = forward / n
    return forward


# ---------------------------------------------------------------------------
# Main retarget pipeline
# ---------------------------------------------------------------------------

def retarget(positions_animo: np.ndarray, glb_data, tiered: bool = True) -> tuple:
    """Full retargeting: AniMo positions → GLB local rotations.

    Uses "delta from rest" approach:
      delta[j] = qbetween(GLB_rest_bone_dir, actual_bone_dir)
      world_anim[j] = delta[j] * rest_world[j]
      local[j] = qinv(world[parent]) * world[j]

    Args:
        positions_animo: (T, 30, 3) AniMo joint positions (Y-up, AniMo ordering)
        glb_data: GLBData from glb_parser
        tiered: if True, compute additional deltas for non-tracked joints
                (slerp, twist, tail decay). If False, only the 30 tracked joints.

    Returns:
        local_rotations_xyzw: (T, N, 4) for all GLB joints
        root_translations: (T, 3)
        tracked_glb_indices: set of GLB joint indices that have animation
    """
    T = positions_animo.shape[0]
    positions = positions_animo.astype(np.float64)
    n_glb = len(glb_data.joint_names)

    # --- Mappings ---
    chain_to_glb = _build_animo_to_glb_chain(glb_data)
    topo_order = _build_glb_topo_order(glb_data)
    parent_indices = glb_data.parent_indices

    # GLB rest-pose world positions and rotations
    glb_rest_pos = glb_data.rest_world_matrices[:, :3, 3]  # (N, 3)
    glb_rest_rot = np.array([_mat4_to_quat_wxyz(m)
                             for m in glb_data.rest_world_matrices])  # (N, 4) wxyz

    # --- Step 1: Root heading delta ---
    # Compute GLB rest forward from rest-pose hip/shoulder positions
    # Map face joint indices through chain_to_glb to get GLB positions
    face_glb = [chain_to_glb[fi] for fi in FACE_JOINT_IDX]
    rest_forward = _compute_forward_single(glb_rest_pos, face_glb)

    # Compute actual forward per frame from AniMo positions
    actual_forward = _compute_forward(positions, FACE_JOINT_IDX)

    # Root delta = rotation from rest forward to actual forward
    rest_fwd_tiled = np.tile(rest_forward, (T, 1))
    root_delta = qbetween(rest_fwd_tiled, actual_forward)  # (T, 4) wxyz

    # Apply reference animation heading offset.
    # AniMo positions are heading-normalized (animal faces +Z at frame 0).
    # The reference GLB animation may have the animal at a different heading.
    # Extract that heading offset and apply it so the output matches the
    # reference orientation.
    if glb_data.ref_root_rot_frame0 is not None:
        ref_xyzw = glb_data.ref_root_rot_frame0
        ref_wxyz = np.array([ref_xyzw[3], ref_xyzw[0], ref_xyzw[1], ref_xyzw[2]],
                            dtype=np.float64)
        rest_rot_wxyz = glb_rest_rot[chain_to_glb[0]]  # root rest world rotation
        # heading_offset = ref_frame0 * inv(rest)
        heading_offset = qmul(ref_wxyz.reshape(1, 4),
                              qinv(rest_rot_wxyz.reshape(1, 4)))  # (1, 4)
        root_delta = qmul(np.tile(heading_offset, (T, 1)), root_delta)

    # --- Step 2: Per-bone direction deltas ---
    # For each tracked joint, find its nearest tracked ancestor in the
    # GLB *physical* hierarchy and compute the bone direction from there.
    # This avoids issues where the AniMo kinematic chain order differs
    # from the GLB skeleton topology (e.g. twist joints).
    #
    # delta[glb_idx] = (T, 4) wxyz quaternion
    deltas = {}  # glb_idx → (T, 4)

    root_glb = chain_to_glb[0]
    deltas[root_glb] = root_delta

    # Reverse mapping: glb_idx → animo_chain_idx (for position lookup)
    glb_to_chain = {v: k for k, v in chain_to_glb.items()}
    tracked_glb_set = set(chain_to_glb.values())

    def _find_tracked_ancestor(glb_idx):
        """Walk up GLB hierarchy to find nearest tracked ancestor."""
        pi = parent_indices[glb_idx]
        while pi >= 0:
            if pi in tracked_glb_set:
                return pi
            pi = parent_indices[pi]
        return None

    for animo_idx, glb_idx in chain_to_glb.items():
        if glb_idx == root_glb:
            continue  # root already handled

        # Find nearest tracked ancestor in GLB hierarchy
        ancestor_glb = _find_tracked_ancestor(glb_idx)
        if ancestor_glb is None:
            # No tracked ancestor — use root delta
            deltas[glb_idx] = root_delta.copy()
            continue

        ancestor_animo = glb_to_chain[ancestor_glb]

        # Actual bone direction from AniMo positions (ancestor → this joint)
        actual_dir = _normalize(positions[:, animo_idx] - positions[:, ancestor_animo])

        # GLB rest bone direction (same pair)
        rest_vec = glb_rest_pos[glb_idx] - glb_rest_pos[ancestor_glb]
        rest_len = np.linalg.norm(rest_vec)
        if rest_len < 1e-10:
            deltas[glb_idx] = root_delta.copy()
            continue
        rest_dir = rest_vec / rest_len
        rest_dir_tiled = np.tile(rest_dir, (T, 1))

        # Delta: rotation from rest direction to actual direction
        deltas[glb_idx] = qbetween(rest_dir_tiled, actual_dir)

    # --- Step 2a: Facial leaf-joint damping ---
    # Jaw and eyes have very short bones from head (1-5cm for jaw, 4-20cm
    # for eyes depending on species). Small AniMo positional errors become
    # large angular swings. Blend their delta toward the head's (ancestor)
    # delta, proportional to how short the bone is.
    _DAMP_REF_LEN = 0.10  # bones >= 10cm keep full delta
    for animo_idx, glb_idx in chain_to_glb.items():
        if glb_idx == root_glb or glb_idx not in deltas:
            continue
        name = glb_data.joint_names[glb_idx]
        if 'jaw' not in name and 'eye' not in name:
            continue
        ancestor_glb = _find_tracked_ancestor(glb_idx)
        if ancestor_glb is None or ancestor_glb not in deltas:
            continue
        bone_len = np.linalg.norm(glb_rest_pos[glb_idx] - glb_rest_pos[ancestor_glb])
        factor = min(1.0, bone_len / _DAMP_REF_LEN)
        if factor < 1.0:
            deltas[glb_idx] = qslerp(deltas[ancestor_glb], deltas[glb_idx], factor)

    # --- Step 2b: Tiered non-tracked joint deltas ---
    if tiered:
        # Build children map for hierarchy traversal
        children_map = {}
        for si in range(n_glb):
            pi = parent_indices[si]
            if pi >= 0:
                children_map.setdefault(pi, []).append(si)

        # Tier 1: Slerp interpolation for sandwiched chains.
        # For each tracked joint, walk UP the hierarchy to find the nearest
        # tracked ancestor. Non-tracked joints on that path form a "sandwiched
        # chain" and get a slerp-interpolated delta between the two endpoints.
        #   spine:     root[T] → [hips, spine1, spine2, spine3] → chest[T]
        #   neck:      neck1[T] → [neck2] → head[T]
        #   horselink: legLwr[T] → [horselink] → foot[T]
        for descendant in list(tracked_glb_set):
            chain = []
            current = parent_indices[descendant]
            while current >= 0 and current not in tracked_glb_set:
                chain.append(current)
                current = parent_indices[current]
            if current < 0 or not chain:
                continue
            ancestor = current
            if ancestor not in deltas or descendant not in deltas:
                continue
            chain.reverse()  # ancestor → descendant order
            n_chain = len(chain)
            for i, joint_idx in enumerate(chain):
                t = (i + 1) / (n_chain + 1)
                deltas[joint_idx] = qslerp(deltas[ancestor], deltas[descendant], t)

        # Tier 2: Twist decomposition for twist joints.
        # HalfTwist joints get 50% of ancestor's twist component around the
        # bone axis; non-tracked AllTwist joints get 100%.
        for si, name in enumerate(glb_data.joint_names):
            if si in deltas:
                continue  # already handled (tracked or sandwiched)
            if 'Twist' not in name:
                continue
            if 'HalfTwist' in name:
                fraction = 0.5
            elif 'AllTwist' in name:
                fraction = 1.0
            else:
                continue
            # Find nearest ancestor with a delta
            pi = parent_indices[si]
            while pi >= 0 and pi not in deltas:
                pi = parent_indices[pi]
            if pi < 0 or pi not in deltas:
                continue
            # Bone axis: rest-pose direction from ancestor to this joint
            bone_vec = glb_rest_pos[si] - glb_rest_pos[pi]
            bone_len = np.linalg.norm(bone_vec)
            if bone_len < 1e-10:
                continue
            bone_axis = bone_vec / bone_len
            twist_q = _twist_decompose(deltas[pi], bone_axis)
            identity = np.zeros((T, 4), dtype=np.float64)
            identity[:, 0] = 1.0
            deltas[si] = qslerp(identity, twist_q, fraction)

        # Tier 3: Exponential decay for tail extensions beyond tracked tail joints.
        # Process in topological order so each successive tail joint decays from
        # its (already-decayed) parent → natural exponential falloff.
        _TAIL_DECAY = 0.7
        for si in topo_order:
            if si in deltas:
                continue
            name = glb_data.joint_names[si]
            if 'tail' not in name.lower():
                continue
            pi = parent_indices[si]
            if pi < 0 or pi not in deltas:
                continue
            if 'tail' not in glb_data.joint_names[pi].lower():
                continue
            identity = np.zeros((T, 4), dtype=np.float64)
            identity[:, 0] = 1.0
            deltas[si] = qslerp(identity, deltas[pi], _TAIL_DECAY)

    tracked_glb = set(deltas.keys())

    # --- Step 3: Apply deltas → world rotations ---
    # world_anim[j] = delta[j] * rest_world[j]
    # For non-tracked: inherit nearest tracked ancestor's delta
    world_anim = np.zeros((T, n_glb, 4), dtype=np.float64)
    world_anim[:, :, 0] = 1.0  # identity default

    for si in topo_order:
        if si in deltas:
            # Tracked joint: apply its own delta
            d = deltas[si]
            rest_q = np.tile(glb_rest_rot[si], (T, 1))
            world_anim[:, si] = qmul(d, rest_q)
        else:
            pi = parent_indices[si]
            if pi < 0:
                # Untracked root → use rest rotation
                world_anim[:, si] = np.tile(glb_rest_rot[si], (T, 1))
            else:
                # Non-tracked child: world = parent_world * rest_local
                rest_local_xyzw = glb_data.rest_local_trs[si][1]
                rest_local_wxyz = np.array([rest_local_xyzw[3], rest_local_xyzw[0],
                                            rest_local_xyzw[1], rest_local_xyzw[2]])
                rest_q = np.tile(rest_local_wxyz, (T, 1))
                world_anim[:, si] = qmul(world_anim[:, pi], rest_q)

    # --- Step 4: Compute local rotations from world ---
    local_rots = np.zeros((T, n_glb, 4), dtype=np.float64)
    local_rots[:, :, 0] = 1.0

    for si in topo_order:
        pi = parent_indices[si]
        if pi < 0:
            local_rots[:, si] = world_anim[:, si]
        else:
            local_rots[:, si] = qmul(qinv(world_anim[:, pi]), world_anim[:, si])

    # --- Step 5: Convert wxyz → xyzw for glTF ---
    local_rots_xyzw = np.concatenate([local_rots[..., 1:], local_rots[..., :1]], axis=-1)

    # --- Step 6: Root translation ---
    # Use AniMo root positions directly so the mesh root matches the
    # skeleton visualization (npy2glb_skeleton) exactly.
    root_translations = positions[:, 0, :].copy()  # (T, 3)

    return local_rots_xyzw, root_translations, tracked_glb
