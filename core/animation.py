"""glTF animation injection via trimesh postprocessor hooks.

Animates joint spheres (translation) AND bone cylinders (translation +
rotation + scale) so that bones track the moving joints every frame.
"""

import numpy as np


def _pack_float32_buffer(arr):
    """Pack a numpy float32 array to bytes, padded to 4-byte alignment."""
    data = arr.astype(np.float32).tobytes()
    pad = (4 - len(data) % 4) % 4
    return data + b"\x00" * pad


# ---------- bone TRS helpers ----------

def _z_to_dir_quats(dirs):
    """Quaternions that rotate +Z to each direction (vectorised).

    Parameters
    ----------
    dirs : (N, 3) float  — *normalised* direction vectors

    Returns
    -------
    (N, 4) float — quaternions in **xyzw** (glTF convention)
    """
    N = dirs.shape[0]
    z = np.array([0, 0, 1], dtype=dirs.dtype)
    quats = np.zeros((N, 4), dtype=dirs.dtype)
    quats[:, 3] = 1.0  # identity default

    dots = dirs @ z  # (N,)

    # anti-parallel → 180° around X
    anti = dots < -0.9999
    quats[anti] = [1, 0, 0, 0]

    # general case
    gen = (~anti) & (dots < 0.9999)
    if gen.any():
        d = dirs[gen]
        cross = np.cross(z, d)
        axes = cross / np.linalg.norm(cross, axis=-1, keepdims=True)
        halves = np.arccos(np.clip(dots[gen], -1, 1)) / 2
        sin_h = np.sin(halves)
        quats[gen, 0] = axes[:, 0] * sin_h
        quats[gen, 1] = axes[:, 1] * sin_h
        quats[gen, 2] = axes[:, 2] * sin_h
        quats[gen, 3] = np.cos(halves)

    return quats


def _compute_bone_trs(positions, bone_connections):
    """Per-frame TRS for every bone (unit-Z-cylinder convention).

    Returns
    -------
    trans  : (T, B, 3) float32 — midpoints
    rots   : (T, B, 4) float32 — xyzw quaternions
    scales : (T, B, 3) float32 — [1, 1, length]
    """
    T = positions.shape[0]
    B = len(bone_connections)
    parents = [p for p, _ in bone_connections]
    children = [c for _, c in bone_connections]

    starts = positions[:, parents, :]  # (T, B, 3)
    ends = positions[:, children, :]

    trans = ((starts + ends) / 2).astype(np.float32)

    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=-1)  # (T, B)
    safe = np.maximum(lengths, 1e-10)
    dir_norm = directions / safe[..., np.newaxis]

    rots = _z_to_dir_quats(dir_norm.reshape(-1, 3).astype(np.float64))
    rots = rots.reshape(T, B, 4).astype(np.float32)

    scales = np.ones((T, B, 3), dtype=np.float32)
    scales[:, :, 2] = lengths.astype(np.float32)

    return trans, rots, scales


# ---------- matrix → TRS decomposition ----------

def _decompose_trs(matrix_col_major):
    """Decompose a column-major flat 16-element matrix into glTF TRS."""
    m = np.array(matrix_col_major, dtype=np.float64).reshape(4, 4, order="F")
    t = m[:3, 3].tolist()

    sx = np.linalg.norm(m[:3, 0])
    sy = np.linalg.norm(m[:3, 1])
    sz = np.linalg.norm(m[:3, 2])
    s = [float(sx), float(sy), float(sz)]

    R = np.column_stack([
        m[:3, 0] / max(sx, 1e-10),
        m[:3, 1] / max(sy, 1e-10),
        m[:3, 2] / max(sz, 1e-10),
    ])
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
        s[0] *= -1

    # Shepperd's method → xyzw
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        r = np.sqrt(1 + tr)
        q = 0.5 / r
        x, y, z, w = (R[2,1]-R[1,2])*q, (R[0,2]-R[2,0])*q, (R[1,0]-R[0,1])*q, 0.5*r
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        r = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
        q = 0.5 / r
        x, y, z, w = 0.5*r, (R[0,1]+R[1,0])*q, (R[0,2]+R[2,0])*q, (R[2,1]-R[1,2])*q
    elif R[1, 1] > R[2, 2]:
        r = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
        q = 0.5 / r
        x, y, z, w = (R[0,1]+R[1,0])*q, 0.5*r, (R[1,2]+R[2,1])*q, (R[0,2]-R[2,0])*q
    else:
        r = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
        q = 0.5 / r
        x, y, z, w = (R[0,2]+R[2,0])*q, (R[1,2]+R[2,1])*q, 0.5*r, (R[1,0]-R[0,1])*q

    rot = [float(x), float(y), float(z), float(w)]
    return t, rot, s


# ---------- public API ----------

def make_postprocessors(positions, bone_connections, fps=30, num_joints=30):
    """
    Create buffer_postprocessor and tree_postprocessor for glTF animation.

    Animates **both** joints (translation) and bones (translation + rotation +
    scale), so the skeleton moves as a connected whole.
    """
    T = positions.shape[0]
    num_bones = len(bone_connections)
    timestamps = np.arange(T, dtype=np.float32) / fps

    # --- precompute all byte buffers ---
    time_bytes = _pack_float32_buffer(timestamps)

    joint_bufs = [_pack_float32_buffer(positions[:, j, :]) for j in range(num_joints)]

    bone_trans, bone_rots, bone_scales = _compute_bone_trs(positions, bone_connections)
    bone_t_bufs = [_pack_float32_buffer(bone_trans[:, b, :]) for b in range(num_bones)]
    bone_r_bufs = [_pack_float32_buffer(bone_rots[:, b, :]) for b in range(num_bones)]
    bone_s_bufs = [_pack_float32_buffer(bone_scales[:, b, :]) for b in range(num_bones)]

    # total animation accessors: 1 (time) + joints + 3*bones
    num_anim_acc = 1 + num_joints + 3 * num_bones

    def buffer_postprocessor(buffer_items, tree):
        acc = tree["accessors"]
        bv = len(buffer_items)  # next bufferView index

        def _add(key, data, accessor):
            nonlocal bv
            buffer_items[key] = data
            accessor["bufferView"] = bv
            acc[key] = accessor
            bv += 1

        # 1) timestamp
        _add("anim_time", time_bytes, {
            "componentType": 5126, "count": T, "type": "SCALAR",
            "min": [float(timestamps[0])], "max": [float(timestamps[-1])],
        })

        # 2) joint translations
        for j in range(num_joints):
            d = positions[:, j, :]
            _add(f"anim_jt_{j}", joint_bufs[j], {
                "componentType": 5126, "count": T, "type": "VEC3",
                "min": d.min(0).tolist(), "max": d.max(0).tolist(),
            })

        # 3) bone translations
        for b in range(num_bones):
            d = bone_trans[:, b, :]
            _add(f"anim_bt_{b}", bone_t_bufs[b], {
                "componentType": 5126, "count": T, "type": "VEC3",
                "min": d.min(0).tolist(), "max": d.max(0).tolist(),
            })

        # 4) bone rotations
        for b in range(num_bones):
            d = bone_rots[:, b, :]
            _add(f"anim_br_{b}", bone_r_bufs[b], {
                "componentType": 5126, "count": T, "type": "VEC4",
                "min": d.min(0).tolist(), "max": d.max(0).tolist(),
            })

        # 5) bone scales
        for b in range(num_bones):
            d = bone_scales[:, b, :]
            _add(f"anim_bs_{b}", bone_s_bufs[b], {
                "componentType": 5126, "count": T, "type": "VEC3",
                "min": d.min(0).tolist(), "max": d.max(0).tolist(),
            })

    def tree_postprocessor(tree):
        nodes = tree["nodes"]
        accessors = tree["accessors"]
        total_acc = len(accessors)

        # Accessor layout (appended at the end, in order):
        #   time | jt_0..jt_29 | bt_0..bt_28 | br_0..br_28 | bs_0..bs_28
        time_idx = total_acc - num_anim_acc
        jt_start = time_idx + 1
        bt_start = jt_start + num_joints
        br_start = bt_start + num_bones
        bs_start = br_start + num_bones

        # --- find node indices ---
        joint_nodes = {}   # joint_index → node_index
        bone_nodes = {}    # bone_index → node_index
        bone_conn_map = {(p, c): i for i, (p, c) in enumerate(bone_connections)}

        for ni, node in enumerate(nodes):
            name = node.get("name", "")
            if name.startswith("joint_") and "_geom" not in name:
                parts = name.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    joint_nodes[int(parts[1])] = ni
            elif name.startswith("bone_") and "_geom" not in name:
                parts = name.split("_")
                if len(parts) == 3:
                    try:
                        key = (int(parts[1]), int(parts[2]))
                        if key in bone_conn_map:
                            bone_nodes[bone_conn_map[key]] = ni
                    except ValueError:
                        pass

        # --- convert animated nodes from matrix → TRS ---
        for j, ni in joint_nodes.items():
            node = nodes[ni]
            if "matrix" in node:
                mat = np.array(node["matrix"]).reshape(4, 4, order="F")
                del node["matrix"]
                node["translation"] = mat[:3, 3].tolist()
                node["rotation"] = [0, 0, 0, 1]
                node["scale"] = [1, 1, 1]

        for b, ni in bone_nodes.items():
            node = nodes[ni]
            if "matrix" in node:
                t, r, s = _decompose_trs(node["matrix"])
                del node["matrix"]
                node["translation"] = t
                node["rotation"] = r
                node["scale"] = s

        # --- build samplers & channels ---
        samplers = []
        channels = []

        def _channel(output_acc, node_idx, path):
            si = len(samplers)
            samplers.append({
                "input": time_idx, "interpolation": "LINEAR",
                "output": output_acc,
            })
            channels.append({
                "sampler": si,
                "target": {"node": node_idx, "path": path},
            })

        # joints — translation only
        for j in range(num_joints):
            if j in joint_nodes:
                _channel(jt_start + j, joint_nodes[j], "translation")

        # bones — translation + rotation + scale
        for b in range(num_bones):
            if b in bone_nodes:
                ni = bone_nodes[b]
                _channel(bt_start + b, ni, "translation")
                _channel(br_start + b, ni, "rotation")
                _channel(bs_start + b, ni, "scale")

        tree["animations"] = [{
            "name": "joint_animation",
            "samplers": samplers,
            "channels": channels,
        }]

    return buffer_postprocessor, tree_postprocessor
