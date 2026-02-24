"""Parse a reference GLB file: hierarchy, rest-pose TRS, IBMs, skinning, AniMo joint mapping."""

import json
import struct
from dataclasses import dataclass, field

import numpy as np

from core.joint_config import JOINT_NAMES

# ---------------------------------------------------------------------------
# GLB constants
# ---------------------------------------------------------------------------
GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

# glTF component type → (struct fmt, byte size)
_COMP = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}

# glTF type → element count
_TYPE_COUNT = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


@dataclass
class GLBData:
    json_tree: dict
    bin_buffer: bytes
    joint_names: list  # 143 joint names ordered by skin.joints
    parent_indices: list  # parent index per joint (-1 for root)
    rest_local_trs: list  # [(t, r_xyzw, s), ...] per joint
    rest_world_matrices: np.ndarray  # (N, 4, 4)
    inverse_bind_matrices: np.ndarray  # (N, 4, 4)
    animo_to_glb: dict = field(default_factory=dict)  # AniMo idx → GLB joint idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_accessor(gltf: dict, buf: bytes, acc_idx: int) -> np.ndarray:
    """Read a glTF accessor into a numpy array."""
    acc = gltf["accessors"][acc_idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    comp_type = acc["componentType"]
    fmt, bsz = _COMP[comp_type]
    count = acc["count"]
    n_components = _TYPE_COUNT[acc["type"]]
    byte_offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    byte_stride = bv.get("byteStride", 0)

    if byte_stride and byte_stride != bsz * n_components:
        # Strided access
        out = np.empty((count, n_components), dtype=np.float64)
        for i in range(count):
            off = byte_offset + i * byte_stride
            vals = struct.unpack_from(f"<{n_components}{fmt}", buf, off)
            out[i] = vals
        return out
    else:
        total = count * n_components
        data = struct.unpack_from(f"<{total}{fmt}", buf, byte_offset)
        arr = np.array(data, dtype=np.float64).reshape(count, n_components)
        return arr


def _trs_to_mat4(t, r_xyzw, s) -> np.ndarray:
    """Build a 4x4 matrix from translation, rotation (xyzw), scale."""
    x, y, z, w = r_xyzw
    m = np.eye(4, dtype=np.float64)
    # Rotation from quaternion
    m[0, 0] = (1 - 2 * (y * y + z * z)) * s[0]
    m[0, 1] = (2 * (x * y - z * w)) * s[1]
    m[0, 2] = (2 * (x * z + y * w)) * s[2]
    m[1, 0] = (2 * (x * y + z * w)) * s[0]
    m[1, 1] = (1 - 2 * (x * x + z * z)) * s[1]
    m[1, 2] = (2 * (y * z - x * w)) * s[2]
    m[2, 0] = (2 * (x * z - y * w)) * s[0]
    m[2, 1] = (2 * (y * z + x * w)) * s[1]
    m[2, 2] = (1 - 2 * (x * x + y * y)) * s[2]
    m[0, 3] = t[0]
    m[1, 3] = t[1]
    m[2, 3] = t[2]
    return m


def _node_local_trs(node: dict):
    """Extract (translation, rotation_xyzw, scale) from a glTF node."""
    if "matrix" in node:
        mat = np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # col-major→row-major
        t = mat[:3, 3].tolist()
        # Extract rotation via polar decomposition (simplified: assume no shear)
        sx = np.linalg.norm(mat[:3, 0])
        sy = np.linalg.norm(mat[:3, 1])
        sz = np.linalg.norm(mat[:3, 2])
        s = [sx, sy, sz]
        rot = mat[:3, :3].copy()
        rot[:, 0] /= sx
        rot[:, 1] /= sy
        rot[:, 2] /= sz
        # Rotation matrix → quaternion (xyzw)
        r_xyzw = _mat3_to_quat_xyzw(rot)
        return (t, r_xyzw, s)
    else:
        t = node.get("translation", [0.0, 0.0, 0.0])
        r = node.get("rotation", [0.0, 0.0, 0.0, 1.0])  # glTF default = identity xyzw
        s = node.get("scale", [1.0, 1.0, 1.0])
        return (list(t), list(r), list(s))


def _mat3_to_quat_xyzw(m: np.ndarray) -> list:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
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
    return [x, y, z, w]


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_glb(path: str) -> GLBData:
    """Parse a reference GLB and return all skeleton/skinning data."""
    with open(path, "rb") as f:
        raw = f.read()

    # --- GLB header ---
    magic, version, total_len = struct.unpack_from("<III", raw, 0)
    assert magic == GLB_MAGIC, f"Not a GLB file: {path}"

    # --- JSON chunk ---
    json_len, json_type = struct.unpack_from("<II", raw, 12)
    assert json_type == CHUNK_JSON
    json_bytes = raw[20 : 20 + json_len]
    gltf = json.loads(json_bytes.decode("utf-8"))

    # --- BIN chunk ---
    bin_offset = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", raw, bin_offset)
    assert bin_type == CHUNK_BIN
    bin_buffer = raw[bin_offset + 8 : bin_offset + 8 + bin_len]

    # --- Skin data ---
    skin = gltf["skins"][0]
    joint_node_indices = skin["joints"]  # list of node indices
    n_joints = len(joint_node_indices)

    # Joint names
    nodes = gltf["nodes"]
    joint_names = [nodes[ni].get("name", f"joint_{ni}") for ni in joint_node_indices]

    # Build node→skin_joint_index map
    node_to_sjidx = {ni: si for si, ni in enumerate(joint_node_indices)}

    # Parent map from hierarchy
    child_to_parent_node = {}
    for ni, node in enumerate(nodes):
        for ci in node.get("children", []):
            child_to_parent_node[ci] = ni

    parent_indices = []
    for si, ni in enumerate(joint_node_indices):
        parent_node = child_to_parent_node.get(ni, -1)
        if parent_node in node_to_sjidx:
            parent_indices.append(node_to_sjidx[parent_node])
        else:
            parent_indices.append(-1)

    # Rest-pose local TRS
    rest_local_trs = []
    for ni in joint_node_indices:
        rest_local_trs.append(_node_local_trs(nodes[ni]))

    # Rest-pose world matrices (top-down walk)
    rest_world = np.zeros((n_joints, 4, 4), dtype=np.float64)
    # Topological order: parents before children
    visited = set()
    topo_order = []

    def _topo(si):
        if si in visited:
            return
        pi = parent_indices[si]
        if pi >= 0 and pi not in visited:
            _topo(pi)
        visited.add(si)
        topo_order.append(si)

    for si in range(n_joints):
        _topo(si)

    for si in topo_order:
        t, r, s = rest_local_trs[si]
        local_mat = _trs_to_mat4(t, r, s)
        pi = parent_indices[si]
        if pi < 0:
            rest_world[si] = local_mat
        else:
            rest_world[si] = rest_world[pi] @ local_mat

    # Inverse bind matrices
    ibm_acc = skin["inverseBindMatrices"]
    ibm_raw = _read_accessor(gltf, bin_buffer, ibm_acc)  # (N, 16)
    ibms = ibm_raw.reshape(n_joints, 4, 4)
    # glTF stores matrices column-major
    ibms = ibms.transpose(0, 2, 1)  # → row-major

    # AniMo joint name → GLB joint index mapping
    glb_name_to_idx = {name: si for si, name in enumerate(joint_names)}
    animo_to_glb = {}
    for animo_idx, animo_name in enumerate(JOINT_NAMES):
        if animo_name in glb_name_to_idx:
            animo_to_glb[animo_idx] = glb_name_to_idx[animo_name]

    return GLBData(
        json_tree=gltf,
        bin_buffer=bin_buffer,
        joint_names=joint_names,
        parent_indices=parent_indices,
        rest_local_trs=rest_local_trs,
        rest_world_matrices=rest_world,
        inverse_bind_matrices=ibms,
        animo_to_glb=animo_to_glb,
    )
