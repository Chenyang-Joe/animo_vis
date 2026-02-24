"""Decode new_joint_vecs (T, 359) → (T, 30, 3) joint positions.

Port of recover_from_ric() from AniMo/utils/motion_process.py to pure numpy.

Note: The saved new_joint_vecs .npy files are already in raw (unnormalized) form.
Mean.npy/Std.npy exist for training normalization but are NOT needed here.
"""

import numpy as np


def _qinv(q):
    """Inverse of quaternion(s) q (..., 4). Assumes unit quaternions."""
    qi = q.copy()
    qi[..., 1:] *= -1
    return qi


def _qrot(q, v):
    """Rotate vector(s) v (..., 3) by quaternion(s) q (..., 4)."""
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    qvec = q[:, 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    result = v + 2 * (q[:, :1] * uv + uuv)
    return result


def decode_joint_vecs(npy_path, joints_num=30):
    """Decode a (T, 359) feature vector into (T, 30, 3) joint positions."""
    data = np.load(npy_path).astype(np.float64)  # (T, 359)

    T = data.shape[0]

    # --- recover_root_rot_pos ---
    rot_vel = data[:, 0]  # (T,)
    r_rot_ang = np.zeros(T, dtype=np.float64)
    r_rot_ang[1:] = rot_vel[:-1]
    r_rot_ang = np.cumsum(r_rot_ang)

    # Y-axis rotation quaternion: q = (cos(a), 0, sin(a), 0) in wxyz convention
    r_rot_quat = np.zeros((T, 4), dtype=np.float64)
    r_rot_quat[:, 0] = np.cos(r_rot_ang)
    r_rot_quat[:, 2] = np.sin(r_rot_ang)

    # Recover root position
    r_pos = np.zeros((T, 3), dtype=np.float64)
    r_pos[1:, [0, 2]] = data[:-1, 1:3]
    # Rotate by inverse root quaternion
    inv_q = _qinv(r_rot_quat)  # (T, 4)
    r_pos = _qrot(inv_q, r_pos).reshape(T, 3)
    r_pos = np.cumsum(r_pos, axis=0)
    r_pos[:, 1] = data[:, 3]

    # --- recover_from_ric ---
    # Extract local joint positions: indices [4 : (joints_num-1)*3 + 4]
    positions = data[:, 4:(joints_num - 1) * 3 + 4]  # (T, 87)
    positions = positions.reshape(T, joints_num - 1, 3)

    # Rotate local joints to world frame
    inv_q_expanded = np.broadcast_to(inv_q[:, np.newaxis, :], (T, joints_num - 1, 4))
    positions = _qrot(inv_q_expanded, positions).reshape(T, joints_num - 1, 3)

    # Add root XZ to joints
    positions[..., 0] += r_pos[:, np.newaxis, 0]
    positions[..., 2] += r_pos[:, np.newaxis, 2]

    # Concatenate root and joints
    positions = np.concatenate([r_pos[:, np.newaxis, :], positions], axis=1)

    return positions.astype(np.float32)  # (T, 30, 3)
