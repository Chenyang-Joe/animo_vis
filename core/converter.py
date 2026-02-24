"""Orchestrator: load → decode → build scene → export GLB."""

import numpy as np
from pathlib import Path

from .joint_config import JOINT_COLORS, BONE_CONNECTIONS
from .scene_builder import build_scene
from .animation import make_postprocessors
from .decoder import decode_joint_vecs


def convert(npy_path, output_path, mode="joints", fps=30, sphere_radius=0.008, max_frames=None):
    """
    Convert an .npy file to an animated .glb.

    Parameters
    ----------
    npy_path : str or Path
    output_path : str or Path
    mode : 'joints' or 'vecs'
    fps : int
    sphere_radius : float
    max_frames : int or None — truncate animation for testing
    """
    if mode == "joints":
        data = np.load(npy_path).astype(np.float32)  # (T, 30, 3)
    else:
        data = decode_joint_vecs(npy_path)  # (T, 30, 3)

    if max_frames is not None and data.shape[0] > max_frames:
        data = data[:max_frames]

    scene = build_scene(data[0], JOINT_COLORS, BONE_CONNECTIONS, sphere_radius)
    buf_pp, tree_pp = make_postprocessors(data, BONE_CONNECTIONS, fps)

    glb_bytes = scene.export(
        file_type="glb",
        buffer_postprocessor=buf_pp,
        tree_postprocessor=tree_pp,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(glb_bytes)
    return output_path
