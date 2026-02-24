"""Build a trimesh.Scene with colored spheres (joints) and cylinder bones."""

import numpy as np
import trimesh


def _quat_xyzw_to_matrix(x, y, z, w):
    """Convert quaternion (xyzw) to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _bone_transform(start, end):
    """Compute 4x4 transform for a unit Z-cylinder to span start→end.

    The unit cylinder has height=1 along Z, centered at origin.
    Returns transform matrix, or None if degenerate.
    """
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return None

    midpoint = (start + end) / 2.0
    d = direction / length
    z = np.array([0, 0, 1], dtype=np.float64)
    dot = np.dot(z, d)

    if dot > 0.9999:
        R = np.eye(3)
    elif dot < -0.9999:
        R = _quat_xyzw_to_matrix(1, 0, 0, 0)  # 180° around X
    else:
        cross = np.cross(z, d)
        axis = cross / np.linalg.norm(cross)
        half = np.arccos(np.clip(dot, -1, 1)) / 2
        R = _quat_xyzw_to_matrix(
            axis[0]*np.sin(half), axis[1]*np.sin(half),
            axis[2]*np.sin(half), np.cos(half))

    # Apply non-uniform scale: [1, 1, length]
    RS = R.copy()
    RS[:, 2] *= length

    mat = np.eye(4)
    mat[:3, :3] = RS
    mat[:3, 3] = midpoint
    return mat


def build_scene(positions_frame0, joint_colors, bone_connections, sphere_radius=0.008):
    """
    Build a trimesh.Scene from frame-0 joint positions.

    Bones are unit cylinders (height=1, Z-axis) with node transforms — not
    baked into vertices — so glTF animation can drive TRS per frame.
    """
    scene = trimesh.Scene()

    # Add joint spheres
    for i, pos in enumerate(positions_frame0):
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=sphere_radius)
        sphere.visual.face_colors = joint_colors[i]
        transform = np.eye(4)
        transform[:3, 3] = pos
        scene.add_geometry(
            sphere,
            node_name=f"joint_{i}",
            geom_name=f"joint_{i}_geom",
            transform=transform,
        )

    # Add bone cylinders — unit height, transform via node
    bone_radius = sphere_radius * 0.3
    for parent, child in bone_connections:
        start = positions_frame0[parent].astype(np.float64)
        end = positions_frame0[child].astype(np.float64)
        mat = _bone_transform(start, end)
        if mat is None:
            continue
        cyl = trimesh.creation.cylinder(radius=bone_radius, height=1.0, sections=6)
        pc = np.array(joint_colors[parent], dtype=np.float32)
        cc = np.array(joint_colors[child], dtype=np.float32)
        cyl.visual.face_colors = ((pc + cc) / 2).astype(np.uint8)
        name = f"bone_{parent}_{child}"
        scene.add_geometry(
            cyl,
            node_name=name,
            geom_name=f"{name}_geom",
            transform=mat,
        )

    return scene
