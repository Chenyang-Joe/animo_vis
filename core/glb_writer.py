"""Build and write a GLB file with new animation data on the original mesh/skin."""

import copy
import json
import struct

import numpy as np


def build_animation_glb(
    glb_data,
    local_rotations_xyzw: np.ndarray,
    root_translations: np.ndarray,
    tracked_glb_indices: set,
    fps: float = 30.0,
    output_path: str = "output.glb",
):
    """Write a new GLB with the original mesh/skin and new skeletal animation.

    Args:
        glb_data: GLBData from glb_parser
        local_rotations_xyzw: (T, N, 4) local rotations for all N GLB joints (xyzw)
        root_translations: (T, 3) root joint translations
        tracked_glb_indices: set of GLB joint indices to animate
        fps: animation framerate
        output_path: path to write the output GLB
    """
    T = local_rotations_xyzw.shape[0]
    n_joints = local_rotations_xyzw.shape[1]

    gltf = copy.deepcopy(glb_data.json_tree)
    old_bin = glb_data.bin_buffer

    # Remove existing animations
    gltf.pop("animations", None)

    # Track where new data starts
    existing_accessor_count = len(gltf.get("accessors", []))
    existing_bv_count = len(gltf.get("bufferViews", []))

    # --- Build new animation binary data ---
    new_bin_parts = []
    new_buffer_views = []
    new_accessors = []

    def _append_data(data: np.ndarray, acc_type: str, comp_type: int = 5126):
        """Append float32 data to the animation buffer.
        Returns the index of the newly created accessor."""
        arr = data.astype(np.float32)
        raw = arr.tobytes()

        # Pad to 4-byte alignment
        pad = (4 - len(raw) % 4) % 4
        raw_padded = raw + b"\x00" * pad

        byte_offset = sum(len(p) for p in new_bin_parts)
        new_bin_parts.append(raw_padded)

        bv_idx = existing_bv_count + len(new_buffer_views)
        new_buffer_views.append({
            "buffer": 0,
            "byteOffset": len(old_bin) + byte_offset,
            "byteLength": len(raw),
        })

        acc_idx = existing_accessor_count + len(new_accessors)
        acc_entry = {
            "bufferView": bv_idx,
            "componentType": comp_type,
            "count": arr.shape[0],
            "type": acc_type,
        }

        # Min/max for glTF validation
        if acc_type == "SCALAR":
            acc_entry["min"] = [float(arr.min())]
            acc_entry["max"] = [float(arr.max())]
        elif acc_type == "VEC3":
            acc_entry["min"] = arr.min(axis=0).tolist()
            acc_entry["max"] = arr.max(axis=0).tolist()
        elif acc_type == "VEC4":
            acc_entry["min"] = arr.min(axis=0).tolist()
            acc_entry["max"] = arr.max(axis=0).tolist()

        new_accessors.append(acc_entry)
        return acc_idx

    # Timestamps
    timestamps = np.arange(T, dtype=np.float32) / fps
    time_acc = _append_data(timestamps.reshape(-1, 1), "SCALAR")

    # Skin joint node indices
    joint_node_indices = gltf["skins"][0]["joints"]

    # Find root joint in skin (index 0 in skin.joints is typically root)
    # Use the AniMo root mapping
    root_glb_idx = glb_data.animo_to_glb.get(0)

    # Build samplers and channels
    samplers = []
    channels = []

    def _add_channel(target_node: int, target_path: str, output_acc: int):
        sampler_idx = len(samplers)
        samplers.append({
            "input": time_acc,
            "output": output_acc,
            "interpolation": "LINEAR",
        })
        channels.append({
            "sampler": sampler_idx,
            "target": {
                "node": target_node,
                "path": target_path,
            },
        })

    # Root translation channel
    if root_glb_idx is not None:
        root_node = joint_node_indices[root_glb_idx]
        trans_acc = _append_data(root_translations.astype(np.float32), "VEC3")
        _add_channel(root_node, "translation", trans_acc)

    # Rotation channels — only for tracked joints
    for glb_idx in sorted(tracked_glb_indices):
        node_idx = joint_node_indices[glb_idx]
        rot_data = local_rotations_xyzw[:, glb_idx, :]  # (T, 4) xyzw
        rot_acc = _append_data(rot_data, "VEC4")
        _add_channel(node_idx, "rotation", rot_acc)

    # --- Assemble new glTF ---
    if "bufferViews" not in gltf:
        gltf["bufferViews"] = []
    gltf["bufferViews"].extend(new_buffer_views)

    if "accessors" not in gltf:
        gltf["accessors"] = []
    gltf["accessors"].extend(new_accessors)

    gltf["animations"] = [{
        "name": "AniMo_animation",
        "samplers": samplers,
        "channels": channels,
    }]

    # Combine binary buffers
    combined_bin = old_bin + b"".join(new_bin_parts)
    # Pad combined bin to 4-byte alignment
    bin_pad = (4 - len(combined_bin) % 4) % 4
    combined_bin += b"\x00" * bin_pad

    gltf["buffers"][0]["byteLength"] = len(combined_bin)

    # --- Pack GLB ---
    json_str = json.dumps(gltf, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    # Pad JSON to 4-byte alignment with spaces
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_pad

    # GLB structure
    # Header: 12 bytes
    # JSON chunk: 8 + len(json_bytes)
    # BIN chunk: 8 + len(combined_bin)
    total_length = 12 + 8 + len(json_bytes) + 8 + len(combined_bin)

    with open(output_path, "wb") as f:
        # GLB header
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))
        # JSON chunk
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        # BIN chunk
        f.write(struct.pack("<II", len(combined_bin), 0x004E4942))
        f.write(combined_bin)
