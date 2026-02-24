import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(_CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

# Ordered list of 30 joint names (0-indexed)
JOINT_NAMES = list(_cfg["joint_names"].keys())

# Kinematic chain from AniMo paramUtil.py
KINEMATIC_CHAIN = [
    [0, 20, 21, 22, 23, 24],
    [0, 25, 26, 27, 28, 29],
    [0, 9, 10, 11],
    [0, 1, 2, 3, 4],
    [1, 7, 12, 13, 14, 15],
    [1, 8, 16, 17, 18, 19],
    [3, 5],
    [3, 6],
]

# Derive bone connections: (parent, child) pairs from consecutive chain entries
BONE_CONNECTIONS = []
for chain in KINEMATIC_CHAIN:
    for i in range(len(chain) - 1):
        BONE_CONNECTIONS.append((chain[i], chain[i + 1]))

# Named color → RGBA (0-255)
_COLOR_NAME_TO_RGBA = {
    "blue": [0, 0, 255, 255],
    "green": [0, 180, 0, 255],
    "red": [255, 0, 0, 255],
    "skyblue": [135, 206, 235, 255],
    "cyan": [0, 255, 255, 255],
    "purple": [128, 0, 128, 255],
    "gray": [128, 128, 128, 255],
    "black": [40, 40, 40, 255],
    "pink": [255, 192, 203, 255],
    "orange": [255, 165, 0, 255],
    "yellow": [255, 255, 0, 255],
}

# Build joint index → RGBA color
_joint_to_part = {}
for part, joints in _cfg["body_parts"].items():
    for jname in joints:
        # First assignment wins (e.g. eye joints appear in both head and eye)
        if jname not in _joint_to_part:
            _joint_to_part[jname] = part

JOINT_COLORS = []
for jname in JOINT_NAMES:
    part = _joint_to_part.get(jname, "black")
    color_name = _cfg["color_map"].get(part, "black")
    JOINT_COLORS.append(_COLOR_NAME_TO_RGBA.get(color_name, [40, 40, 40, 255]))
