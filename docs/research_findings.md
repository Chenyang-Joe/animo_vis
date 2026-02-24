# Research Findings: Skeleton Sequence -> Mesh Animation

## 1. AniMo Model Output Format

- **Final output**: `(T, 359)` RIC feature vector (same as `new_joint_vecs/`)
- Decodes via `recover_from_ric()` -> `(T, 30, 3)` world-space joint positions
- Same format as GT `new_joints/` -- no additional conversion needed
- **Key files**: `AniMo/utils/eval_t2m.py:859-878`, `AniMo/models/vq/model.py:65-69`

## 2. Sample Data Analysis

### Raw data (`sample_data/aardvark_female_raw/`)
- `aardvark_female_.ms2` (3.3 MB) -- skeleton + mesh
- 7 `.manis` files (4.9 MB total) -- animations

### Reference GLB (`sample_data/aardvark_female_glb/`)
- 2 GLB files (~3.6 MB each), 4 mesh primitives, 33,930 vertices
- **143 skeleton joints** with full skinning (JOINTS_0, WEIGHTS_0, inverse bind matrices)
- **429 animation channels** (143 joints x 3: translation/rotation/scale)
- The 30 AniMo joints are a SUBSET of the 143 GLB joints (all 30 found by name)

## 3. Coordinate System Relationship

### AniMo preprocessing (notebook cell 2):
1. `uniform_skeleton()` -- IK->FK to standardized bone lengths
2. Floor centering (Y -= min)
3. XZ origin (subtract root XZ)
4. Rotate: -90 deg X then -90 deg Y
5. Floor again

### Key finding: bone lengths nearly identical
GLB vs AniMo bone length ratio is ~1.0 for aardvark (template may match the species).
This means for matching species, positions can be mapped directly after undoing rotation.

### Rotation undo: +90 deg Y then +90 deg X (reverse order)

## 4. GLB Skinning Architecture

Deformation formula (Linear Blend Skinning):
```
final_pos = sum(weight[i] * world_joint_matrix[joint_idx[i]] * ibm[joint_idx[i]] * rest_pos)
```

To drive the mesh with new joint positions, we need to provide correct `world_joint_matrix` for each joint.

## 5. Implementation Plan

### Version 1: GLB-based (reference GLB provided)

**Input**: skeleton sequence (T, 30, 3) + reference GLB
**Output**: new GLB with mesh deforming to the given motion

**Algorithm**:
1. Parse reference GLB: skeleton hierarchy, rest pose TRS, inverse bind matrices, mesh/skinning
2. Map 30 AniMo joint names -> GLB skin joint indices
3. For each frame of the AniMo sequence:
   a. Undo coordinate rotation (+90Y, +90X) to get to GLB space
   b. For each of the 30 known joints, compute world transform:
      - Translation = given position
      - Rotation = derived from bone direction (parent->this or this->child)
   c. Convert world -> local TRS using parent inverse
   d. For non-AniMo joints, keep rest pose local TRS
4. Pack into glTF animation channels
5. Write new GLB (keep original mesh/skinning/materials, replace animation)

**Key challenge**: Computing joint ROTATIONS from positions alone.
- Use bone direction vectors (parent->child) to derive orientation
- For twist joints without clear children, interpolate from parent

### Version 2: ms2-based (ms2 file provided, no GLB)

**Input**: skeleton sequence (T, 30, 3) + ms2 file
**Output**: GLB with mesh and animation

**Algorithm**:
1. Parse ms2 binary: extract mesh geometry, skeleton, skinning weights
2. Build glTF structure from scratch (meshes, skin, materials)
3. Apply same joint->animation logic as V1
4. Export as GLB

This requires a Python ms2 parser (port from cobra-tools).
