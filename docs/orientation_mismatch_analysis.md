# Orientation Mismatch Analysis: npy2glb_skeleton.py vs npy2glb_recon_mesh.py

## Summary

The two GLB pipelines produced outputs with mismatched root positions and subtly
different orientations. **Root cause**: the mesh pipeline placed the root at the
GLB rest-pose position (which differs from the AniMo root position by up to 24cm),
while the skeleton pipeline uses AniMo positions directly.

**Status: FIXED** — `core/retarget.py` now uses AniMo root positions directly.

---

## Root Cause

### 1. Root position mismatch (primary)

The old code started the mesh root at the GLB rest-pose position and added AniMo
displacement:

```python
# OLD (core/retarget.py lines 333-338)
root_rest_trans = glb_data.rest_local_trs[root_glb][0]
root_displacement = animo_root - animo_root[0]
root_translations = root_rest_trans + root_displacement
```

This caused the mesh root to be offset from the skeleton root by
`rest_pos - animo_root[0]`, which varies by animal:

| Animal                  | Y offset | Z offset | Total  |
|-------------------------|----------|----------|--------|
| Rednecked Wallaby       | -0.005   | -0.016   | 0.017  |
| Bactrian Camel          | +0.082   | -0.148   | 0.169  |
| Reticulated Giraffe     | +0.223   | -0.099   | **0.244** |

For the giraffe, the mesh root was **24cm** away from the skeleton root.

### 2. Heading delta residual (minor, inherent)

The heading delta (`qbetween(GLB_rest_forward, AniMo_forward)`) correctly rotates
the mesh to face the AniMo forward direction. The residual orientation error
(2-3°) comes from bone length differences between the GLB rest-pose skeleton
and the AniMo joint distances. This is inherent to the delta-from-rest approach
and not fixable without a full IK solver.

| Animal              | Heading offset (rest vs AniMo frame-0) | Residual after retarget |
|---------------------|----------------------------------------|------------------------|
| Capuchin Monkey     | 0.1°                                   | ~0°                    |
| Bactrian Camel      | 1.0°                                   | ~1°                    |
| Pygmy Hippo         | 9.2°                                   | ~2°                    |
| Gray Wolf           | 12.4°                                  | ~2°                    |
| Rednecked Wallaby   | 21.8°                                  | ~3°                    |

---

## Fix Applied

```python
# NEW (core/retarget.py)
# Use AniMo root positions directly — matches skeleton pipeline exactly
root_translations = positions[:, 0, :].copy()
```

This ensures:
- Root position matches between skeleton and mesh at every frame
- Forward direction matches within 2-3° (inherent bone-length limitation)

---

## Pipeline Comparison

| Aspect | Skeleton (npy2glb_skeleton) | Mesh (npy2glb_recon_mesh) |
|--------|---------------------------|--------------------------|
| Root position | AniMo `positions[:, 0]` directly | AniMo `positions[:, 0]` directly (FIXED) |
| Joint positions | Raw AniMo positions as translations | Derived from rotation chain + rest bone offsets |
| Orientation | Implicit from joint positions | Heading delta + per-bone deltas |
| Bone lengths | N/A (point-to-point) | GLB rest-pose proportions |

## Key Code Locations

| File | Lines | Role |
|------|-------|------|
| `core/retarget.py:226-237` | Root heading delta: `qbetween(rest_forward, actual_forward)` |
| `core/retarget.py:157-173` | Forward direction from hip/shoulder positions |
| `core/retarget.py:333-335` | Root translation (FIXED: uses AniMo positions directly) |
| `core/scene_builder.py:64-74` | Skeleton places raw positions with identity transform |
| `core/animation.py:150,178-183` | Skeleton animates raw position translations |
