# Retarget Pipeline Optimizations

## Overview

`core/retarget.py` converts AniMo 30-joint position sequences `(T, 30, 3)` into per-joint
local rotations for a full GLB skeleton (117-169 joints). The pipeline uses a
"delta-from-rest" approach:

```
delta[j] = qbetween(rest_bone_dir, actual_bone_dir)
world_anim[j] = delta[j] * rest_world[j]
local[j] = qinv(world[parent]) * world[j]
```

Below documents all optimizations applied to improve output quality, in pipeline order.

---

## Step 2a: Facial Leaf-Joint Damping

**Problem**: Jaw and eye joints have very short bones from their ancestor (head). The
head→jaw bone is only 1-5cm depending on species. When AniMo positional error is even
1-2cm, the computed bone direction swings wildly, causing the jaw (and attached lip/tongue
mesh) to flail unrealistically.

**Root cause**: `delta = qbetween(rest_dir, actual_dir)` amplifies noise inversely
proportional to bone length. A 1cm error on a 5cm bone is a ~11° angular error; the same
1cm error on a 20cm bone is only ~3°.

**Solution**: For jaw and eye joints, blend their computed delta toward the head's
(ancestor's) delta. The blend factor is proportional to bone length:

```python
factor = min(1.0, bone_length / 0.10)   # bones >= 10cm: full delta
delta[jaw] = slerp(delta[head], delta[jaw], factor)
```

**Why only jaw/eye?** A fixed absolute threshold (10cm) would over-dampen small animals
(e.g. Capuchin Monkey where most bones are <10cm). By targeting only facial joints by
name, we avoid degrading limb/spine accuracy on small species while fixing the visually
most prominent artifact.

**Measured bone lengths and damping factors**:

| Species | head→jaw | factor | head→eye | factor |
|---------|----------|--------|----------|--------|
| Capuchin Monkey | 1.0cm | 0.10 | 4.1cm | 0.41 |
| Bactrian Camel | 2.7cm | 0.27 | 19.5cm | 1.00 |
| Aardvark Female | 1.8cm | 0.18 | 12.4cm | 1.00 |
| Gray Wolf | 4.8cm | 0.48 | 10.5cm | 1.00 |

**Code**: `core/retarget.py` Step 2a (~line 333-351)

---

## Step 2b: Tiered Non-Tracked Joint Deltas

After computing deltas for the 30 tracked joints (and damping facial ones), three tiers
add deltas for non-tracked joints. This increases the total animated joints from 30 to
~52-54 per skeleton.

### Tier 1: Slerp for Sandwiched Chains

**Problem**: Non-tracked joints between two tracked joints (e.g. 4 spine joints between
root and chest) kept rest-pose local rotation, concentrating all bending at a single
tracked joint → "broken back" artifact.

**Solution**: For each tracked joint, walk UP the GLB hierarchy to find the nearest
tracked ancestor. Non-tracked joints on that path get a slerp-interpolated delta:

```python
t = (i + 1) / (chain_length + 1)
delta[joint_i] = slerp(delta[ancestor], delta[descendant], t)
```

**Affected chains** (automatically discovered from GLB hierarchy):

| Chain | Non-tracked joints | t values |
|-------|-------------------|----------|
| root → ... → chest | hips, spine1, spine2, spine3 | 0.2, 0.4, 0.6, 0.8 |
| neck1 → ... → head | neck2 | 0.5 |
| frontLegLwr → ... → frontFoot (×2) | frontHorselink L/R | 0.5 |
| rearLegLwr → ... → rearFoot (×2) | rearHorselink L/R | 0.5 |

**Total**: ~9 joints (species-dependent; some lack neck2 or horselinks)

**Code**: `core/retarget.py` Step 2b Tier 1 (~line 360-383)

### Tier 2: Twist Decomposition

**Problem**: HalfTwist joints (designed to carry 50% of limb axial twist) and non-tracked
AllTwist joints (front upper leg) stayed at rest pose → "candy wrapper" skin deformation
on limb twisting.

**Solution**: Extract the twist component of the ancestor's delta around the bone axis,
then scale by the joint's twist fraction:

```python
bone_axis = normalize(rest_pos[twist_joint] - rest_pos[ancestor])
twist_q = twist_decompose(delta[ancestor], bone_axis)
delta[twist_joint] = slerp(identity, twist_q, fraction)
# fraction: 0.5 for HalfTwist, 1.0 for non-tracked AllTwist
```

**Twist-swing decomposition**: Project the quaternion's imaginary part onto the bone axis
to isolate the twist rotation. This is the standard game-animation technique for
distributing limb twist across helper bones.

**Affected**: 8 HalfTwist + 2 front AllTwist = **10 joints**

**Code**: `core/retarget.py` Step 2b Tier 2 (~line 385-414)

### Tier 3: Tail Exponential Decay

**Problem**: AniMo tracks only tail1 and tail2. Species with long tails (monkey: 12
segments, wolf: 5) had rigid tails beyond tail2.

**Solution**: Each tail joint beyond the last tracked one inherits a decayed version of
its parent's delta. Processed in topological order so decay compounds naturally:

```python
DECAY = 0.7
delta[tail_N] = slerp(identity, delta[tail_N-1], DECAY)
```

This produces exponential falloff:
- tail3: 70% of tail2
- tail4: 49% of tail2
- tail5: 34% of tail2
- ...

The result is a natural "whip-like" continuation where motion amplitude decreases toward
the tail tip.

**Affected**: 3 joints (wolf) to 10 joints (monkey)

**Code**: `core/retarget.py` Step 2b Tier 3 (~line 416-433)

### Tier 4: Rest Pose (Unchanged)

Remaining 85-115 joints (facial details, toes, breath) keep rest-pose local rotation.
No AniMo data exists for these, and their visual impact is low.

---

## `tiered` Flag

All Tier 1-3 logic is gated by `retarget(..., tiered=True)` (default). Passing
`tiered=False` produces the raw 30-joint-only output for comparison. Facial damping
(Step 2a) always applies regardless of the flag since it fixes a noise issue in the
tracked joints themselves.

**Scripts**:
- `npy2glb_recon_mesh.py` — calls `retarget(pos, glb, tiered=True)` (optimized)
- `npy2glb_recon_mesh_raw.py` — calls `retarget(pos, glb, tiered=False)` (raw baseline)

---

## Summary of Pipeline Steps

```
Input: AniMo positions (T, 30, 3) + reference GLB

Step 1:  Root heading delta (align GLB rest-pose forward to AniMo forward)
Step 2:  Per-bone delta for 30 tracked joints
Step 2a: Facial damping (jaw/eye → blend toward head delta)
Step 2b: Tiered non-tracked deltas (slerp → twist → tail decay)
Step 3:  Apply deltas → world rotations for all joints
Step 4:  World → local rotations
Step 5:  Root translation = AniMo root positions directly

Output: local_rotations (T, N, 4), root_translations (T, 3), tracked_indices
```

---

## Verification

| Species | GLB Joints | Raw Tracked | After Optimization | Added |
|---------|-----------|-------------|-------------------|-------|
| Gray Wolf | 139 | 30 | 52 | +9 slerp, +10 twist, +3 tail |
| Capuchin Monkey | 169 | 30 | 54 | +4 slerp, +10 twist, +10 tail |
| Aardvark Female | 143 | 30 | ~52 | +9 slerp, +10 twist, +3 tail |

All 24 batch samples (12 animals × 2 actions) + 4 Aardvark clips generated successfully.

---

## Key Files

| File | Role |
|------|------|
| `core/retarget.py` | All optimization logic (Steps 2a, 2b) |
| `core/retarget.py:77-96` | `qslerp()` — spherical linear interpolation |
| `core/retarget.py:99-113` | `_twist_decompose()` — twist-swing decomposition |
| `core/glb_writer.py` | Outputs channels for `tracked_glb_indices` (auto-expanded) |
| `npy2glb_recon_mesh.py` | CLI with `tiered=True` |
| `npy2glb_recon_mesh_raw.py` | CLI with `tiered=False` (baseline) |
