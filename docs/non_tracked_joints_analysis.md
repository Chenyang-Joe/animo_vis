# Non-Tracked Joint Reconstruction Quality Analysis

## Overview

The AniMo pipeline tracks 30 joints. The GLB reference skeletons contain 117-169 joints
(varying by species). The remaining 87-139 joints per skeleton are "non-tracked" -- they
receive no direct positional data from AniMo and must have their rotations inferred or
approximated.

**Status: FIXED** — A tiered strategy is now implemented in `core/retarget.py` that
computes deltas for 20-24 additional joints per skeleton (species-dependent), bringing
the total animated joints from 30 to ~52-54.

---

## Current Implementation

In `core/retarget.py` (lines 300-317), the logic for non-tracked joints is:

```python
# Non-tracked child: world = parent_world * rest_local
rest_local_xyzw = glb_data.rest_local_trs[si][1]
rest_local_wxyz = np.array([rest_local_xyzw[3], rest_local_xyzw[0],
                            rest_local_xyzw[1], rest_local_xyzw[2]])
rest_q = np.tile(rest_local_wxyz, (T, 1))
world_anim[:, si] = qmul(world_anim[:, pi], rest_q)
```

This computes `world_anim[child] = world_anim[parent] * rest_local[child]`, meaning the
child keeps its rest-pose offset from its parent. When the parent rotates (because it is
tracked, or because its own parent is tracked), the non-tracked child rigidly follows,
but never adjusts its own local rotation. The result is that the non-tracked joint
maintains the same relative angle to its parent as in the T-pose/rest-pose, regardless
of the actual motion.

Additionally, in `core/glb_writer.py` (line 125), the output GLB only writes animation
channels for `tracked_glb_indices`, so non-tracked joints receive no animation keyframes
at all in the output file -- they fall back to the rest-pose local TRS stored in the
glTF node, not even the parent-propagated values computed in `retarget()`.

**Critical bug/limitation**: The retarget function computes adjusted world rotations for
ALL joints (tracked and non-tracked) and converts them to local rotations. However, the
writer discards non-tracked joints' computed local rotations entirely by only writing
channels for tracked indices. This means the parent-propagation logic in retarget.py has
no effect on the final output.

---

## Non-Tracked Joint Categories

Based on analysis of multiple species (Gray Wolf, Bactrian Camel, Capuchin Monkey, Black
Rhino), non-tracked joints fall into these categories:

### 1. Spine Subdivisions (HIGH IMPACT, 4 joints)

**Joints**: `def_c_hips_joint`, `def_c_spine1_joint`, `def_c_spine2_joint`,
`def_c_spine3_joint`

**Hierarchy**: `def_c_root_joint [TRACKED] -> def_c_hips_joint -> def_c_spine1_joint ->
def_c_spine2_joint -> def_c_spine3_joint -> def_c_chest_joint [TRACKED]`

**Problem**: There are 4 non-tracked joints forming a chain between the root and chest.
These joints distribute the torso bend across the spine. In the current approach, all
spine bending is concentrated at the chest joint (the first tracked joint after root).
This creates an unnatural "broken back" appearance where the spine snaps at a single
point instead of curving smoothly. This is the single highest-impact problem because the
spine is the largest, most visible articulated structure.

### 2. Intermediate Chain Joints (HIGH IMPACT, 4-8 joints per animal)

**Joints**: `def_frontHorselink_joint.{L,R}`, `def_rearHorselink_joint.{L,R}`,
`def_c_neck2_joint`

**Hierarchy examples**:
- `def_frontLegLwr_joint [TRACKED] -> def_frontHorselink_joint -> def_frontFoot_joint [TRACKED]`
- `def_c_neck1_joint [TRACKED] -> def_c_neck2_joint -> def_c_head_joint [TRACKED]`

**Problem**: The "horselink" joint sits between the lower leg and foot in the GLB
hierarchy. It functions as an additional articulation point (the pastern/fetlock in
hooved animals, or an ankle-like joint in digitigrade animals). Keeping it at rest pose
means the foot orientation is derived from a chain that skips a real joint, causing the
foot to float or penetrate the ground. Similarly, `def_c_neck2_joint` between neck1 and
head concentrates all neck bending at neck1, losing the smooth neck curve.

### 3. Twist/Roll Joints (MEDIUM IMPACT, 10 joints)

**Non-tracked twist joints**: `def_frontLegLwrHalfTwist_joint.{L,R}`,
`def_frontLegUprHalfTwist_joint.{L,R}`, `def_frontLegUprAllTwist_joint.{L,R}`,
`def_rearLegLwrHalfTwist_joint.{L,R}`, `def_rearLegUprHalfTwist_joint.{L,R}`

**Note**: The "AllTwist" variants for lower-leg and rear-upper-leg ARE tracked in AniMo
(6 of the 30 joints). The "HalfTwist" variants and front-upper-leg AllTwist are NOT
tracked.

**Problem**: Twist joints handle the axial rotation (twist along the bone axis) of limb
segments. Without proper twist values, the mesh shows the "candy wrapper" deformation
artifact -- the skin between two joints twists unrealistically when the distal joint
rotates around its longitudinal axis. The HalfTwist joints are designed to carry 50% of
the twist to distribute deformation smoothly; at rest pose, they contribute 0% twist.

### 4. Tail Extensions (MEDIUM IMPACT, 3-10 joints)

**Joints**: `def_c_tail3_joint` through `def_c_tail5_joint` (wolf) or up to
`def_c_tail12_joint` (monkey)

**Hierarchy**: `def_c_tail2_joint [TRACKED] -> def_c_tail3_joint -> def_c_tail4_joint
-> ...`

**Problem**: AniMo tracks only tail1 and tail2. Many animals have long tails with 3-12
segments. The untracked segments remain rigid at rest pose, causing the tail to appear
stiff and unnatural. Long-tailed species (monkeys with up to 10 extra tail joints) are
most affected.

### 5. Facial Detail Joints (LOW IMPACT, 30-50 joints)

**Joints**: Lips, tongue, eyelids, brows, cheeks, nostrils, ears, jowls, snout

**Problem**: These joints control facial expressions and secondary motions (ear
twitching, lip movement, etc.). AniMo tracks head, jaw, and eye joints but not the
detailed facial rig. At rest pose, the face appears frozen. However, since AniMo does
not capture this level of facial detail anyway, the information to drive these joints
does not exist in the input data.

### 6. Toe/Digit Joints (LOW IMPACT, 16-20 joints per animal)

**Joints**: `def_toeFrontMid1_joint`, `def_toeFrontMid2_joint`, etc.

**Problem**: Individual toe joints remain at rest pose. Since AniMo tracks only the foot
as a whole, there is no per-toe information. The visual impact is low because toes are
small and rarely the focus of attention.

### 7. Breath/Secondary Joints (LOW IMPACT, 3 joints)

**Joints**: `def_c_chestBreath_joint`, `def_chestBreath_joint.{L,R}`

**Problem**: These simulate chest expansion during breathing. Without animation, the
chest appears static. Impact is subtle but contributes to an overall "lifeless" feel.

---

## Alternative Approaches

### Approach A: Delta Inheritance

**Description**: Non-tracked joints inherit the delta rotation of their nearest tracked
ancestor. Instead of `world = parent_world * rest_local`, compute
`world = ancestor_delta * rest_world`.

**Implementation**: For each non-tracked joint, walk up the hierarchy to find the
nearest tracked ancestor. Use that ancestor's delta (from the `deltas` dict in
retarget.py) as if it were the non-tracked joint's own delta.

**Pros**:
- Simple to implement (5-10 lines of code change)
- Works for all joint categories
- No additional data needed
- Spine joints would at least bend with the root, reducing the "broken back" effect

**Cons**:
- Applies the same rotation everywhere, which is physically wrong for most cases. A
  spine should distribute rotation gradually, not uniformly.
- Twist joints need twist-axis-specific handling, not full rotation inheritance
- Tail joints would all point the same direction as tail2, losing any natural cascade
- Can actually look worse than rest-pose for joints far from the ancestor

**Best for**: Simple fallback, quick improvement for joints very close to their ancestor

### Approach B: Spherical Interpolation (Slerp) Between Tracked Ancestors/Descendants

**Description**: For non-tracked joints sandwiched between two tracked joints in the
hierarchy, compute the delta as a spherical linear interpolation (slerp) between the
ancestor's delta and the descendant's delta, weighted by the joint's position in the
chain.

**Implementation**:
1. For each non-tracked joint, find the nearest tracked ancestor AND the nearest tracked
   descendant in the hierarchy.
2. Compute the chain position `t = steps_from_ancestor / total_chain_length`.
3. `delta[non_tracked] = slerp(delta[ancestor], delta[descendant], t)`
4. `world = delta * rest_world`

**Example**: For the spine chain `root [T] -> hips -> spine1 -> spine2 -> spine3 ->
chest [T]`, with 4 intermediate joints:
- hips: t=1/5, delta = slerp(root_delta, chest_delta, 0.2)
- spine1: t=2/5, delta = slerp(root_delta, chest_delta, 0.4)
- spine2: t=3/5, delta = slerp(root_delta, chest_delta, 0.6)
- spine3: t=4/5, delta = slerp(root_delta, chest_delta, 0.8)

**Pros**:
- Produces smooth, natural-looking distributions for spine and neck
- Physically plausible for chain joints (horselink between lower leg and foot)
- Computationally cheap (slerp is a single operation per joint per frame)
- Handles the highest-impact category (spine) very well

**Cons**:
- Only works for joints that have BOTH a tracked ancestor and tracked descendant
  (does not apply to leaf joints like toes, tail extensions, or facial joints)
- Assumes linear distribution of rotation, which may not match actual anatomy
  (e.g., lumbar vertebrae flex more than thoracic)
- Requires building ancestor/descendant lookup per non-tracked joint

**Best for**: Spine subdivisions, neck2, horselink joints -- the highest-impact
categories

### Approach C: Twist Decomposition

**Description**: For twist/roll joints, decompose the rotation between parent and child
tracked joints into twist (rotation around the bone axis) and swing components. Apply
only the twist component, scaled by the joint's twist fraction (0.5 for HalfTwist, 1.0
for AllTwist).

**Implementation**:
1. Identify twist joints by name pattern (`*Twist*` or `*Roll*`).
2. Compute the bone axis from the rest pose (direction from parent to child).
3. Decompose the tracked parent's rotation into swing and twist around the bone axis.
4. For HalfTwist: apply 50% of the twist component.
5. For AllTwist: apply 100% of the twist component.

**Twist-swing decomposition**: Given a quaternion `q` and axis `a`:
- `twist = normalize(q.w, dot(q.xyz, a) * a)` (project imaginary part onto axis)
- `swing = q * conjugate(twist)`

**Pros**:
- Anatomically correct for the specific purpose of twist joints
- Eliminates the "candy wrapper" deformation artifact
- The naming convention (`HalfTwist`, `AllTwist`) directly encodes the fraction
- Well-established technique in game animation

**Cons**:
- Only applies to twist joints (10 joints), not other categories
- Requires correct identification of the twist axis (bone direction)
- Slightly more complex math (twist-swing decomposition)
- The tracked AllTwist joints from AniMo may already carry some twist information,
  so the HalfTwist joints may need to reference the AllTwist rotation rather than the
  parent joint

**Best for**: Specifically the 10 non-tracked HalfTwist and UprAllTwist joints

### Approach D: Copy from Reference Animation

**Description**: The reference GLB files contain full animation data for ALL 139 joints
(verified: all joints have rotation keyframes). For non-tracked joints, copy the local
rotation directly from the reference animation rather than computing from AniMo data.

**Implementation**:
1. Parse animation channels from the reference GLB for non-tracked joints.
2. For each non-tracked joint, extract the rotation keyframe data.
3. When the AniMo animation has different frame count, resample (interpolate) the
   reference animation to match.
4. Use the reference animation's local rotation for non-tracked joints.

**Pros**:
- Produces the most visually complete result for facial joints, toes, breath joints
- Uses actual authored animation data rather than approximations
- No mathematical approximations or assumptions needed
- Handles ALL non-tracked joint categories

**Cons**:
- **Fundamentally flawed for the core problem**: The reference animation is a DIFFERENT
  motion than what AniMo is producing. A reference walk cycle's spine rotations are
  wrong for an AniMo-captured run or idle pose.
- Frame count mismatch: reference animations may have different lengths
- Creates a temporal Frankenstein: tracked joints follow AniMo while non-tracked joints
  follow a completely different animation. This will cause visible discontinuities at
  joints that border tracked/non-tracked boundaries (e.g., spine3 from reference +
  chest from AniMo).
- Only works when a reference GLB with animation is available
- Does not generalize to novel motions not in the reference set

**Best for**: Potentially useful as a FALLBACK for purely cosmetic joints (facial
details, breath) that have no other information source, but only if the reference
animation is carefully selected or the motion is similar.

### Approach E: Proportional Rotation Distribution

**Description**: When a tracked joint has a large rotation delta relative to its parent,
distribute a fraction of that delta to intermediate non-tracked children. The
distribution follows a decay or proportional rule based on chain depth.

**Implementation**:
1. For each tracked joint, compute its delta from rest.
2. Walk the chain from tracked parent to tracked child.
3. Distribute the total delta across the chain using weights
   `w[i] = (n - i) / sum(1..n)` (linear decay) or equal weights `w[i] = 1/n`.
4. Each non-tracked joint gets `delta[i] = slerp(identity, total_delta, w[i])`.

**Pros**:
- More nuanced than uniform delta inheritance
- Can produce smooth bending for spine-like chains
- Works without needing to know the descendant's delta (unlike slerp approach B)

**Cons**:
- Somewhat arbitrary weight distribution
- Does not account for the descendant tracked joint's delta, so the total rotation may
  not sum to the correct end result
- Less accurate than slerp between two known endpoints (approach B)
- Similar mathematical complexity to approach B but with less information

**Best for**: Situations where only the ancestor delta is known (no tracked descendant),
such as tail extensions beyond tail2.

---

## Implemented Fix: Tiered Strategy

The tiered strategy is implemented in `core/retarget.py` (Step 2b, lines ~331-411).
After computing deltas for the 30 tracked joints, three additional tiers add deltas for
non-tracked joints. The writer (`core/glb_writer.py`) automatically outputs channels for
all joints with deltas since `tracked_glb = set(deltas.keys())`.

### Tier 1: Slerp Interpolation for Sandwiched Joints (IMPLEMENTED)

For each tracked joint, walk UP the GLB hierarchy until hitting another tracked joint.
Non-tracked joints on that path get a slerp-interpolated delta:

```python
t = (position_in_chain) / (chain_length + 1)
delta[joint] = slerp(delta[ancestor], delta[descendant], t)
```

Affected chains:
- `root[T] → [hips, spine1, spine2, spine3] → chest[T]` (4 joints, t=0.2/0.4/0.6/0.8)
- `neck1[T] → [neck2] → head[T]` (1 joint, t=0.5)
- `frontLegLwr[T] → [frontHorselink] → frontFoot[T]` (2 joints, L+R)
- `rearLegLwr[T] → [rearHorselink] → rearFoot[T]` (2 joints, L+R)

### Tier 2: Twist Decomposition for Twist Joints (IMPLEMENTED)

For joints with "Twist" in their name:
- `HalfTwist`: extract 50% of ancestor's twist component around bone axis
- Non-tracked `AllTwist` (front upper leg): extract 100% of twist

```python
bone_axis = normalize(rest_pos[twist_joint] - rest_pos[ancestor])
twist_q = twist_decompose(delta[ancestor], bone_axis)
delta[twist_joint] = slerp(identity, twist_q, fraction)
```

Affected: 8 HalfTwist + 2 front AllTwist = **10 joints**

### Tier 3: Exponential Decay for Tail Extensions (IMPLEMENTED)

Tail joints beyond tail2 get the parent's delta attenuated by 0.7 per segment.
Processed in topological order so decay compounds naturally:

- tail3: 0.7 × tail2 delta
- tail4: 0.7 × tail3 = 0.49 × tail2
- tail5: 0.7 × tail4 = 0.34 × tail2
- etc.

Affected: 3 joints (short-tailed species) to 10 joints (monkeys)

### Tier 4: Rest Pose for Facial/Toe/Breath Joints (Unchanged)

50-70 joints per skeleton keep rest-pose local rotation. No AniMo data exists for these.

---

## Verification Results

| Species | GLB Joints | Original Tracked | After Tiers | New Joints Added |
|---------|-----------|-----------------|-------------|------------------|
| Gray Wolf | 139 | 30 | 52 | +9 slerp, +10 twist, +3 tail |
| Capuchin Monkey | 169 | 30 | 54 | +4 slerp, +10 twist, +10 tail |

All 24 batch samples (12 animals × 2 actions) regenerated successfully.

---

## Key Code Locations

| File | Lines | Role |
|------|-------|------|
| `core/retarget.py:331-337` | Children map construction |
| `core/retarget.py:339-361` | Tier 1: slerp for sandwiched chains |
| `core/retarget.py:363-392` | Tier 2: twist decomposition |
| `core/retarget.py:394-411` | Tier 3: tail decay |
| `core/retarget.py:77-96` | `qslerp()` helper |
| `core/retarget.py:99-113` | `_twist_decompose()` helper |

---

## Remaining Limitations

- **Facial joints**: No input data from AniMo. Would require facial capture.
- **Toe joints**: No per-toe data. Visual impact is minimal.
- **Breath joints**: Could be added as procedural sinusoidal animation in the future.
- **Spine weight distribution**: Currently linear slerp. Anatomically, lumbar vertebrae
  flex more than thoracic. Could use non-uniform weights if needed.
- **Tail dynamics**: Exponential decay is static. A physics-based spring simulation
  would produce more natural secondary motion but adds complexity.
