#!/usr/bin/env python3
"""Batch npy2glb_recon_mesh_raw for matched animals in glb_example.

Same samples as batch_sample2.py but uses the raw (no tiered optimization)
pipeline. Output goes to mesh01/ for side-by-side comparison with mesh/.
"""

import os
import random
import subprocess
import sys

random.seed(42)

GLB_BASE = "/home/ubuntu/cy_home/codebase/animo4d_vis/sample_data/glb_example"
NPY_DIR = "/opt/dlami/nvme/animo_data/AniMo4D/new_joints"
OUTDIR = "/home/ubuntu/cy_home/codebase/animo4d_vis/test_out/batch_20"
CD = "/home/ubuntu/cy_home/codebase/animo4d_vis"

os.makedirs(f"{OUTDIR}/mesh01", exist_ok=True)

npy_files = os.listdir(NPY_DIR)

pairs = []  # (animal_dir, ref_glb_path, npy_name)

for animal_dir in sorted(os.listdir(GLB_BASE)):
    prefix = animal_dir.lower().replace(" ", "_")
    matches = [f for f in npy_files if f.startswith(prefix + "__") and f.endswith(".npy")]
    if not matches:
        continue

    # Pick first GLB as reference
    glb_dir = os.path.join(GLB_BASE, animal_dir)
    ref_glb = os.path.join(glb_dir, sorted(os.listdir(glb_dir))[0])

    chosen = random.sample(matches, min(2, len(matches)))
    for npy_name in sorted(chosen):
        pairs.append((animal_dir, ref_glb, npy_name))

print(f"Processing {len(pairs)} files from {len(set(p[0] for p in pairs))} animals\n")

for i, (animal, ref_glb, npy_name) in enumerate(pairs):
    stem = npy_name.replace("_keypoints.json.npy", "").replace(".npy", "")
    npy_path = os.path.join(NPY_DIR, npy_name)

    print(f"[{i+1}/{len(pairs)}] {animal}: {stem}")

    mesh_out = f"{OUTDIR}/mesh01/{stem}_mesh.glb"
    r = subprocess.run(
        [sys.executable, f"{CD}/npy2glb_recon_mesh_raw.py", npy_path, ref_glb,
         "--output", mesh_out],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr.strip()}")
    else:
        print(f"  OK")

print(f"\nDone!")
print(f"Raw mesh: {OUTDIR}/mesh01/")
print(f"Tiered mesh: {OUTDIR}/mesh/")
