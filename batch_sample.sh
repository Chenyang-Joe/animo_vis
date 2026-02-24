#!/bin/bash
# Batch convert 20 random samples: npy2glb (skeleton) + skel2glb (mesh)
set -e

OUTDIR="/home/ubuntu/cy_home/codebase/animo4d_vis/test_out/batch_20"
NPY_DIR="/opt/dlami/nvme/animo_data/AniMo4D/new_joints"
SKEL_OUT="$OUTDIR/skeleton"
MESH_OUT="$OUTDIR/mesh"

mkdir -p "$SKEL_OUT" "$MESH_OUT"

# Reference GLBs
REF_AARDVARK="/home/ubuntu/cy_home/codebase/animo4d_vis/sample_data/aardvark_female_glb/Aardvark_Female/aardvark_female__animationmotionextractedbehaviour.manisete15d87f2_aardvark_female_enrichmentboxshake.glb"
REF_WILDDOG="/home/ubuntu/cy_home/codebase/animo4d_vis/sample_data/african_wild_dog_juvenile/african_wild_dog_juvenile__animationmotionextractedbehaviour.maniset995ba078_african_wild_dog_juvenile_enrichmentboxshake_keypoints.json.glb"

CD="/home/ubuntu/cy_home/codebase/animo4d_vis"

# 4 aardvark_female
AARDVARK_FILES=(
  "aardvark_female__animationmotionextractedbehaviour.manisete15d87f2_aardvark_female_enrichmentboxshake_keypoints.json.npy"
  "aardvark_female__animationmotionextractedbehaviour.manisete15d87f2_aardvark_female_enrichmentrub_keypoints.json.npy"
  "aardvark_female__animationmotionextractedbehaviour.manisete15d87f2_aardvark_female_matingritual_keypoints.json.npy"
  "aardvark_female__animationmotionextractedbehaviour.manisete15d87f2_aardvark_female_matingritual_large_keypoints.json.npy"
)

# 16 african_wild_dog_juvenile
WILDDOG_FILES=(
  "african_wild_dog_juvenile__animationmotionextractedbehaviour.maniset995ba078_african_wild_dog_juvenile_interactjuvenilebresting_keypoints.json.npy"
  "african_wild_dog_juvenile__animationmotionextractedlocomotion.manisetb76e0a14_african_wild_dog_juvenile_runbase_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedbehaviour.maniset1aa1bd7a_african_wild_dog_juvenile_walktodrinktroughturnr_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedbehaviour.maniset1aa1bd7a_african_wild_dog_juvenile_walktodrinkturnl_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedbehaviour.maniset1aa1bd7a_african_wild_dog_juvenile_walktodrinkturnr_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedlocomotion.manisete5604cb6_african_wild_dog_juvenile_drinkloop02_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedlocomotion.manisete5604cb6_african_wild_dog_juvenile_enrichmentboxshake_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedlocomotion.manisete5604cb6_african_wild_dog_juvenile_standdie_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedlocomotion.manisete5604cb6_african_wild_dog_juvenile_standturnl089_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedlocomotion.manisete5604cb6_african_wild_dog_juvenile_swimturnr000_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedpartials.maniset943f46f0_african_wild_dog_juvenile_interactdominant_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedpartials.maniset943f46f0_african_wild_dog_juvenile_jumpmidonspot_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedpartials.maniset943f46f0_african_wild_dog_juvenile_standpreen01_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedpartials.maniset943f46f0_african_wild_dog_juvenile_standturnl180_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedpartials.maniset943f46f0_african_wild_dog_juvenile_walktodrinktroughonspot_keypoints.json.npy"
  "african_wild_dog_juvenile__animationnotmotionextractedpartials.maniset943f46f0_african_wild_dog_juvenile_walktoswim_keypoints.json.npy"
)

process_file() {
  local npy_file="$1"
  local ref_glb="$2"
  local stem="${npy_file%.npy}"
  stem="${stem%_keypoints.json}"

  echo "=== Processing: $stem ==="

  # npy2glb (skeleton)
  python "$CD/npy2glb.py" "$NPY_DIR/$npy_file" --output-dir "$SKEL_OUT" --overwrite 2>&1 | tail -1

  # skel2glb (mesh)
  python "$CD/skel2glb.py" "$NPY_DIR/$npy_file" "$ref_glb" --output "$MESH_OUT/${stem}_mesh.glb" 2>&1 | tail -1
}

echo "Processing 4 aardvark_female files..."
for f in "${AARDVARK_FILES[@]}"; do
  process_file "$f" "$REF_AARDVARK"
done

echo ""
echo "Processing 16 african_wild_dog_juvenile files..."
for f in "${WILDDOG_FILES[@]}"; do
  process_file "$f" "$REF_WILDDOG"
done

echo ""
echo "=== Done ==="
echo "Skeleton GLBs: $SKEL_OUT"
echo "Mesh GLBs: $MESH_OUT"
ls "$SKEL_OUT" | wc -l
ls "$MESH_OUT" | wc -l
