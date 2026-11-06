#!/bin/bash
#SBATCH --job-name=basal_shell
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/mtub/maryk88/logs/output/basal_shell_%j.out
#SBATCH --error=/gpfs/projects/mtub/maryk88/logs/error/basal_shell_%j.err

# Adjust paths, account/partition, and environment for your cluster.
module load conda
source ~/.bashrc
conda activate tubeseg

# ---------------------------------------------------------------------
# Input and output directories
# ---------------------------------------------------------------------
RAW=/gpfs/projects/mtub/maryk88/261013_sqhutrgfp_ctbpalmmkate_live/261013_sqhutrgfp_ctbpalmmkate_live_001_mem_denoised_clahe
MASK=/gpfs/projects/mtub/maryk88/261013_sqhutrgfp_ctbpalmmkate_live/tube_extraction/1_tube_seg_all
SHELL_OUT=/gpfs/projects/mtub/maryk88/261013_sqhutrgfp_ctbpalmmkate_live/tube_extraction/5_membrane_shell
INTERIOR_OUT=/gpfs/projects/mtub/maryk88/261013_sqhutrgfp_ctbpalmmkate_live/tube_extraction/6_membrane_interior

# ---------------------------------------------------------------------
# File matching
# Regex must contain ONE capture group for the timepoint number when
# pairing multiple files. If there is one static mask, a no-group regex
# matching that single file is also accepted.
# ---------------------------------------------------------------------
RAW_REGEX='c2t(\d+)\.tif?$'
MASK_REGEX='c2t(\d+)_tube_final\.tiff?$'
MASK_MODE='paired'       # 'auto', 'single' (use same mask for all tps), or 'paired' (use matching tube seg for each tp)
MASK_OFFSET=0          # paired mode: mask number = raw number + offset

# Voxel spacing in micrometers, in NumPy axis order (Z Y X for 3D).
VOXEL_SPACING=(0.4374 0.2398 0.2398)

# Thickness removed from the outside of the tube to define the shell in um.
SHELL_THICKNESS=2.0

# Optional inclusive raw timepoint range; leave empty for all timepoints.
# TIME_RANGE=(102 241)
TIME_RANGE=()

CMD=(
    python extract_shell_and_interior_signal.py
    --raw-dir "$RAW"
    --mask-dir "$MASK"
    --shell-out-dir "$SHELL_OUT"
    --interior-out-dir "$INTERIOR_OUT"
    --raw-regex "$RAW_REGEX"
    --mask-regex "$MASK_REGEX"
    --mask-mode "$MASK_MODE"
    --mask-offset "$MASK_OFFSET"
    --voxel-spacing "${VOXEL_SPACING[@]}"
    --shell-thickness "$SHELL_THICKNESS"
    --workers "$SLURM_CPUS_PER_TASK"
)

if ((${#TIME_RANGE[@]})); then
    CMD+=(--time-range "${TIME_RANGE[@]}")
fi

# Add --overwrite to CMD to replace existing output TIFFs.
"${CMD[@]}"
