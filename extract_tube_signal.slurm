#!/bin/bash
#SBATCH --job-name=extract_masked
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/projects/mtub/maryk88/logs/output/extract_%j.out
#SBATCH --error=/gpfs/projects/mtub/maryk88/logs/error/extract_%j.err 

# Adjust partition/account and the environment line for your cluster.
# source ~/.bashrc && conda activate napari   # needs numpy pandas tifffile (and zarr>=3 for tiff_to_zarr.py)
source ~/.bashrc
conda activate tubeseg


RAW=/gpfs/projects/mtub/maryk88/261013_sqhutrgfp_ctbpalmmkate_live/261013_sqhutrgfp_ctbpalmmkate_live_001_factin_split_timepoints
MASK=/gpfs/projects/mtub/maryk88/261013_sqhutrgfp_ctbpalmmkate_live/tube_extraction/1_tube_seg_all
OUT=/gpfs/projects/mtub/maryk88/261013_sqhutrgfp_ctbpalmmkate_live/tube_extraction/2_factin_extracted

python extract_tube_signal.py \
    --raw-dir  "$RAW" \
    --mask-dir "$MASK" \
    --out-dir  "$OUT" \
    --raw-regex  'c1t(\d+)\.tif?$' \
    --mask-regex 'c2t(\d+)_tube_final\.tiff?$' \
    --stats-csv "$OUT/stats.csv" \
    --workers "$SLURM_CPUS_PER_TASK"

# Optional: also stack the result into a single zarr
# python tiff_to_zarr.py --in-dir "$OUT" --out "$OUT.zarr" --regex 't(\d+)_masked\.tiff?$' --workers "$SLURM_CPUS_PER_TASK"
