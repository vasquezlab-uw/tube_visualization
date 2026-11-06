#!/usr/bin/env python
"""Extract raw signal in the basal shell of tube segmentation masks.

For each raw timepoint, the script:
  1. Converts the tube segmentation to a binary mask (all values > 0).
  2. Computes the Euclidean distance, in micrometers, from each tube voxel
     to the nearest background voxel using scipy.ndimage.distance_transform_edt.
  3. Defines the basal shell as tube voxels whose interior distance is
     <= --shell-thickness.
  4. Defines the eroded tube interior as tube voxels whose interior distance
     is > --shell-thickness.
  5. Writes separate raw-signal TIFFs for the basal shell and eroded interior,
     with 0 everywhere outside each respective region.

Voxel spacing is required and must be supplied in NumPy array-axis order:
Z Y X for 3D images, or Y X for 2D images. The notebook this script replaces
uses 3D data and (Z, Y, X) spacing.

Mask handling:
  --mask-mode auto (default): reuse the mask if exactly one mask file matches;
      otherwise pair masks and raw images by their captured timepoint numbers.
  --mask-mode single: require exactly one matching mask and reuse it for all
      raw timepoints.
  --mask-mode paired: pair raw and mask files by timepoint number.

If paired numbering differs, use --mask-offset (mask number = raw number +
mask offset).
"""
import argparse
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi


def index_files(folder, regex):
    """Map captured timepoint number -> TIFF path."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    pattern = re.compile(regex, re.IGNORECASE)
    files = {}
    for f in sorted(folder.iterdir()):
        # Ignore hidden files (including macOS AppleDouble files such as ._name.tif).
        if f.name.startswith(".") or not f.is_file():
            continue
        match = pattern.search(f.name)
        if match:
            if len(match.groups()) > 1:
                raise ValueError(f"Regex may contain at most one capture group: {regex!r}")
            if len(match.groups()) == 0:
                if files:
                    raise ValueError(
                        f"Regex {regex!r} must contain one capture group when matching "
                        "multiple files; otherwise use a pattern matching exactly one mask"
                    )
                number = 0
            else:
                number = int(match.group(1))
            if number in files:
                raise ValueError(
                    f"Timepoint {number} matched by two files in {folder}: "
                    f"{files[number].name}, {f.name}"
                )
            files[number] = f
    if not files:
        raise ValueError(f"No files in {folder} matched regex {regex!r}")
    return files


def process_one(job):
    (timepoint, raw_path, mask_path, shell_out_path, interior_out_path,
     voxel_spacing, shell_thickness, compression, overwrite) = job

    shell_exists = shell_out_path.exists()
    interior_exists = interior_out_path.exists()
    if shell_exists and interior_exists and not overwrite:
        return timepoint, "skipped (both outputs exist)", None

    raw = tifffile.imread(raw_path)
    tube_seg = tifffile.imread(mask_path)

    if raw.ndim not in (2, 3):
        raise ValueError(
            f"t={timepoint}: expected a 2D or 3D raw image, got shape {raw.shape}"
        )
    if len(voxel_spacing) != raw.ndim:
        raise ValueError(
            f"t={timepoint}: image has {raw.ndim} dimensions but "
            f"--voxel-spacing has {len(voxel_spacing)} values; supply spacing "
            "in array-axis order (2D: Y X; 3D: Z Y X)"
        )
    if raw.shape != tube_seg.shape:
        raise ValueError(
            f"t={timepoint}: shape mismatch raw {raw.shape} ({raw_path.name}) "
            f"vs tube mask {tube_seg.shape} ({mask_path.name})"
        )

    tube_mask = tube_seg > 0
    if not np.any(tube_mask):
        shell_mask = np.zeros(tube_mask.shape, dtype=bool)
    else:
        distance_inside = ndi.distance_transform_edt(
            tube_mask,
            sampling=voxel_spacing,
        )
        shell_mask = tube_mask & (distance_inside <= shell_thickness)

    # The interior is the remaining tube volume after removing the shell.
    # The strict > boundary makes the two regions mutually exclusive.
    interior_mask = tube_mask & (distance_inside > shell_thickness) if np.any(tube_mask) else tube_mask

    shell_signal = np.where(shell_mask, raw, 0).astype(raw.dtype, copy=False)
    interior_signal = np.where(interior_mask, raw, 0).astype(raw.dtype, copy=False)

    def write_tiff_atomic(out_path, image):
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        try:
            tifffile.imwrite(tmp_path, image, compression=compression)
            os.replace(tmp_path, out_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    if overwrite or not shell_out_path.exists():
        write_tiff_atomic(shell_out_path, shell_signal)
    if overwrite or not interior_out_path.exists():
        write_tiff_atomic(interior_out_path, interior_signal)

    return timepoint, "done", (
        int(tube_mask.sum()),
        int(shell_mask.sum()),
        int(interior_mask.sum()),
        mask_path.name,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw-dir", required=True,
                        help="directory containing split raw timepoint TIFFs")
    parser.add_argument("--mask-dir", required=True,
                        help="directory containing tube segmentation mask TIFF(s)")
    parser.add_argument("--shell-out-dir", required=True,
                        help="output directory for raw signal inside the basal shell")
    parser.add_argument("--interior-out-dir", required=True,
                        help="output directory for raw signal in the tube interior after shell erosion")
    parser.add_argument(
        "--raw-regex", default=r"t(\d+)\.tiff?$",
        help="regex with ONE capture group for raw timepoint number",
    )
    parser.add_argument(
        "--mask-regex", default=r"t(\d+).*\.tiff?$",
        help="regex with ONE capture group for mask timepoint number",
    )
    parser.add_argument("--mask-offset", type=int, default=0,
                        help="paired mode: mask number = raw number + offset")
    parser.add_argument(
        "--mask-mode", choices=("auto", "single", "paired"), default="auto",
        help=("auto reuses one mask if only one is found, otherwise pairs by "
              "timepoint; single reuses one mask; paired pairs by timepoint"),
    )
    parser.add_argument(
        "--voxel-spacing", type=float, nargs="+", required=True, metavar="UM",
        help="voxel spacing in micrometers in array-axis order (3D: Z Y X; 2D: Y X)",
    )
    parser.add_argument(
        "--shell-thickness", type=float, default=1.0, metavar="UM",
        help="basal shell thickness in micrometers (default: 1.0)",
    )
    parser.add_argument("--time-range", type=int, nargs=2, metavar=("START", "END"),
                        default=None, help="inclusive raw timepoint range")
    parser.add_argument(
        "--shell-out-suffix", default=None,
        help="suffix for basal-shell TIFFs (default uses shell thickness)",
    )
    parser.add_argument(
        "--interior-out-suffix", default=None,
        help="suffix for eroded-interior TIFFs (default uses shell thickness)",
    )
    parser.add_argument("--compression", default="zlib",
                        help="TIFF compression, or 'none' (default: zlib)")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if any(value <= 0 for value in args.voxel_spacing):
        parser.error("--voxel-spacing values must all be positive")
    if args.shell_thickness <= 0:
        parser.error("--shell-thickness must be positive")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    raw_files = index_files(args.raw_dir, args.raw_regex)
    mask_files = index_files(args.mask_dir, args.mask_regex)

    mode = args.mask_mode
    if mode == "auto":
        mode = "single" if len(mask_files) == 1 else "paired"
    if mode == "single" and len(mask_files) != 1:
        parser.error(
            f"--mask-mode single requires exactly one matching mask; found {len(mask_files)}"
        )

    numbers = sorted(raw_files)
    if args.time_range:
        start, end = args.time_range
        if start > end:
            parser.error("--time-range START must be <= END")
        numbers = [n for n in numbers if start <= n <= end]
    if not numbers:
        sys.exit("Nothing to process: no raw timepoints selected.")

    if mode == "paired":
        missing = [n for n in numbers if n + args.mask_offset not in mask_files]
        if missing:
            raise SystemExit(
                f"No matching tube mask for raw timepoint(s) {missing}. "
                f"Mask numbers span {min(mask_files)}-{max(mask_files)}; "
                "check --mask-regex or --mask-offset."
            )

    shell_out_dir = Path(args.shell_out_dir)
    interior_out_dir = Path(args.interior_out_dir)
    shell_out_dir.mkdir(parents=True, exist_ok=True)
    interior_out_dir.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression.lower() == "none" else args.compression

    thickness_tag = f"{args.shell_thickness:g}".replace(".", "p")
    shell_suffix = args.shell_out_suffix or f"_basal_shell_{thickness_tag}um"
    interior_suffix = args.interior_out_suffix or f"_tube_interior_eroded_{thickness_tag}um"

    spacing = tuple(args.voxel_spacing)
    jobs = []
    for number in numbers:
        raw_path = raw_files[number]
        mask_path = next(iter(mask_files.values())) if mode == "single" else mask_files[number + args.mask_offset]
        shell_out_path = shell_out_dir / f"{raw_path.stem}{shell_suffix}.tif"
        interior_out_path = interior_out_dir / f"{raw_path.stem}{interior_suffix}.tif"
        jobs.append((
            number, raw_path, mask_path, shell_out_path, interior_out_path, spacing,
            args.shell_thickness, compression, args.overwrite,
        ))

    print(f"Raw timepoints: {len(jobs)} ({numbers[0]}-{numbers[-1]})", flush=True)
    print(f"Mask mode: {mode}", flush=True)
    print(f"Voxel spacing (um; array-axis order): {spacing}", flush=True)
    print(f"Basal shell thickness: {args.shell_thickness:g} um", flush=True)
    print(f"Shell output: {shell_out_dir}", flush=True)
    print(f"Eroded interior output: {interior_out_dir}", flush=True)
    print(f"Workers: {min(args.workers, len(jobs))}", flush=True)

    start_time = time.time()
    completed = 0
    with Pool(min(args.workers, len(jobs))) as pool:
        for number, status, details in pool.imap_unordered(process_one, jobs):
            completed += 1
            if details is None:
                print(f"[{completed}/{len(jobs)}] t={number}: {status}", flush=True)
            else:
                tube_voxels, shell_voxels, interior_voxels, mask_name = details
                print(
                    f"[{completed}/{len(jobs)}] t={number}: {status}; "
                    f"tube={tube_voxels:,} voxels, shell={shell_voxels:,}, "
                    f"interior={interior_voxels:,}; mask={mask_name} "
                    f"({time.time() - start_time:.0f}s)",
                    flush=True,
                )

    print(f"Finished in {time.time() - start_time:.0f}s.", flush=True)
    print(f"Basal-shell TIFFs: {shell_out_dir}", flush=True)
    print(f"Eroded-interior TIFFs: {interior_out_dir}", flush=True)


if __name__ == "__main__":
    main()
