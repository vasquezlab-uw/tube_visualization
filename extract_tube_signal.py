#!/usr/bin/env python
"""Extract raw signal under segmentation masks, one timepoint at a time.

Inputs
  --raw-dir   folder of single-channel, per-timepoint tifs   (e.g. ..._tc1t001.tif)
  --mask-dir  folder of per-timepoint label tifs              (e.g. ..._seg_t001.tif)

Output
  --out-dir   one tif per timepoint: raw intensity where mask > 0 (or where the
              mask is in --labels), 0 elsewhere. Same dtype/shape as the raw frame.
  optional    --stats-csv: per-label, per-timepoint area/mean/median/sum/max

Raw and mask files are paired by the timepoint number captured by each regex
(group 1). If the numbering differs, use --mask-offset
(mask number = raw number + mask offset).

Example
  python extract_masked_signal.py \
      --raw-dir /path/factin_split --mask-dir /path/tube_seg_split \
      --out-dir /path/factin_masked \
      --raw-regex 'c2t(\\d+)\\.tiff?$' --mask-regex 't(\\d+)\\.tiff?$' \
      --labels 16 81 82 259 --stats-csv /path/stats.csv
"""
import argparse
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def index_files(folder, regex):
    """Map timepoint number -> file path for a folder of per-timepoint tifs."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    pattern = re.compile(regex, re.IGNORECASE)
    files = {}
    for f in sorted(folder.iterdir()):
        m = pattern.search(f.name)
        if m:
            n = int(m.group(1))
            if n in files:
                raise ValueError(f"Timepoint {n} matched by two files in {folder}: {files[n].name}, {f.name}")
            files[n] = f
    if not files:
        raise ValueError(f"No files in {folder} matched regex {regex!r}")
    return files


def process_one(job):
    (t, raw_path, mask_path, out_path, labels, want_stats, compression, overwrite) = job
    if out_path.exists() and not overwrite and not want_stats:
        return t, [], "skipped (exists)"

    raw = tifffile.imread(raw_path)
    mask = tifffile.imread(mask_path)
    if raw.shape != mask.shape:
        raise ValueError(f"t={t}: shape mismatch raw {raw.shape} ({raw_path.name}) vs mask {mask.shape} ({mask_path.name})")

    if labels is not None:
        keep = np.isin(mask, labels)
    else:
        keep = mask > 0

    if not (out_path.exists() and not overwrite):
        masked = np.where(keep, raw, 0).astype(raw.dtype)
        tmp = out_path.with_name(out_path.name + ".tmp")
        tifffile.imwrite(tmp, masked, compression=compression)
        os.replace(tmp, out_path)  # atomic, so killed jobs never leave half-written files
        status = "done"
    else:
        status = "skipped (exists)"

    rows = []
    if want_stats:
        present = labels if labels is not None else [int(x) for x in np.unique(mask) if x != 0]
        for lab in present:
            sel = mask == lab
            n = int(sel.sum())
            if n == 0:
                continue
            vals = raw[sel].astype(np.float64)
            rows.append({
                "timepoint": t,
                "label": int(lab),
                "area_px": n,
                "mean": vals.mean(),
                "median": float(np.median(vals)),
                "sum": vals.sum(),
                "max": vals.max(),
            })
    return t, rows, status


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--raw-regex", default=r"t(\d+)\.tiff?$", help="regex with ONE group capturing the timepoint number in raw file names")
    p.add_argument("--mask-regex", default=r"t(\d+)\.tiff?$", help="same, for mask file names")
    p.add_argument("--mask-offset", type=int, default=0, help="mask number = raw number + this")
    p.add_argument("--labels", type=int, nargs="+", default=None, help="only keep these label IDs (default: every label > 0)")
    p.add_argument("--time-range", type=int, nargs=2, metavar=("START", "END"), default=None,
                   help="inclusive range of RAW file numbers to process (default: all that have a matching mask)")
    p.add_argument("--out-suffix", default="_masked", help="appended to the raw file stem")
    p.add_argument("--stats-csv", default=None, help="optional per-label/per-timepoint stats table")
    p.add_argument("--compression", default="zlib", help="tifffile compression, or 'none'")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    raw_files = index_files(args.raw_dir, args.raw_regex)
    mask_files = index_files(args.mask_dir, args.mask_regex)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression.lower() == "none" else args.compression

    numbers = sorted(raw_files)
    if args.time_range:
        a, b = args.time_range
        numbers = [n for n in numbers if a <= n <= b]
    missing = [n for n in numbers if n + args.mask_offset not in mask_files]
    if missing:
        print(f"WARNING: no mask for raw timepoint(s) {missing}; skipping them "
              f"(mask numbers span {min(mask_files)}-{max(mask_files)}, check --mask-offset)", file=sys.stderr)
        numbers = [n for n in numbers if n not in missing]
    if not numbers:
        sys.exit("Nothing to process: no raw timepoints have a matching mask.")

    print(f"{len(numbers)} timepoints ({numbers[0]}-{numbers[-1]}), {args.workers} workers", flush=True)

    jobs = []
    for n in numbers:
        rp = raw_files[n]
        op = out_dir / f"{rp.stem}{args.out_suffix}.tif"
        jobs.append((n, rp, mask_files[n + args.mask_offset], op, args.labels,
                     args.stats_csv is not None, compression, args.overwrite))

    t_start = time.time()
    all_rows = []
    done = 0
    with Pool(min(args.workers, len(jobs))) as pool:
        for t, rows, status in pool.imap_unordered(process_one, jobs):
            all_rows.extend(rows)
            done += 1
            print(f"[{done}/{len(jobs)}] t={t} {status}  ({time.time() - t_start:.0f}s)", flush=True)

    if args.stats_csv:
        stats = pd.DataFrame(all_rows)
        if len(stats):
            stats = stats.sort_values(["timepoint", "label"]).reset_index(drop=True)
        stats.to_csv(args.stats_csv, index=False)
        print(f"Stats written to {args.stats_csv}")
        if args.labels and len(stats):
            gone = sorted(set(args.labels) - set(stats["label"]))
            if gone:
                print(f"WARNING: labels never seen: {gone}")

    print(f"Finished in {time.time() - t_start:.0f}s. Output: {out_dir}")


if __name__ == "__main__":
    main()
