#!/bin/bash
# =============================================================================
# Tube segmentation / ghost-cell filling, one SLURM array task per timepoint.
# Supports TIFF or per-timepoint Zarr raw images; TIFF or Zarr cell masks;
# and waypoint CSV or saved centerline NPZ inputs.
#
# Usage:
#   bash run_tube_seg.sh                 # submit SLURM array
#   bash run_tube_seg.sh --local 0       # run sorted image index 0 locally
#
# Edit CONFIG only.
# =============================================================================

# ------------------------------- CONFIG --------------------------------------
# Define log directory
LOG_DIR="/gpfs/projects/mtub/maryk88/logs"
LOG_PREFIX="${LOG_PREFIX:-tube_$(date +%Y%m%d_%H%M%S)}"


# Membrane channel input: can be split timepoint tiff or zarr
IMG_DIR="/gpfs/projects/mtub/maryk88/260729_sqhutrgfp_ctbpalmmkate_live/260729_membrane_split_timepoints"
IMG_GLOB="*.tif"
ZARR_IMG_ARRAY=""
TIMEPOINT_REGEX='tc2t([0-9]+)$'       # change to match your naming convention
FIRST_TP="4"                        # leave these blank to run all tps
LAST_TP="4"


# Cell seg input: choose one
# Split timepoint tiffs
SEG_DIR=""             

# Zarr file
SEG_ALL_ZARR="/gpfs/projects/mtub/maryk88/260729_sqhutrgfp_ctbpalmmkate_live/260729_overlap8_relabeled_seg.zarr"
ZARR_SEG_ARRAY="data"               # Zarr array name; empty means the .zarr itself is an array. 
SEG_TIME_AXIS=0
SEG_TIME_FIRST_ID=1             # corrects for 0-based index of seg compared to 1-based index of raw images, set to 0 if the first tp index matches between datasets

# For either option, define file suffix (e.g. _masks.tif or _masks.zarr)
SEG_SUFFIX=""


# Centerline input:
CENTERLINE_SOURCE="centerlines"   # "waypoints" CSVs or "centerlines" NPZs
WAYPOINTS_DIR="/path/to/waypoints"
CENTERLINES_DIR="/gpfs/projects/mtub/maryk88/260729_sqhutrgfp_ctbpalmmkate_live/260729_centerlines_T2-100"
CENTERLINE_FILE_KEY=""          # empty = image stem; otherwise unique substring
CENTERLINE_PATTERN="T{tp}.npz"  # used for centerline mode only; if no exact pattern match, substring matching is used


OUT_DIR="/gpfs/projects/mtub/maryk88/260729_sqhutrgfp_ctbpalmmkate_live/tube_extraction/1_tube_seg"
VOXEL_SPACING="0.4374 0.2158 0.2158"   # "z y x" in µm

# Pipeline parameters (radii are z y x, µm)
CLOSING_RADIUS="7.5 3 3"
BIG_CLOSING_RADIUS_UM=10.0
IMAGE_GATE_PERCENTILE=10
S_JUMP_THRESHOLD=20
S_FAR_THRESHOLD=1e9
CELL_CONSENSUS_MIN_SUPPORT=0.55
CELL_CONSENSUS_SECOND_PASS_SUPPORT=0.55
NO_FILL_RADIUS="0 0 0"
MIN_SIZE=50
TRIM_RADIUS="none"              # "none" to trim self-contact voxels | "z y x" in um
FILL_HOLES=1
MAX_HOLE_UM3=300
SPUR_RADIUS="1.3 0.9 0.9"       # "none" for no trimming | "z y x" in µm
SMOOTH_SIGMA_UM=0.7
WAYPOINT_SMOOTH_UM=0.0
CENTERLINE_SPACING_UM=0.5

# SLURM / execution
JOB_NAME="tube_seg"
PARTITION=""
QOS="normal"
MAIL_TYPE=""
MAIL_USER=""
TIME="01:00:00"
CPUS=4
MEM="32G"
MAX_CONCURRENT=20
CONDA_ENV="/gpfs/home/maryk88/.conda/envs/tubeseg"
PYTHON="${CONDA_ENV}/bin/python"
SETUP_CMD=""
# ----------------------------- END CONFIG ------------------------------------

set -euo pipefail
SCRIPT_PATH="$(readlink -f "$0")"

centerline_match_count() {
    local image_path="$1" stem key cdir pattern
    stem="${image_path##*/}"; stem="${stem%.*}"
    key="${CENTERLINE_FILE_KEY:-$stem}"
    cdir="$WAYPOINTS_DIR"
    pattern="*.csv"
    if [[ "$CENTERLINE_SOURCE" == "centerlines" ]]; then
        cdir="$CENTERLINES_DIR"
        pattern="*.npz"
    fi
    find "$cdir" -maxdepth 1 -type f -name "$pattern" -printf '%f\n' 2>/dev/null \
        | awk -v key="$key" 'index($0,key)>0 {n++} END {print n+0}'
}

list_images() {
    local f stem id n
    shopt -s nocasematch
    while IFS= read -r f; do
        stem="${f##*/}"; stem="${stem%.*}"
        [[ "$stem" =~ $TIMEPOINT_REGEX ]] || continue
        id=$((10#${BASH_REMATCH[1]}))
        [[ -n "$FIRST_TP" && $id -lt $FIRST_TP ]] && continue
        [[ -n "$LAST_TP" && $id -gt $LAST_TP ]] && continue
        if [[ "$CENTERLINE_SOURCE" == "waypoints" ]]; then
            n="$(centerline_match_count "$f")"
            [[ "$n" -eq 1 ]] || continue
        fi
        echo "$f"
    done < <(find "$IMG_DIR" -maxdepth 1 -type f \
        -name "$IMG_GLOB" ! -name '.*' ! -name '._*' | sort)
    shopt -u nocasematch
}

list_skipped_centerlines() {
    local f stem id n
    shopt -s nocasematch
    while IFS= read -r f; do
        stem="${f##*/}"; stem="${stem%.*}"
        [[ "$stem" =~ $TIMEPOINT_REGEX ]] || continue
        id=$((10#${BASH_REMATCH[1]}))
        [[ -n "$FIRST_TP" && $id -lt $FIRST_TP ]] && continue
        [[ -n "$LAST_TP" && $id -gt $LAST_TP ]] && continue
        if [[ "$CENTERLINE_SOURCE" == "waypoints" ]]; then
            n="$(centerline_match_count "$f")"
            [[ "$n" -eq 1 ]] || echo "SKIP timepoint $id: expected one waypoint CSV for '$stem'; found $n"
        fi
    done < <(find "$IMG_DIR" -maxdepth 1 -type f \
        -name "$IMG_GLOB" ! -name '.*' ! -name '._*' | sort)
    shopt -u nocasematch
}

if [[ -z "${SLURM_JOB_ID:-}" && "${1:-}" != "--local" ]]; then
    N=$(list_images | wc -l)
    [[ "$N" -gt 0 ]] || { echo "No input images found in $IMG_DIR" >&2; exit 1; }
    mkdir -p "$OUT_DIR" "$LOG_DIR/output" "$LOG_DIR/error"
    # Slurm's own per-task files are disabled: each task appends to the two
    # shared logs configured below.
    ARGS=(--job-name="$JOB_NAME" --array="0-$((N-1))%${MAX_CONCURRENT}"
          --time="$TIME" --cpus-per-task="$CPUS" --mem="$MEM"
          --output=/dev/null --error=/dev/null)
    [[ -n "$PARTITION" ]] && ARGS+=(--partition="$PARTITION")
    [[ -n "$QOS" ]] && ARGS+=(--qos="$QOS")
    [[ -n "$MAIL_TYPE" ]] && ARGS+=(--mail-type="$MAIL_TYPE" --mail-user="$MAIL_USER")
    echo "Submitting $N timepoint(s) with matching centerlines"
    echo "Shared logs: $LOG_DIR/output/${LOG_PREFIX}.log and $LOG_DIR/error/${LOG_PREFIX}.log"
    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Selected $N timepoint(s) with matching centerlines"
        list_skipped_centerlines
    } >> "$LOG_DIR/output/${LOG_PREFIX}.log"
    sbatch --export="ALL,LOG_PREFIX=${LOG_PREFIX}" "${ARGS[@]}" "$SCRIPT_PATH"
    exit 0
fi

if [[ "${1:-}" == "--local" ]]; then
    IDX="${2:?give sorted image index after --local}"
else
    IDX="${SLURM_ARRAY_TASK_ID:?expected SLURM array task}"
fi

mkdir -p "$OUT_DIR" "$LOG_DIR/output" "$LOG_DIR/error"
OUTPUT_LOG="$LOG_DIR/output/${LOG_PREFIX}.log"
ERROR_LOG="$LOG_DIR/error/${LOG_PREFIX}.log"
# Append every timepoint to one shared output log and one shared error log.
# The task header makes interleaved array-task output identifiable.
exec >> "$OUTPUT_LOG" 2>> "$ERROR_LOG"
TASK_LABEL="local-index-${IDX}"
[[ -n "${SLURM_ARRAY_JOB_ID:-}" ]] && TASK_LABEL="array-${SLURM_ARRAY_JOB_ID}-task-${IDX}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${TASK_LABEL}"
if [[ "${1:-}" == "--local" ]]; then
    list_skipped_centerlines
fi
trap 'status=$?; echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR ${TASK_LABEL} (exit ${status}, line ${LINENO})" >&2' ERR

[[ -n "$SETUP_CMD" ]] && eval "$SETUP_CMD"
# Use the interpreter inside the selected environment directly. This avoids
# relying on Conda initialization in non-interactive SLURM shells.
if [[ -n "$CONDA_ENV" && "$PYTHON" == "python" ]]; then
    PYTHON="$CONDA_ENV/bin/python"
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found or not executable: $PYTHON" >&2
    exit 1
fi
echo "Using Python: $PYTHON"
"$PYTHON" --version

IMG_PATH="$(list_images | sed -n "$((IDX+1))p")"
[[ -n "$IMG_PATH" ]] || { echo "No image at sorted index $IDX" >&2; exit 1; }

export IMG_PATH SEG_DIR OUT_DIR TIMEPOINT_REGEX SEG_SUFFIX SEG_ALL_ZARR SEG_TIME_AXIS SEG_TIME_FIRST_ID
export CENTERLINE_SOURCE WAYPOINTS_DIR CENTERLINES_DIR CENTERLINE_FILE_KEY CENTERLINE_PATTERN
export WAYPOINT_SMOOTH_UM CENTERLINE_SPACING_UM ZARR_IMG_ARRAY ZARR_SEG_ARRAY VOXEL_SPACING
export CLOSING_RADIUS BIG_CLOSING_RADIUS_UM IMAGE_GATE_PERCENTILE S_JUMP_THRESHOLD S_FAR_THRESHOLD
export CELL_CONSENSUS_MIN_SUPPORT CELL_CONSENSUS_SECOND_PASS_SUPPORT NO_FILL_RADIUS MIN_SIZE
export TRIM_RADIUS FILL_HOLES MAX_HOLE_UM3 SPUR_RADIUS SMOOTH_SIGMA_UM PYTHON

"$PYTHON" - <<'PYEOF'
import os, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from skimage.segmentation import watershed
from skimage.morphology import remove_small_holes

E = os.environ
def floats(key): return tuple(float(x) for x in E[key].split())
def optional_floats(key):
    return None if E[key].strip().lower() == "none" else floats(key)

voxel_spacing = floats("VOXEL_SPACING")
closing_radius = floats("CLOSING_RADIUS")
no_fill_radius = floats("NO_FILL_RADIUS")
trim_radius = optional_floats("TRIM_RADIUS")
spur_radius = optional_floats("SPUR_RADIUS")
s_jump_threshold = float(E["S_JUMP_THRESHOLD"])
s_far_threshold = float(E["S_FAR_THRESHOLD"])
min_size = int(E["MIN_SIZE"])
fill_holes = E["FILL_HOLES"] == "1"
max_hole_um3 = float(E["MAX_HOLE_UM3"])
base = Path(E["IMG_PATH"]).name
for ext in (".tiff", ".tif", ".zarr"):
    if base.lower().endswith(ext):
        base = base[:-len(ext)]
        break

m = re.search(E["TIMEPOINT_REGEX"], base, re.IGNORECASE)
if not m: sys.exit(f"TIMEPOINT_REGEX did not match {base!r}")
tp = int(m.group(1))
img_path = E["IMG_PATH"]

def is_zarr(path): return str(path).rstrip("/").lower().endswith(".zarr")
def open_zarr(path, key):
    try: import zarr
    except ImportError: sys.exit("Install zarr in the selected Python environment.")
    node = zarr.open(path, mode="r")
    if key: node = node[key]
    if not hasattr(node, "shape"):
        sys.exit(f"{path} is a Zarr group; set the appropriate ZARR_*_ARRAY. Keys: {list(node.keys())}")
    return node
def read_volume(path, key=""):
    if is_zarr(path): return np.squeeze(np.asarray(open_zarr(path, key)[...]))
    return np.squeeze(tifffile.imread(path))
def read_shape(path, key=""):
    if is_zarr(path): shape = open_zarr(path, key).shape
    else:
        with tifffile.TiffFile(path) as tf: shape = tf.series[0].shape
    return tuple(d for d in shape if d != 1)

# Raw image
img = read_volume(img_path, E["ZARR_IMG_ARRAY"])
if img.ndim != 3: sys.exit(f"Expected 3D raw image after squeeze; got {img.shape}")

# Cell segmentation: per-timepoint file or a single time-series Zarr.
seg_all = E["SEG_ALL_ZARR"]
if seg_all:
    arr = open_zarr(seg_all, E["ZARR_SEG_ARRAY"])
    axis = int(E["SEG_TIME_AXIS"])
    idx = tp - int(E["SEG_TIME_FIRST_ID"])
    if arr.ndim != 4: sys.exit(f"Expected 4D time+ZYX segmentation Zarr, got {arr.shape}")
    if not 0 <= idx < arr.shape[axis]: sys.exit(f"Timepoint {tp} maps outside segmentation time axis: index {idx}")
    sl = [slice(None)] * 4; sl[axis] = idx
    seg = np.squeeze(np.asarray(arr[tuple(sl)]))
    seg_path = seg_all
else:
    seg_path = str(Path(E["SEG_DIR"]) / (base + E["SEG_SUFFIX"]))
    if not os.path.exists(seg_path): sys.exit(f"Segmentation not found: {seg_path}")
    seg = read_volume(seg_path, E["ZARR_SEG_ARRAY"])
if seg.ndim != 3: sys.exit(f"Expected 3D cell mask; got {seg.shape}")
if img.shape != seg.shape: sys.exit(f"Image/mask shapes differ: {img.shape} vs {seg.shape}")
print(f"Timepoint {tp}\\n image: {img_path}\\n segmentation: {seg_path}", flush=True)

# Centerline: NPZ stores physical spline coordinates; CSV stores voxel waypoints.
source = E["CENTERLINE_SOURCE"].lower()
if source not in ("waypoints", "centerlines"): sys.exit("CENTERLINE_SOURCE must be waypoints or centerlines")
key = E["CENTERLINE_FILE_KEY"].strip()
cdir = Path(E["WAYPOINTS_DIR"] if source == "waypoints" else E["CENTERLINES_DIR"])
if source == "centerlines":
    # NPZ centerlines are named by timepoint (e.g. T005.npz), whereas raw
    # image stems may contain a long prefix (e.g. ..._crop_c2t005.tif).
    # Match the numeric timepoint, not the complete image filename.
    pattern = E["CENTERLINE_PATTERN"].strip()
    candidate = None
    if pattern:
        try:
            candidate = cdir / pattern.format(tp=tp, tp03=f"{tp:03d}", tp_padded=f"{tp:03d}")
        except (KeyError, ValueError) as exc:
            sys.exit(f"Invalid CENTERLINE_PATTERN {pattern!r}: {exc}")
    if candidate is not None and candidate.is_file():
        matches = [candidate]
    elif key:
        matches = sorted(p for p in cdir.glob("*.npz") if key in p.name)
    else:
        matches = []
        for p in sorted(cdir.glob("*.npz")):
            m_tp = re.fullmatch(r"[Tt]0*(\d+)\.npz", p.name)
            if m_tp and int(m_tp.group(1)) == tp:
                matches.append(p)
else:
    key = key or base
    matches = sorted(p for p in cdir.glob("*.csv") if key in p.name)
if len(matches) != 1:
    sys.exit(f"Expected exactly one {source} file for timepoint {tp} in {cdir}; found: {[str(p) for p in matches]}")
centerline_path = matches[0]
if source == "centerlines":
    with np.load(centerline_path) as d:
        if "spline" not in d: sys.exit(f"{centerline_path} has no 'spline' array; keys={list(d.keys())}")
        spline = np.asarray(d["spline"], float)
else:
    df = pd.read_csv(centerline_path)
    cols = ["axis-0", "axis-1", "axis-2"]
    if not set(cols).issubset(df.columns): sys.exit(f"Waypoint CSV needs columns {cols}")
    if "index" in df: df = df.sort_values("index")
    wps = df[cols].to_numpy(float) * np.asarray(voxel_spacing)
    if len(wps) < 2: sys.exit("At least two waypoints are required")
    lengths = np.linalg.norm(np.diff(wps, axis=0), axis=1)
    u = np.r_[0, np.cumsum(lengths)]
    if u[-1] <= 0: sys.exit("Waypoint coordinates have zero length")
    u /= u[-1]
    tck, _ = splprep(wps.T, u=u, s=len(wps)*float(E["WAYPOINT_SMOOTH_UM"])**2, k=min(3,len(wps)-1))
    npts = max(2, int(np.ceil(u[-1] * 0 + lengths.sum()/float(E["CENTERLINE_SPACING_UM"])))+1)
    spline = np.asarray(splev(np.linspace(0,1,npts), tck)).T
if spline.ndim != 2 or spline.shape[1] != 3 or len(spline) < 2: sys.exit(f"Invalid spline shape {spline.shape}")
s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(spline, axis=0), axis=1))]
centerline_tree = cKDTree(spline)
print(f"Centerline: {centerline_path}; length={s[-1]:.2f} um")


def ellipsoid_footprint(radius_um, spacing):
    rz, ry, rx = [int(np.ceil(r/d)) for r,d in zip(radius_um,spacing)]
    z,y,x = np.ogrid[-rz:rz+1,-ry:ry+1,-rx:rx+1]
    return ((z*spacing[0]/radius_um[0])**2+(y*spacing[1]/radius_um[1])**2+(x*spacing[2]/radius_um[2])**2)<=1

def assign_s(mask, existing=None):
    out = np.full(mask.shape,np.nan,np.float32) if existing is None else existing.copy()
    coords = np.argwhere(mask)
    if len(coords):
        _, nearest = centerline_tree.query(coords*np.asarray(voxel_spacing))
        out[tuple(coords.T)] = s[nearest]
    return out
def find_self_contacts(mask, smap, threshold):
    out=np.zeros(mask.shape,bool)
    for ax in range(3):
        a=[slice(None)]*3; b=[slice(None)]*3
        a[ax]=slice(None,-1); b[ax]=slice(1,None); a=tuple(a); b=tuple(b)
        bad=mask[a]&mask[b]&(np.abs(smap[a]-smap[b])>threshold)
        out[a]|=bad; out[b]|=bad
    return out
def filter_small(mask, minimum):
    lab,n=ndi.label(mask); sizes=np.bincount(lab.ravel()); sizes[0]=0
    return sizes[lab]>=minimum

# Keep largest connected component of original cell union.
tube_raw=seg>0
lab,n=ndi.label(tube_raw); sizes=np.bincount(lab.ravel()); sizes[0]=0
tube_seg=lab==sizes.argmax()
seg=np.where(tube_seg,seg,0)
if not tube_seg.any(): sys.exit("Cell segmentation is empty")
vox=float(np.prod(voxel_spacing))
print(f"Removed {n-1} disconnected component(s)")

# Cell-level s correction using shared physical surface area.
vox_s_nearest=assign_s(tube_seg)
ids=np.unique(seg[tube_seg & (seg>0)])
original_s={int(cid):float(np.nanmedian(vox_s_nearest[seg==cid])) for cid in ids}
spacing=np.asarray(voxel_spacing)
face_area={0:spacing[1]*spacing[2],1:spacing[0]*spacing[2],2:spacing[0]*spacing[1]}
shared=defaultdict(float)
for ax in range(3):
    a=[slice(None)]*3;b=[slice(None)]*3;a[ax]=slice(None,-1);b[ax]=slice(1,None)
    aa=seg[tuple(a)];bb=seg[tuple(b)];ok=(aa>0)&(bb>0)&(aa!=bb)
    if ok.any():
        pairs=np.stack((aa[ok],bb[ok]),axis=1);pairs.sort(axis=1)
        up,cnt=np.unique(pairs,axis=0,return_counts=True)
        for (x,y),c in zip(up,cnt): shared[(int(x),int(y))]+=float(c)*face_area[ax]
neighbors=defaultdict(list)
for (a,b),area in shared.items(): neighbors[a].append((b,area));neighbors[b].append((a,area))
def wmedian(vals,weights):
    vals=np.asarray(vals);weights=np.asarray(weights);o=np.argsort(vals);v=vals[o];w=weights[o]
    return float(v[min(np.searchsorted(np.cumsum(w),w.sum()/2),len(v)-1)])
consensus={};support={}
for cid in original_s:
    pairs=[(original_s[o],ar) for o,ar in neighbors[cid] if o in original_s and ar>0]
    if not pairs: consensus[cid]=np.nan;support[cid]=0.;continue
    vv=np.array([x for x,_ in pairs]);ww=np.array([a for _,a in pairs])
    target=wmedian(vv,ww);consensus[cid]=target
    support[cid]=float(ww[np.abs(vv-target)<=float(E["S_JUMP_THRESHOLD"])].sum()/ww.sum())
first=original_s.copy()
for cid in original_s:
    if np.isfinite(consensus[cid]) and abs(original_s[cid]-consensus[cid])>=float(E["S_JUMP_THRESHOLD"]) and support[cid]>=float(E["CELL_CONSENSUS_MIN_SUPPORT"]):
        first[cid]=consensus[cid]
second=first.copy()
for cid in original_s:
    if first[cid] != original_s[cid] or not np.isfinite(consensus[cid]): continue
    pairs=[(first[o],ar) for o,ar in neighbors[cid] if o in first and ar>0]
    if not pairs: continue
    vv=np.array([x for x,_ in pairs]);ww=np.array([a for _,a in pairs])
    target=wmedian(vv,ww);sup=ww[np.abs(vv-target)<=float(E["S_JUMP_THRESHOLD"])].sum()/ww.sum()
    if abs(original_s[cid]-target)>=float(E["S_JUMP_THRESHOLD"]) and sup>=float(E["CELL_CONSENSUS_SECOND_PASS_SUPPORT"]):
        second[cid]=target
tube_s_map=vox_s_nearest.copy()
for cid,val in second.items(): tube_s_map[seg==cid]=val
self_contact_mask=find_self_contacts(tube_seg,tube_s_map,s_jump_threshold)
print(f"Self-contact voxels: {self_contact_mask.sum():,}")

# Candidate fill.
closing=ellipsoid_footprint(closing_radius,voxel_spacing)
candidate=ndi.binary_closing(tube_seg,structure=closing)&~tube_seg
big_r=float(E["BIG_CLOSING_RADIUS_UM"])
if big_r>0:
    dil=ndi.distance_transform_edt(~tube_seg,sampling=voxel_spacing)<=big_r
    big_closed=ndi.distance_transform_edt(dil,sampling=voxel_spacing)>big_r
    img_s=ndi.gaussian_filter(img.astype(np.float32),sigma=[.5/v for v in voxel_spacing])
    gate=np.percentile(img_s[tube_seg],float(E["IMAGE_GATE_PERCENTILE"]))
    gated=big_closed&~tube_seg&(img_s>=gate);candidate|=gated
    print(f"Image-gated fill candidates: {gated.sum():,}")
clab,ncand=ndi.label(candidate); cs=assign_s(candidate,tube_s_map)
for i,sl in enumerate(ndi.find_objects(clab),1):
    if sl is None: continue
    part=clab[sl]==i; vals=cs[sl][part]
    if vals.size and vals.max()-vals.min()>float(E["S_FAR_THRESHOLD"]): clab[sl][part]=0
sizes=np.bincount(clab.ravel())[1:];passes=np.r_[False,sizes>=min_size]
candidate_large=passes[clab]
if np.all(np.asarray(no_fill_radius) == 0):
    no_fill = np.zeros_like(tube_seg, dtype=bool)
else:
    no_fill = ndi.binary_dilation(
        self_contact_mask,
        structure=ellipsoid_footprint(no_fill_radius, voxel_spacing)
    ) & ~tube_seg
selected=filter_small(candidate_large&~no_fill,min_size)
print(f"Candidate components: {ncand}; selected fill voxels: {selected.sum():,}")

# Watershed assigns added fill to the neighboring cell whose membrane reaches it.
combined=tube_seg|selected
elev_sigma=1.0
img_elev=ndi.gaussian_filter(img.astype(np.float32),sigma=[elev_sigma/v for v in voxel_spacing])
denom=float(img_elev.max()-img_elev.min())
elev=np.zeros_like(img_elev) if denom==0 else (img_elev.max()-img_elev)/denom
markers=np.where(tube_seg,seg,0).astype(np.int32)
basin=watershed(elev,markers=markers,mask=combined,connectivity=1)
lut=np.full(int(seg.max())+1,np.nan,np.float32)
for cid,val in second.items():
    if cid<len(lut): lut[cid]=val
fill_basin=lut[basin]
fill_nearest=assign_s(selected,tube_s_map)
fill_s=np.where(np.isfinite(fill_basin),fill_basin,fill_nearest)
s_map=np.where(selected,fill_s,tube_s_map)
contacts=find_self_contacts(combined,s_map,s_jump_threshold)
fill_cut_direct=contacts&selected
fill_cut=fill_cut_direct.copy()
tube_cut=self_contact_mask.copy()
if trim_radius is not None:
    fp=ellipsoid_footprint(trim_radius,voxel_spacing)
    tube_cut=ndi.binary_dilation(tube_cut,structure=fp)
    fill_cut=ndi.binary_dilation(fill_cut,structure=fp)&selected
tube_trimmed=tube_seg&~tube_cut
fill_after=selected&~fill_cut
fill_final=filter_small(fill_after,min_size)
fill_labels,nfill=ndi.label(fill_final)
rejected=selected&~fill_final
print(f"Trimmed original voxels: {(tube_seg&~tube_trimmed).sum():,}; final fill: {fill_final.sum():,}")

# Fill small enclosed holes, accepting only holes whose boundary has a consistent s span.
tube_pre=tube_trimmed|fill_final
hole_fill=np.zeros_like(tube_pre);hole_labels=np.zeros(tube_pre.shape,np.int32);n_holes=n_kept=0
if fill_holes:
    maxvox=int(max_hole_um3/vox)
    try: closed=remove_small_holes(tube_pre,max_size=maxvox,connectivity=1)
    except TypeError: closed=remove_small_holes(tube_pre,area_threshold=maxvox,connectivity=1)
    candidates=closed&~tube_pre;hlab,n_holes=ndi.label(candidates);keep=np.zeros(n_holes+1,bool)
    for i,sl in enumerate(ndi.find_objects(hlab),1):
        if sl is None: continue
        pad=tuple(slice(max(a.start-1,0),min(a.stop+1,tube_pre.shape[j])) for j,a in enumerate(sl))
        touch=ndi.binary_dilation(hlab[pad]==i)&tube_pre[pad]
        vals=s_map[pad][touch]
        keep[i]=vals.size>0 and np.nanmax(vals)-np.nanmin(vals)<=s_jump_threshold
    hole_labels,n_kept=ndi.label(keep[hlab]);hole_fill=hole_labels>0
    print(f"Holes kept: {n_kept}/{n_holes}")

# Smooth added contour only.
added=fill_final|hole_fill;base_mask=tube_trimmed|added;keep=np.ones_like(base_mask)
if spur_radius is not None:
    keep &= ndi.binary_opening(base_mask,structure=ellipsoid_footprint(spur_radius,voxel_spacing))
sigma=float(E["SMOOTH_SIGMA_UM"])
if sigma>0:
    sm=ndi.gaussian_filter(base_mask.astype(np.float32),sigma=[sigma/v for v in voxel_spacing])
    keep &= sm>=.5
fill_final=filter_small(fill_final&keep,min_size);hole_fill &= keep
tube_pre=tube_trimmed|fill_final
fill_labels,nfill=ndi.label(fill_final);hole_labels,n_kept=ndi.label(hole_fill)

# Assemble and save.
assembled=tube_pre|hole_fill
lab,ncomp=ndi.label(assembled);sz=np.bincount(lab.ravel());sz[0]=0
tube_final=lab==sz.argmax()
out=Path(E["OUT_DIR"]);out.mkdir(parents=True,exist_ok=True)
meta=dict(imagej=True,resolution=(1/voxel_spacing[2],1/voxel_spacing[1]),
          metadata={"spacing":voxel_spacing[0],"unit":"um","axes":"ZYX"})
tifffile.imwrite(out/(base+"_tube_final.tif"),tube_final.astype(np.uint8),**meta)
print(f"Saved outputs to {out}")
print(f"Tube voxels: {tube_final.sum():,}; connected components: {ncomp}; fill: {fill_final.sum():,}; holes: {hole_fill.sum():,}")
PYEOF
echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISH ${TASK_LABEL}"
