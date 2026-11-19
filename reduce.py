## Memory-efficient VATT data reduction pipeline (reduce.py)
##
## Steps:
##   1. Build master biases (per night) and master flats (per filter set)
##   2. De-bias and flat-field every science image
##   3. Build a single reference master fringe frame from the i+Clear filter
##      set (i+Clear and Clear+i are pooled)
##   4. Fit a fringe scale factor to, and subtract the reference master fringe
##      from, every image with VR, i, I, or z in its filter set
##   5. Write the reduced images
##
## Usage:
##   python3 reduce.py path_to_day_subdirs [-t | --time] [-m | --memory] [--no_defringing]
##
## Arguments and flags:
##   path_to_day_subdirs   Directory containing one subdirectory per night, each
##                         named YYYYMMDD. Any other subdirectory (including
##                         'reduced') is ignored. All output goes to
##                         path_to_day_subdirs/reduced/.
##   -t, --time            Print the total processing time at the end (time spent
##                         waiting at the flat-combination prompt is excluded).
##   -m, --memory          Print the peak memory usage at the end.
##   --no_defringing       Skip master fringe construction and all fringe fitting
##                         and subtraction. Biases and flats are still applied.
##                         Any leftover reduced/temp and reduced/master_fringes
##                         directories from a previous run are deleted.
##
## Prompts:
##   If a filter set has flats from more than one night, the script lists every
##   combination of nights and asks which one to use for that filter set's
##   master flat. This is the only step that waits for input.
##
## Outputs (all under path_to_day_subdirs/reduced/):
##   <night>/master_bias.fits               master bias for each night
##   master_flats/<upper>_<lower>_master_flat.fits
##   master_fringes/master_fringe_IClear_normalized.fits
##   <night>/red_<original name>.fits       reduced science images
##   fringe_scale_log.csv                   fitted fringe scale for every image that
##                                          was defringed (or failed to be); only
##                                          written when a master fringe was built
##
## Filter handling:
##   Filter names are case-sensitive ('i' and 'I' are different filters), with
##   one exception: "clear" is recognized regardless of case. x+Clear and
##   Clear+x are treated as exactly the same filter set, so their flats and
##   science frames are pooled together. Combinations without clear keep their
##   wheel positions as given.
##
## Reference master fringe:
##   Built from all i+Clear science frames (lowercase i) with exptime greater
##   than FRINGE_MIN_EXPTIME (currently 31 s; frames at or below this are
##   excluded). Each frame is debiased and flat-fielded, converted to counts/sec,
##   median-subtracted, and divided by its sigma-clipped whole-image standard
##   deviation before combining, since fringe amplitude varies night-to-night and
##   within a night. If there are at least MGM_MIN_TOTAL qualifying frames
##   (currently 35), the frames are split into N_GROUPS interleaved groups
##   (currently 5), each group is combined with a strip-wise sigma-clipped mean,
##   and the final master is the pixel-wise median of the group means. With fewer
##   frames, a single strip-wise sigma-clipped mean is used instead. The master
##   is unitless (normalized), so no exposure time applies to it.
##
## Fringe fitting and subtraction (every image with VR, i, I, or z in its set,
## regardless of exposure time):
##   N_LINES vertical and N_LINES horizontal lines (currently 10 each) are
##   sampled across the image. The science image is median-subtracted and pixels
##   above the clipping sigma are masked. The sigma depends on the filter set
##   (FILTER_SIGMA_SCHEDULES): VR tries 1.5 and then 2.0 if that fails; every
##   other filter set uses 3.0. Lines with more than MAX_MASKED_FRAC (25%) of
##   their pixels masked are replaced by lines placed between two of the
##   remaining good lines. Along each line, a LINE_SMOOTH_WIDTH-pixel (5)
##   mask-aware running mean is taken of both the science image and the master
##   fringe, and a least-squares scale is fit per line. Lines with non-positive
##   scales are dropped, then the half of the remaining lines with the largest
##   mean per-pixel deviation are dropped, and the median of the remaining
##   scales is the fringe scale. The scaled master fringe is subtracted from the
##   image. If no valid scale can be found, the image is written without
##   defringing and a warning is printed.

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
import sys
import shutil
import resource
import csv
from pathlib import Path
from datetime import datetime
import re
import time
from itertools import combinations
from collections import defaultdict

start_time = time.time()
elapsed_before_input = 0

# Filters that need fringe correction (either wheel, case-sensitive)
FRINGE_FILTERS = {'VR', 'i', 'I', 'z'}
FRINGE_MIN_EXPTIME = 31   # master fringe uses only frames with exptime > this many seconds (31 s itself is excluded)

# Canonical label used for the "clear" position in a filter key
CLEAR_LABEL = 'Clear'

# Filter set used to build the one reference master fringe frame
REFERENCE_FILTER_KEY = ('i', CLEAR_LABEL)


def parse_filter(filter_str):
    upper = re.search(r'upper:\s*(\S+)', filter_str).group(1)
    lower = re.search(r'lower:\s*(\S+)', filter_str).group(1)
    return upper, lower


def is_clear(name):
    """'clear' is matched regardless of case; every other filter name is case-sensitive."""
    return name.lower() == 'clear'


def canonical_filter_key(upper, lower):
    """
    Return an order-independent filter key for clear combinations, so that
    x+Clear and Clear+x are the same key: (x, 'Clear'). Combinations without
    clear keep their wheel positions as given.
    """
    if is_clear(upper) and is_clear(lower):
        return (CLEAR_LABEL, CLEAR_LABEL)
    if is_clear(upper):
        return (lower, CLEAR_LABEL)
    if is_clear(lower):
        return (upper, CLEAR_LABEL)
    return (upper, lower)


def needs_fringe(upper, lower):
    """Return True if either filter in the set is one of FRINGE_FILTERS (VR/i/I/z)."""
    return upper in FRINGE_FILTERS or lower in FRINGE_FILTERS


def read_and_prep_image(im_file):
    """Read FITS file, extract chips, remove overscan. Returns (amp1, amp2, header)."""
    with fits.open(im_file) as hdu_list:
        header = hdu_list[0].header
        amp1 = np.flipud(hdu_list[1].data)   # bottom chip, flip vertically
        amp2 = hdu_list[2].data               # top chip, no transform needed

        # Remove overscan from each chip individually
        amp1 = amp1[:, :-24]
        amp2 = amp2[:, :-24]

    return amp1, amp2, header


def stitch_for_output(amp1, amp2):
    """Stitch chips for writing: amp2 on top, amp1 on bottom, then flipud."""
    return np.flipud(np.concatenate((amp2, amp1), axis=0))


def normalize_for_fringe(image_data, exptime):
    """
    Normalize a stitched, bias/flat-corrected image for master fringe
    construction: convert to counts/sec, subtract the median, then divide
    by the sigma-clipped whole-image standard deviation. This puts every
    input frame on a common, unitless scale regardless of exposure time,
    since fringe amplitude varies night-to-night and within a night in
    ways a simple exptime scaling can't capture.
    Used only for master fringe construction — not applied to final output.
    """
    cps = image_data / exptime
    centered = cps - np.median(cps)
    clipped = sigma_clip(centered, sigma=3.0)
    std = np.ma.std(clipped)
    return centered / std


def debias_and_flatten(im_file, master_bias, master_flat):
    """
    Load a raw science image, apply bias and flat corrections, return
    the stitched float array and header. Returns (stitched, header).
    """
    amp1, amp2, header = read_and_prep_image(im_file)

    # Bias subtract
    amp1 = amp1.astype(float) - master_bias[0]
    amp2 = amp2.astype(float) - master_bias[1]

    # Flat field
    mf_amp1, mf_amp2 = master_flat
    amp1 /= mf_amp1
    amp2 /= mf_amp2

    return stitch_for_output(amp1, amp2), header


print()

time_flag        = '-t' in sys.argv or '--time'   in sys.argv
memory_flag      = '-m' in sys.argv or '--memory' in sys.argv
no_defringe_flag = '--no_defringing' in sys.argv
args = [a for a in sys.argv[1:] if a not in
        ('-t', '--time', '-m', '--memory', '--no_defringing')]

im_dir = Path(args[0])


def is_date_dirname(name, fmt="%Y%m%d"):
    """Return True if `name` parses as a date in the given format (default YYYYMMDD)."""
    try:
        datetime.strptime(name, fmt)
        return True
    except ValueError:
        return False


# Only descend into top-level subdirectories of im_dir that are named as
# dates (YYYYMMDD). This naturally skips 'reduced', any renamed variant of
# it (e.g. 'reduced_with_standard_defringing'), and any other non-night
# directory, regardless of what it's called.
date_dirs = []
ignored_dirs = []
for entry in sorted(im_dir.iterdir()):
    if entry.is_dir():
        if is_date_dirname(entry.name):
            date_dirs.append(entry)
        else:
            ignored_dirs.append(entry)

for d in ignored_dirs:
    print(f"Ignoring directory (name is not a recognized YYYYMMDD date): {d}")

im_files = []
for d in date_dirs:
    im_files.extend(d.rglob("*.fits"))

# Filter out master and test images within the night directories
temp = []
for file in im_files:
    name = file.name
    if (name[0] == 'm') or (name[0:4] == 'test'):
        print(f"Throwing out {file}")
    else:
        temp.append(file)

Path(f"{im_dir}/reduced").mkdir(exist_ok=True)
im_files = temp

# Categorize files by type without loading them yet
bias_files = []
flat_files = []
science_files = []

print(f"\nScanning {len(im_files)} images for type...")

for im_file in im_files:
    with fits.open(im_file, memmap=True) as hdu_list:
        header = hdu_list[0].header
        imagetyp = header.get('IMAGETYP', 'unknown')

        if imagetyp == 'zero':
            bias_files.append(im_file)
        elif imagetyp == 'flat':
            flat_files.append(im_file)
        elif imagetyp == 'object':
            science_files.append(im_file)

print(f"Found {len(bias_files)} biases, {len(flat_files)} flats, {len(science_files)} science images\n")

# ============================================================================
# PASS 1: Build master biases
# ============================================================================
print("=" * 60)
print("PASS 1: BUILDING MASTER BIASES")
print("=" * 60)

# Organize bias files by date directory
biases_by_dir = defaultdict(list)
for bias_file in bias_files:
    date_dir = bias_file.parts[-2]
    biases_by_dir[date_dir].append(bias_file)

unique_dirs = sorted(biases_by_dir.keys())
master_biases = {}  # {date_dir: (master_amp1, master_amp2)} or date_dir: None

for date_dir in unique_dirs:
    bias_files_for_date = biases_by_dir[date_dir]

    if len(bias_files_for_date) == 0:
        master_biases[date_dir] = None
        continue

    means = []
    stds = []
    bias_chips_temp = []

    for bias_file in bias_files_for_date:
        amp1, amp2, _ = read_and_prep_image(bias_file)
        bias_chips_temp.append((amp1, amp2))
        means.append(np.mean((amp1 + amp2) / 2))
        stds.append(np.std(np.concatenate([amp1.ravel(), amp2.ravel()])))

    med_mean = np.median(means)
    med_std = np.median(stds)

    keep_list = []
    for j, (amp1, amp2) in enumerate(bias_chips_temp):
        keep = True
        if (means[j] > med_mean * 2) or (means[j] < med_mean / 2):
            keep = False
        if (stds[j] > med_std * 2) or (stds[j] < med_std / 2):
            keep = False
        keep_list.append(keep)

    keepers = [(amp1, amp2) for (amp1, amp2), keep in zip(bias_chips_temp, keep_list) if keep]

    if len(keepers) < 9:
        master_biases[date_dir] = None
        del bias_chips_temp
        continue

    if len(keepers) % 2 == 0:
        keepers = keepers[1:]

    master_amp1 = np.median(np.stack([c[0] for c in keepers], axis=0), axis=0)
    master_amp2 = np.median(np.stack([c[1] for c in keepers], axis=0), axis=0)
    master_biases[date_dir] = (master_amp1, master_amp2)

    out_path = im_dir / "reduced" / date_dir
    out_path.mkdir(parents=True, exist_ok=True)
    stitched = stitch_for_output(master_amp1, master_amp2)
    fits.writeto(out_path / "master_bias.fits", stitched, overwrite=True)

    del bias_chips_temp, keepers

print(f"\nWrote master biases to {im_dir}/reduced")


def find_nearest_bias(target_dir, available_bias_dirs, master_biases_dict):
    """Find nearest bias by date, load and return it."""
    if target_dir in master_biases_dict and master_biases_dict[target_dir] is not None:
        return master_biases_dict[target_dir]

    target_date = datetime.strptime(target_dir, "%Y%m%d")
    available_dates = [(datetime.strptime(d, "%Y%m%d"), d) for d in master_biases_dict.keys()
                       if master_biases_dict[d] is not None]

    if not available_dates:
        raise ValueError(f"No master biases available for {target_dir}")

    deltas = [(abs((d - target_date).days), d, dir_name) for d, dir_name in available_dates]
    deltas.sort(key=lambda x: (x[0], -x[1].timestamp()))
    nearest_date, nearest_dir = deltas[0][1], deltas[0][2]
    print(f"No master bias for {target_dir}, using {nearest_dir}")

    bias_path = im_dir / "reduced" / nearest_dir / "master_bias.fits"
    with fits.open(bias_path) as hdul:
        stitched = hdul[0].data
        unstitched = np.flipud(stitched)
        amp2, amp1 = np.array_split(unstitched, 2, axis=0)
        return (amp1, amp2)


# ============================================================================
# PASS 2: Build master flats
# ============================================================================
print("\n" + "=" * 60)
print("PASS 2: BUILDING MASTER FLATS")
print("=" * 60)

flats_by_dir = defaultdict(list)
for flat_file in flat_files:
    date_dir = flat_file.parts[-2]
    flats_by_dir[date_dir].append(flat_file)


def make_master_chip(amp1_frames, amp2_frames):
    """Normalize each frame by the median of both chips combined, then sigma-clip and average."""
    stack_amp1 = np.stack([a1 / np.median(np.concatenate([a1.ravel(), a2.ravel()]))
                           for a1, a2 in zip(amp1_frames, amp2_frames)], axis=0)
    stack_amp2 = np.stack([a2 / np.median(np.concatenate([a1.ravel(), a2.ravel()]))
                           for a1, a2 in zip(amp1_frames, amp2_frames)], axis=0)
    clipped_amp1 = sigma_clip(stack_amp1, sigma=3, axis=0)
    clipped_amp2 = sigma_clip(stack_amp2, sigma=3, axis=0)
    return np.ma.mean(clipped_amp1, axis=0).data, np.ma.mean(clipped_amp2, axis=0).data


def make_master_flat_from_raw_flats(chip_pairs):
    """Create master flat from raw flats with outlier rejection."""
    good_pairs = []
    for amp1, amp2 in chip_pairs:
        med = (np.median(amp1) + np.median(amp2)) / 2
        if med < 20000 or med > 50000:
            continue
        good_pairs.append((amp1, amp2))

    if len(good_pairs) == 0:
        return None, 0

    master_amp1, master_amp2 = make_master_chip([p[0] for p in good_pairs],
                                                [p[1] for p in good_pairs])
    return (master_amp1, master_amp2), len(good_pairs)


def generate_date_combinations(dates):
    """Generate all combinations of dates from 1 date up to all dates."""
    all_combos = []
    for r in range(1, len(dates) + 1):
        for combo in combinations(dates, r):
            all_combos.append(list(combo))
    return all_combos


flats_by_filter = {}
for date_dir, flats_for_date in flats_by_dir.items():
    for flat_file in flats_for_date:
        amp1, amp2, header = read_and_prep_image(flat_file)

        master_bias = find_nearest_bias(date_dir, list(master_biases.keys()), master_biases)
        if master_bias:
            amp1 = amp1 - master_bias[0]
            amp2 = amp2 - master_bias[1]

        upper, lower = parse_filter(header['FILTER'])
        filter_key = canonical_filter_key(upper, lower)   # x+Clear and Clear+x are pooled

        if filter_key not in flats_by_filter:
            flats_by_filter[filter_key] = {}
        if date_dir not in flats_by_filter[filter_key]:
            flats_by_filter[filter_key][date_dir] = []

        flats_by_filter[filter_key][date_dir].append((amp1, amp2))

Path(f"{im_dir}/reduced/master_flats").mkdir(parents=True, exist_ok=True)

master_flats = {}  # {canonical filter key: (master_amp1, master_amp2)}

for filter_key, nights_dict in flats_by_filter.items():
    upper, lower = filter_key
    nights = sorted(nights_dict.keys())

    if len(nights) == 1:
        night = nights[0]
        chip_pairs = nights_dict[night]
        master_flat, num_frames = make_master_flat_from_raw_flats(chip_pairs)
        if master_flat:
            master_flats[filter_key] = master_flat
            print(f"Master flat for {upper}+{lower} from {num_frames} frames on {night}")
        else:
            print(f"No good flats for {upper}+{lower}")
    else:
        combinations_list = generate_date_combinations(nights)

        print(f"\nMultiple nights with flats for {upper}+{lower}:")
        for j, combo in enumerate(combinations_list):
            combo_str = " + ".join(combo)
            print(f"  {j}: {combo_str}")

        elapsed_before_input += time.time() - start_time
        while True:
            choice = input(f"Which combination to use? Enter number 0-{len(combinations_list)-1}: ")
            if choice.isdigit() and 0 <= int(choice) < len(combinations_list):
                selected_combo = combinations_list[int(choice)]

                all_chip_pairs = []
                for night in selected_combo:
                    all_chip_pairs.extend(nights_dict[night])

                master_flat, num_frames = make_master_flat_from_raw_flats(all_chip_pairs)
                if master_flat:
                    master_flats[filter_key] = master_flat
                    combo_str = " + ".join(selected_combo)
                    print(f"Combined {num_frames} frames from {len(selected_combo)} night(s): {combo_str}")
                    break
                else:
                    print("No good frames in that combination, try another")
            else:
                print("Invalid choice, try again")
        start_time = time.time()

    if filter_key in master_flats:
        mf_amp1, mf_amp2 = master_flats[filter_key]
        stitched = stitch_for_output(mf_amp1, mf_amp2)
        out_name = f"{upper}_{lower}_master_flat.fits"
        fits.writeto(im_dir / "reduced" / "master_flats" / out_name, stitched, overwrite=True)

print(f"\nWrote master flats to {im_dir}/reduced/master_flats")

# ============================================================================
# FRINGE HELPERS (master construction + line-profile scale fitting)
# ============================================================================
# --- BEGIN FRINGE HELPERS ---
np.random.seed(1234)

# Master fringe construction
STRIP_HEIGHT  = 64
SIGMA_THRESH  = 3.0
MGM_MIN_TOTAL = 35    # minimum number of qualifying frames to use grouped-means -> median
N_GROUPS      = 5     # number of interleaved groups for grouped-means -> median

# Line-profile scale fitting
N_LINES            = 10     # number of vertical AND number of horizontal lines
LINE_SMOOTH_WIDTH  = 5      # running-mean width along each line (pixels)
MIN_VALID_IN_WINDOW = 3     # min unmasked pixels in a smoothing window
MAX_MASKED_FRAC    = 0.25   # lines with more than this fraction masked get replaced

# Clipping sigma schedules by filter. Each list is tried in order; the first
# sigma that yields a valid scale is used. Keys are matched against either
# filter of the set (case-sensitive). Filters not listed use DEFAULT_SIGMA_SCHEDULE.
FILTER_SIGMA_SCHEDULES = {
    'VR': [1.5, 2.0],
}
DEFAULT_SIGMA_SCHEDULE = [3.0]


def image_number(rec):
    """Extract the trailing 4-digit image number from a filename."""
    m = re.search(r'(\d{4})\.fits$', rec['path'].name)
    return int(m.group(1)) if m else 0


def interleave_by_night(records):
    """
    Sort records so that images from different nights are interleaved
    as evenly as possible and adjacent image numbers from the same
    night are never consecutive in the output list.

    Algorithm:
      1. Group records by date, sorting each night's images by image number.
      2. Draw round-robin across nights in date order until exhausted.
    """
    by_night = defaultdict(list)
    for r in records:
        by_night[r['date']].append(r)
    for night in by_night:
        by_night[night].sort(key=image_number)

    ordered = []
    nights = sorted(by_night.keys())
    queues = [by_night[n] for n in nights]
    idx = [0] * len(queues)
    while True:
        added = False
        for q_i, queue in enumerate(queues):
            if idx[q_i] < len(queue):
                ordered.append(queue[idx[q_i]])
                idx[q_i] += 1
                added = True
        if not added:
            break
    return ordered


def strip_clipped_mean(temp_paths, image_shape):
    """Compute a strip-wise sigma-clipped mean over a list of temp FITS paths."""
    n_rows, n_cols = image_shape
    result = np.zeros(image_shape, dtype=np.float64)
    for row_start in range(0, n_rows, STRIP_HEIGHT):
        row_end = min(row_start + STRIP_HEIGHT, n_rows)
        strips = np.stack(
            [fits.getdata(p, memmap=False)[row_start:row_end].astype(np.float32)
             for p in temp_paths],
            axis=0
        )
        clipped = sigma_clip(strips, sigma=SIGMA_THRESH, axis=0)
        result[row_start:row_end] = np.ma.mean(clipped, axis=0).data
        del strips, clipped
    return result


def get_sigma_schedule(upper, lower):
    """Return the list of clipping sigmas to try for this filter set."""
    for filt, schedule in FILTER_SIGMA_SCHEDULES.items():
        if filt in (upper, lower):
            return schedule
    return DEFAULT_SIGMA_SCHEDULE


def sky_subtract_and_mask(image, sigma):
    """
    Median-subtract the image, then mask pixels more than `sigma` above the
    sigma-clipped scatter (stars, cosmic rays, etc.). Returns a masked array
    in the original (raw count) units.
    """
    centered = image - np.median(image)
    clipped = sigma_clip(centered, sigma_upper=sigma, sigma_lower=np.inf,
                         maxiters=5, masked=True)
    return np.ma.masked_array(centered, mask=np.ma.getmaskarray(clipped))


def get_line_positions(shape, n_lines=N_LINES):
    """Evenly spaced row/column indices, kept away from the very edges."""
    n_rows, n_cols = shape
    cols = np.linspace(0.05 * n_cols, 0.95 * n_cols, n_lines).astype(int)
    rows = np.linspace(0.05 * n_rows, 0.95 * n_rows, n_lines).astype(int)
    return rows, cols


def line_masked_fraction(masked, kind, idx):
    """Fraction of pixels on a full row/column removed by the sigma mask."""
    line = masked[:, idx] if kind == 'col' else masked[idx, :]
    return float(np.mean(np.ma.getmaskarray(line)))


def select_lines(masked, kind, init_positions, max_frac=MAX_MASKED_FRAC, max_levels=4):
    """
    Keep the initial lines whose masked fraction is <= max_frac. Each line
    over the limit is replaced by a new line of the same orientation at the
    midpoint between two adjacent good lines (nearest-first to the line being
    replaced). If all midpoints at one level fail, the search subdivides
    further, up to max_levels. If not enough good lines can be found, fewer
    than the initial number are returned.
    """
    init_positions = [int(p) for p in init_positions]
    good = sorted(p for p in init_positions
                  if line_masked_fraction(masked, kind, p) <= max_frac)
    bad = [p for p in init_positions if p not in good]
    n_missing = len(bad)
    if n_missing == 0 or len(good) < 2:
        return good

    tried = set(init_positions)
    for _ in range(max_levels):
        mids = []
        for a, b in zip(good[:-1], good[1:]):
            m = (a + b) // 2
            if b - a > 1 and m not in tried:
                mids.append(m)
        mids.sort(key=lambda m: min(abs(m - p) for p in bad))

        added = []
        for m in mids:
            if n_missing == 0:
                break
            tried.add(m)
            if line_masked_fraction(masked, kind, m) <= max_frac:
                added.append(m)
                n_missing -= 1
        good = sorted(set(good) | set(added))
        if n_missing == 0:
            break
    return good


def smooth_line(line, width=LINE_SMOOTH_WIDTH):
    """
    Mask-aware 1-D running mean along a line. Each output pixel is the mean
    of the unmasked pixels in a `width`-pixel window centered on it. Pixels
    with fewer than MIN_VALID_IN_WINDOW valid neighbors come out masked.
    Works on masked arrays (science) and plain arrays (master fringe).
    """
    mask = np.ma.getmaskarray(line)
    vals = np.where(mask, 0.0, np.ma.getdata(line)).astype(float)
    valid = (~mask).astype(float)
    kernel = np.ones(width)
    sums = np.convolve(vals, kernel, mode='same')
    counts = np.convolve(valid, kernel, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        means = sums / counts
    out_mask = counts < MIN_VALID_IN_WINDOW
    return np.ma.masked_array(np.where(out_mask, 0.0, means), mask=out_mask)


def fit_line_scale(sci_line, master_line):
    """
    Closed-form least-squares scale s minimizing sum((sci - s*master)^2)
    over unmasked pixels, plus the mean absolute per-pixel deviation between
    the science line and the scaled master line at that scale.
    Returns (scale, mean_abs_dev); both are np.nan if too few good pixels.
    """
    good = ~np.ma.getmaskarray(sci_line)
    if good.sum() < 50:
        return np.nan, np.nan
    s = np.asarray(sci_line.data[good], dtype=float)
    m = np.asarray(np.ma.getdata(master_line)[good], dtype=float)
    denom = np.sum(m * m)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan, np.nan
    scale = np.sum(s * m) / denom
    dev = np.mean(np.abs(s - scale * m))
    return scale, dev


def fit_fringe_scale_lines(image, master_fringe, sigma, n_lines=N_LINES):
    """
    Fit one scale per line (vertical + horizontal) using running means along
    each line, then:
      1. drop lines with a NaN or non-positive scale,
      2. of the rest, keep the half (rounded up) with the smallest mean
         absolute per-pixel deviation,
      3. return the median scale of the kept lines.
    Returns (median_scale, n_lines_used); (nan, 0) if no line has a valid scale.
    """
    masked = sky_subtract_and_mask(image, sigma)
    rows0, cols0 = get_line_positions(image.shape, n_lines)
    cols = select_lines(masked, 'col', cols0)
    rows = select_lines(masked, 'row', rows0)

    results = []
    for c in cols:
        results.append(fit_line_scale(smooth_line(masked[:, c]),
                                      smooth_line(master_fringe[:, c])))
    for r in rows:
        results.append(fit_line_scale(smooth_line(masked[r, :]),
                                      smooth_line(master_fringe[r, :])))

    if not results:
        return np.nan, 0
    scales = np.array([res[0] for res in results], dtype=float)
    devs = np.array([res[1] for res in results], dtype=float)

    valid_idx = np.where(np.isfinite(scales) & (scales > 0))[0]
    if valid_idx.size == 0:
        return np.nan, 0

    n_keep = int(np.ceil(valid_idx.size / 2))
    keep_idx = valid_idx[np.argsort(devs[valid_idx])[:n_keep]]
    return float(np.median(scales[keep_idx])), int(keep_idx.size)


def fit_fringe_scale(image, master_fringe, sigma_schedule):
    """
    Try each clipping sigma in order and return the first valid result.
    Returns (scale, sigma_used, n_lines_used); scale is nan if every sigma fails.
    """
    for sig in sigma_schedule:
        scale, n_used = fit_fringe_scale_lines(image, master_fringe, sig)
        if np.isfinite(scale):
            return scale, sig, n_used
    return np.nan, sigma_schedule[-1], 0
# --- END FRINGE HELPERS ---

# ============================================================================
# PASS 3: Build a single reference master fringe frame (i+Clear only),
# memory-efficient disk-based strip-wise approach
# ============================================================================
print("\n" + "=" * 60)
print("PASS 3: BUILDING REFERENCE MASTER FRINGE FRAME (i+CLEAR)")
print("=" * 60)

# science_files_by_dir is needed below regardless of fringing (also used in PASS 4)
science_files_by_dir = defaultdict(list)
for sci_file in science_files:
    date_dir = sci_file.parts[-2]
    science_files_by_dir[date_dir].append(sci_file)

# -----------------------------------------------------------------
# Only i+Clear frames (either wheel order) go into the reference master.
# This single master is later scaled and subtracted from every science frame
# whose filter set contains VR, i, I, or z.
#
#   Step 1 — Grouping and ordering.
#       Frames are sorted to interleave nights as evenly as possible, so that
#       consecutive image numbers from the same night are never placed in the
#       same group: within each night, sort by trailing 4-digit image number,
#       then draw round-robin across nights in date order.
#
#   Step 2 — Normalization and temp file writing.
#       Each ordered image is debias-corrected, flat-fielded (with the i+Clear
#       master flat), converted to counts/sec, median-subtracted, and divided
#       by its sigma-clipped whole-image standard deviation, then written to
#       reduced/temp/ as float32. Never more than one image in memory at once.
#
# Grouped-means mode (default, >= MGM_MIN_TOTAL frames):
#   Step 3 — Frames are split into N_GROUPS groups by round-robin across the
#       interleaved order. For each group a strip-wise sigma-clipped mean is
#       computed.
#   Step 4 — The group means are combined with a pixel-wise median to give the
#       final reference master fringe.
#
# Fallback (fewer than MGM_MIN_TOTAL frames):
#   Step 3 — Single strip-wise sigma-clipped mean over all temp files.
#
#   Step 5 — All temp files are deleted; the temp directory is removed if empty.
# -----------------------------------------------------------------

temp_dir = Path(f"{im_dir}/reduced/temp")
master_fringes_dir = Path(f"{im_dir}/reduced/master_fringes")
iclear_master_fringe = None   # single reference master fringe (i+Clear)

if no_defringe_flag:
    print("--no_defringing set: skipping master fringe construction")
    # Remove any leftover fringe artifacts from a previous run (e.g. one
    # done without --no_defringing), since none will be produced now.
    for stale_dir in (temp_dir, master_fringes_dir):
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
            print(f"Removed leftover directory: {stale_dir}")
else:
    iclear_records = []

    if REFERENCE_FILTER_KEY not in master_flats:
        print("\n  No master flat for i+Clear; cannot build a reference master fringe.")
    else:
        # Header-only scan for i+Clear science frames with long enough exposures
        print("Scanning science headers for i+Clear frames...")
        for date_dir in sorted(science_files_by_dir.keys()):
            for sci_file in science_files_by_dir[date_dir]:
                with fits.open(sci_file, memmap=True) as hdul:
                    header = hdul[0].header
                    try:
                        upper, lower = parse_filter(header['FILTER'])
                    except (KeyError, AttributeError):
                        continue
                    if canonical_filter_key(upper, lower) != REFERENCE_FILTER_KEY:
                        continue
                    exptime = header.get('EXPTIME')
                    if exptime is None:
                        continue
                    exptime = float(exptime)
                    if exptime <= FRINGE_MIN_EXPTIME:
                        continue
                    iclear_records.append({
                        'path':    sci_file,
                        'date':    date_dir,
                        'exptime': exptime,
                    })

        print(f"  i+Clear: {len(iclear_records)} frames with exptime > {FRINGE_MIN_EXPTIME}s")

    if not iclear_records:
        if REFERENCE_FILTER_KEY in master_flats:
            print("\n  No usable i+Clear science frames found; cannot build a "
                  "reference master fringe.")
        print("  Fringe correction will be skipped for all filters.")
    else:
        master_fringes_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        n_frames = len(iclear_records)
        print(f"\n  Building reference i+Clear master fringe from {n_frames} frames...")

        # ---- Step 1: interleave images across nights ----
        ordered = interleave_by_night(iclear_records)

        use_mgm = n_frames >= MGM_MIN_TOTAL
        if not use_mgm:
            print(f"  Only {n_frames} frames available "
                  f"(need >= {MGM_MIN_TOTAL} for grouped means). "
                  f"Falling back to single-pass sigma-clipped mean.")

        # ---- Step 2: normalize and write temp files ----
        temp_paths = []
        image_shape = None
        print(f"  Writing {n_frames} normalized temp files...")

        for i, rec in enumerate(ordered):
            stitched, _ = debias_and_flatten(
                rec['path'],
                find_nearest_bias(rec['date'], list(master_biases.keys()), master_biases),
                master_flats[REFERENCE_FILTER_KEY],
            )
            normalized = normalize_for_fringe(stitched, rec['exptime'])
            del stitched

            if image_shape is None:
                image_shape = normalized.shape

            temp_path = temp_dir / f"tmp_fringe_IClear_{i:04d}.fits"
            fits.writeto(temp_path, normalized.astype(np.float32), overwrite=True)
            temp_paths.append(temp_path)
            del normalized

        # ---- Step 3+4: build master fringe ----
        if use_mgm:
            print(f"  Splitting {n_frames} frames into {N_GROUPS} groups "
                  f"(~{n_frames // N_GROUPS} frames each) via round-robin assignment...")

            # Round-robin so each group is itself a well-mixed, interleaved subset
            group_paths = [[] for _ in range(N_GROUPS)]
            for i, p in enumerate(temp_paths):
                group_paths[i % N_GROUPS].append(p)

            group_means = []
            for g_i, g_paths in enumerate(group_paths):
                print(f"    Group {g_i + 1}/{N_GROUPS}: "
                      f"sigma-clipped mean of {len(g_paths)} frames...")
                group_means.append(strip_clipped_mean(g_paths, image_shape))

            print(f"  Taking pixel-wise median across {N_GROUPS} group means...")
            iclear_master_fringe = np.median(np.stack(group_means, axis=0), axis=0)
            del group_means
        else:
            print(f"  Computing strip-wise sigma-clipped mean "
                  f"({image_shape[0] // STRIP_HEIGHT + 1} strips)...")
            iclear_master_fringe = strip_clipped_mean(temp_paths, image_shape)

        # ---- Step 5: clean up temp files ----
        for p in temp_paths:
            p.unlink()

        # Write reference master fringe (normalized, unitless: counts/sec,
        # median-subtracted, divided by sigma-clipped whole-image std)
        out_name = "master_fringe_iClear_normalized.fits"
        fits.writeto(master_fringes_dir / out_name, iclear_master_fringe, overwrite=True)
        print(f"  Saved: {master_fringes_dir}/{out_name}")

# Remove temp directory if empty
try:
    temp_dir.rmdir()
except OSError:
    pass

if no_defringe_flag:
    print("\nFringe correction skipped (--no_defringing)")
elif iclear_master_fringe is not None:
    print(f"\nWrote reference master fringe frame to {master_fringes_dir}")
else:
    print("\nNo reference master fringe frame was built")

# ============================================================================
# PASS 4: Reduce science images; fit and subtract fringes where needed
# ============================================================================
print("\n" + "=" * 60)
print("PASS 4: REDUCING SCIENCE IMAGES")
print("=" * 60)
print(f"Processing {len(science_files)} science images...\n")

processed_count = 0
n_defringed = 0
n_fringe_failed = 0
warned_no_master = False
fringe_log_rows = []   # (image_path, upper, lower, exptime, fringe_scale, fringe_scale_per_sec, sigma_used, n_lines_used)

for date_dir in sorted(science_files_by_dir.keys()):
    sci_files_for_date = science_files_by_dir[date_dir]

    master_bias = find_nearest_bias(date_dir, list(master_biases.keys()), master_biases)

    for sci_file in sci_files_for_date:
        amp1, amp2, header = read_and_prep_image(sci_file)

        # Bias subtract
        amp1 = amp1.astype(float) - master_bias[0]
        amp2 = amp2.astype(float) - master_bias[1]

        # Flat field
        upper, lower = parse_filter(header['FILTER'])
        filter_key = canonical_filter_key(upper, lower)   # x+Clear and Clear+x are pooled

        if filter_key not in master_flats:
            print(f"No master flat for {filter_key}, skipping {sci_file.name}")
            continue

        mf_amp1, mf_amp2 = master_flats[filter_key]
        amp1 /= mf_amp1
        amp2 /= mf_amp2

        # Stitch
        reduced = stitch_for_output(amp1, amp2)

        # Fringe correction: every image with VR, i, I, or z in its filter set
        if (not no_defringe_flag) and needs_fringe(*filter_key):
            if iclear_master_fringe is None:
                if not warned_no_master:
                    print("WARNING: no reference master fringe available; "
                          "fringe-affected images will be written without defringing")
                    warned_no_master = True
            else:
                sigma_schedule = get_sigma_schedule(*filter_key)
                scale, sigma_used, n_lines_used = fit_fringe_scale(
                    reduced, iclear_master_fringe, sigma_schedule)

                exptime = header.get('EXPTIME')
                exptime = float(exptime) if exptime is not None else np.nan
                scale_per_sec = scale / exptime if np.isfinite(scale) else np.nan

                if np.isfinite(scale):
                    reduced -= scale * iclear_master_fringe
                    n_defringed += 1
                else:
                    n_fringe_failed += 1
                    print(f"WARNING: could not fit a fringe scale for {sci_file.name} "
                          f"(sigma schedule {sigma_schedule}); writing without defringing")

                fringe_log_rows.append((str(sci_file), upper, lower, exptime,
                                        scale, scale_per_sec, sigma_used, n_lines_used))

        # Write
        out_dir = im_dir / "reduced" / date_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = "red_" + sci_file.name
        fits.writeto(out_dir / out_name, reduced, header, overwrite=True)

        processed_count += 1
        if processed_count % 10 == 0:
            print(f"  Processed {processed_count} science images...")

print(f"\nWrote {processed_count} reduced science images")

if not no_defringe_flag and iclear_master_fringe is not None:
    print(f"Defringed {n_defringed} images; fringe fit failed for {n_fringe_failed}")

    # Log of fitted fringe scales (raw and exptime-normalized), for tracking
    # fringe amplitude over the nights
    log_path = im_dir / "reduced" / "fringe_scale_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "filter_upper", "filter_lower", "exptime",
                         "fringe_scale", "fringe_scale_per_sec",
                         "clip_sigma_used", "n_lines_used"])
        writer.writerows(fringe_log_rows)
    print(f"Wrote fringe scale log to {log_path}")

if time_flag:
    total_time = elapsed_before_input + (time.time() - start_time)
    print(f"\nProcessing time: {total_time:.1f}s")

if memory_flag:
    # getrusage returns peak RSS in bytes on Linux, kilobytes on macOS
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import platform
    if platform.system() == 'Darwin':
        peak_mb = peak / 1024 / 1024
    else:
        peak_mb = peak / 1024
    print(f"\nPeak memory usage: {peak_mb:.1f} MB")

print("\n\nDone!\n")