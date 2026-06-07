## Memory-efficient version of reduce.py
## Steps:
## Build master biases and flats
## De-bias and flatten images
## Build master fringe frames for filters using VR, i, or I
## De-fringe and write.
##
## python3 reduce.py path_to_day_subdirs (-t or --time) (-m or --memory)

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from photutils.background import Background2D, MedianBackground
import sys
import resource
from pathlib import Path
from datetime import datetime
import re
import time
from itertools import combinations
from collections import defaultdict

start_time = time.time()
elapsed_before_input = 0

# Filters that need fringe correction (lower filter, case-sensitive)
FRINGE_LOWER_FILTERS = {'VR', 'i', 'I'}
FRINGE_BOX_SIZE = 500


def parse_filter(filter_str):
    upper = re.search(r'upper:\s*(\S+)', filter_str).group(1)
    lower = re.search(r'lower:\s*(\S+)', filter_str).group(1)
    return upper, lower


def needs_fringe(upper, lower):
    """Return True if either the upper or lower filter is in the fringe set (VR/i/I)."""
    return upper in FRINGE_LOWER_FILTERS or lower in FRINGE_LOWER_FILTERS


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


def sky_subtract_stitched(image_data, box_size=FRINGE_BOX_SIZE):
    """
    Subtract a coarse 2-D sky background from a stitched image.
    Used only for master fringe construction — not applied to final output.
    """
    bkg = Background2D(
        image_data,
        box_size=(box_size, box_size),
        filter_size=(3, 3),
        bkg_estimator=MedianBackground(),
        exclude_percentile=50.0,
    )
    return image_data - bkg.background


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

time_flag   = '-t' in sys.argv or '--time'   in sys.argv
memory_flag = '-m' in sys.argv or '--memory' in sys.argv
args = [a for a in sys.argv[1:] if a not in ('-t', '--time', '-m', '--memory')]

im_dir = Path(args[0])
im_files = list(im_dir.rglob("*.fits"))

# Filter out reduced, master, and test images
temp = []
for file in im_files:
    name = file.name
    if 'reduced' in file.parts:
        pass
    elif (name[0] == 'm') or (name[0:4] == 'test'):
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
        filter_key = (upper, lower)

        if filter_key not in flats_by_filter:
            flats_by_filter[filter_key] = {}
        if date_dir not in flats_by_filter[filter_key]:
            flats_by_filter[filter_key][date_dir] = []

        flats_by_filter[filter_key][date_dir].append((amp1, amp2))

Path(f"{im_dir}/reduced/master_flats").mkdir(parents=True, exist_ok=True)

master_flats = {}  # {(upper, lower): (master_amp1, master_amp2)}

for filter_key, nights_dict in flats_by_filter.items():
    upper, lower = filter_key
    nights = sorted(nights_dict.keys())

    if len(nights) == 1:
        night = nights[0]
        chip_pairs = nights_dict[night]
        master_flat, num_frames = make_master_flat_from_raw_flats(chip_pairs)
        if master_flat:
            master_flats[filter_key] = master_flat
            print(f"Master flat for upper={upper} lower={lower} from {num_frames} frames on {night}")
        else:
            print(f"No good flats for upper={upper} lower={lower}")
    else:
        combinations_list = generate_date_combinations(nights)

        print(f"\nMultiple nights with flats for upper={upper} lower={lower}:")
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
# PASS 3: Build master fringe frames (memory-efficient two-pass Welford's)
# ============================================================================
print("\n" + "=" * 60)
print("PASS 3: BUILDING MASTER FRINGE FRAMES")
print("=" * 60)

# Identify which filter keys need fringe correction and have a master flat
fringe_filter_keys = [fk for fk in master_flats if needs_fringe(fk[0], fk[1])]

# For each fringe filter key, find the best exposure time across all nights
# (best = highest n_frames * exptime)
science_files_by_dir = defaultdict(list)
for sci_file in science_files:
    date_dir = sci_file.parts[-2]
    science_files_by_dir[date_dir].append(sci_file)

# Scan headers to build exptime tallies per fringe filter key
# (header-only scan, no pixel data loaded)
fringe_exptime_counts = {}   # {filter_key: {exptime: {'count': n, 'total': t}}}
fringe_sci_records  = []     # list of dicts for science files needing fringing

print("Scanning science headers for fringe filter sets...")
for date_dir in sorted(science_files_by_dir.keys()):
    for sci_file in science_files_by_dir[date_dir]:
        with fits.open(sci_file, memmap=True) as hdul:
            header = hdul[0].header
            try:
                upper, lower = parse_filter(header['FILTER'])
            except (KeyError, AttributeError):
                continue
            filter_key = (upper, lower)
            if filter_key not in fringe_filter_keys:
                continue
            exptime = header.get('EXPTIME')
            if exptime is None:
                continue
            exptime = float(exptime)

            if filter_key not in fringe_exptime_counts:
                fringe_exptime_counts[filter_key] = defaultdict(lambda: {'count': 0, 'total': 0.0})
            fringe_exptime_counts[filter_key][exptime]['count'] += 1
            fringe_exptime_counts[filter_key][exptime]['total'] += exptime

            fringe_sci_records.append({
                'path':       sci_file,
                'date':       date_dir,
                'upper':      upper,
                'lower':      lower,
                'filter_key': filter_key,
                'exptime':    exptime,
            })

# Choose best exptime per filter key
best_exptimes = {}   # {filter_key: float}
for fk in fringe_filter_keys:
    if fk not in fringe_exptime_counts:
        print(f"  No science images found for {fk}, skipping fringe correction")
        continue
    counts = fringe_exptime_counts[fk]
    chosen = max(counts, key=lambda et: counts[et]['total'])
    info = counts[chosen]
    best_exptimes[fk] = chosen
    upper, lower = fk
    print(f"  {upper}+{lower}: best exptime {chosen:.0f}s "
          f"({info['count']} frames, {info['total']:.0f}s total)")

# -----------------------------------------------------------------
# Disk-based strip-wise sigma-clipped mean.
#
# Step 1: debias, flatten, sky-subtract each frame and write to a
#         temp FITS file under reduced/temp/. One file per frame,
#         never more than one frame in memory at once.
# Step 2: read the temp files back in horizontal strips of
#         STRIP_HEIGHT rows. For each strip, load that strip from
#         every temp file, sigma-clip across the frame axis, take
#         the mean, and write the strip into the master fringe.
#         Peak memory = N_frames * STRIP_HEIGHT * image_width * 4 bytes.
# Step 3: delete all temp files for this filter key.
# -----------------------------------------------------------------

STRIP_HEIGHT = 64
SIGMA_THRESH = 3.0

Path(f"{im_dir}/reduced/master_fringes").mkdir(parents=True, exist_ok=True)
temp_dir = Path(f"{im_dir}/reduced/temp")
temp_dir.mkdir(parents=True, exist_ok=True)
master_fringes = {}   # {filter_key: ndarray}

for fk in fringe_filter_keys:
    if fk not in best_exptimes:
        continue

    upper, lower = fk
    master_et = best_exptimes[fk]
    master_files = [
        r for r in fringe_sci_records
        if r['filter_key'] == fk and r['exptime'] == master_et
    ]

    print(f"\n  Building master fringe for {upper}+{lower} "
          f"({len(master_files)} frames at {master_et:.0f}s)...")

    # ---- Step 1: write sky-subtracted temp files ----
    temp_paths = []
    image_shape = None

    print(f"  Writing sky-subtracted temp files...")
    for i, rec in enumerate(master_files):
        stitched, _ = debias_and_flatten(
            rec['path'],
            find_nearest_bias(rec['date'], list(master_biases.keys()), master_biases),
            master_flats[fk],
        )
        sky_sub = sky_subtract_stitched(stitched)
        del stitched

        if image_shape is None:
            image_shape = sky_sub.shape

        temp_path = temp_dir / f"tmp_fringe_{upper}_{lower}_{i:04d}.fits"
        fits.writeto(temp_path, sky_sub.astype(np.float32), overwrite=True)
        temp_paths.append(temp_path)
        del sky_sub

    if not temp_paths or image_shape is None:
        print(f"  No frames written for {upper}+{lower}, skipping")
        continue

    n_rows, n_cols = image_shape
    master_fringe = np.zeros(image_shape, dtype=np.float64)

    # ---- Step 2: strip-wise sigma-clipped mean ----
    print(f"  Computing strip-wise sigma-clipped mean "
          f"({n_rows // STRIP_HEIGHT + 1} strips)...")

    for row_start in range(0, n_rows, STRIP_HEIGHT):
        row_end = min(row_start + STRIP_HEIGHT, n_rows)

        # Load this strip from every temp file — shape (N_frames, strip_h, n_cols)
        strips = np.stack(
            [fits.getdata(p, memmap=False)[row_start:row_end].astype(np.float32)
             for p in temp_paths],
            axis=0
        )

        # Sigma-clip across frame axis (axis=0), same as original approach
        clipped = sigma_clip(strips, sigma=SIGMA_THRESH, axis=0)
        master_fringe[row_start:row_end] = np.ma.mean(clipped, axis=0).data
        del strips, clipped

    # ---- Step 3: clean up temp files for this filter key ----
    for p in temp_paths:
        p.unlink()

    master_fringes[fk] = master_fringe

    # Write master fringe
    out_name = f"master_fringe_{upper}_{lower}_{master_et:.0f}s.fits"
    fits.writeto(im_dir / "reduced" / "master_fringes" / out_name,
                 master_fringe, overwrite=True)
    print(f"  Saved: {im_dir}/reduced/master_fringes/{out_name}")

# Remove temp directory if empty
try:
    temp_dir.rmdir()
except OSError:
    pass

print(f"\nWrote master fringe frames to {im_dir}/reduced/master_fringes")

# ============================================================================
# PASS 4: Reduce science images and apply fringe correction where needed
# ============================================================================
print("\n" + "=" * 60)
print("PASS 4: REDUCING SCIENCE IMAGES")
print("=" * 60)
print(f"Processing {len(science_files)} science images...\n")

processed_count = 0
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
        filter_key = (upper, lower)

        if filter_key not in master_flats:
            print(f"No master flat for {filter_key}, skipping {sci_file.name}")
            continue

        mf_amp1, mf_amp2 = master_flats[filter_key]
        amp1 /= mf_amp1
        amp2 /= mf_amp2

        # Stitch
        reduced = stitch_for_output(amp1, amp2)

        # Fringe correction (only for target filter sets with a master fringe)
        if filter_key in master_fringes:
            exptime = header.get('EXPTIME')
            if exptime is not None:
                master_et = best_exptimes[filter_key]
                scale = float(exptime) / master_et
                reduced -= scale * master_fringes[filter_key]

        # Write
        out_dir = im_dir / "reduced" / date_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = "red_" + sci_file.name
        fits.writeto(out_dir / out_name, reduced, header, overwrite=True)

        processed_count += 1
        if processed_count % 10 == 0:
            print(f"  Processed {processed_count} science images...")

print(f"\nWrote {processed_count} reduced science images")

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