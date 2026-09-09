## quick_reduce.py - version of reduce.py intended for quick reduction at the telescope
## Steps:
## Build master biases and flats
## Find master flats in the specified directory for flats we can't make from the data
## De-bias and flatten images
##
## python3 quick_reduce.py date_dir1 (date_dir2 etc...) backup_master_flat_dir

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from photutils.background import Background2D, MedianBackground
import sys
import shutil
import resource
from pathlib import Path
from datetime import datetime
import re
import time
from itertools import combinations
from collections import defaultdict

def parse_filter(filter_str):
    upper = re.search(r'upper:\s*(\S+)', filter_str).group(1)
    lower = re.search(r'lower:\s*(\S+)', filter_str).group(1)
    return upper, lower

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

args = [a for a in sys.argv[1:]]
if(len(args) == 1):
    raise ValueError("Needs at least two command line arguments, one dir of images, one of master flats")

im_dirs = [Path(dir) for dir in args[:-1]]

def is_date_dirname(name, fmt="%Y%m%d"):
    """Return True if `name` parses as a date in the given format (default YYYYMMDD)."""
    try:
        datetime.strptime(name, fmt)
        return True
    except ValueError:
        return False

## -----------------------------------
## script is right through here

# The last command line argument is a directory of backup master flats to
# fall back on for any filter set we can't build a master flat for from
# the date directories themselves.
backup_flat_dir = Path(args[-1])

# Each remaining argument is now a date directory itself (images live
# directly inside it, not nested under some parent). We no longer scan
# for date-named subdirectories -- just sanity-check the name and move on.
for d in im_dirs:
    if not is_date_dirname(d.name):
        print(f"Warning: '{d.name}' doesn't look like a YYYYMMDD date directory, processing anyway")

date_dir_lookup = {d.name: d for d in im_dirs}

# Timestamp this reduction run (UTC), once, at the start of the script.
# Every "reduced" output directory below is named from this same moment,
# so the date in the folder name is the date of the reduction itself,
# not the date of the observations.
run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H-%M")
reduced_dirname = f"reduced_{run_timestamp}"

# ============================================================================
# Scan each date directory for images and classify them by type
# ============================================================================
bias_files_by_dir = defaultdict(list)      # {date_name: [Path, ...]}
flat_files_by_dir = defaultdict(list)
science_files_by_dir = defaultdict(list)

print(f"\nScanning {len(im_dirs)} date director{'y' if len(im_dirs) == 1 else 'ies'} for images...")

for date_dir in im_dirs:
    date_name = date_dir.name
    files = sorted(date_dir.rglob("*.fits"))

    for im_file in files:
        name = im_file.name

        # Skip anything inside a previous run's output directory (e.g.
        # reduced_20260908_14-32/), so re-running over the same date
        # directory doesn't re-ingest already-reduced files as raw data.
        rel_parts = im_file.relative_to(date_dir).parts[:-1]
        if any(part.startswith("reduced") for part in rel_parts):
            continue

        if (name[0] == 'm') or (name[0:4] == 'test'):
            print(f"Throwing out {im_file}")
            continue

        with fits.open(im_file, memmap=True) as hdu_list:
            header = hdu_list[0].header
            imagetyp = header.get('IMAGETYP', 'unknown')

        if imagetyp == 'zero':
            bias_files_by_dir[date_name].append(im_file)
        elif imagetyp == 'flat':
            flat_files_by_dir[date_name].append(im_file)
        elif imagetyp == 'object':
            science_files_by_dir[date_name].append(im_file)

    (date_dir / reduced_dirname).mkdir(exist_ok=True)

n_bias = sum(len(v) for v in bias_files_by_dir.values())
n_flat = sum(len(v) for v in flat_files_by_dir.values())
n_sci = sum(len(v) for v in science_files_by_dir.values())
print(f"Found {n_bias} biases, {n_flat} flats, {n_sci} science images\n")

# ============================================================================
# PASS 1: Build master biases
# ============================================================================
print("=" * 60)
print("PASS 1: BUILDING MASTER BIASES")
print("=" * 60)

unique_dirs = [d.name for d in im_dirs]
master_biases = {}  # {date_name: (master_amp1, master_amp2)} or date_name: None

for date_name in unique_dirs:
    bias_files_for_date = bias_files_by_dir.get(date_name, [])

    if len(bias_files_for_date) == 0:
        master_biases[date_name] = None
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
        master_biases[date_name] = None
        del bias_chips_temp
        continue

    if len(keepers) % 2 == 0:
        keepers = keepers[1:]

    master_amp1 = np.median(np.stack([c[0] for c in keepers], axis=0), axis=0)
    master_amp2 = np.median(np.stack([c[1] for c in keepers], axis=0), axis=0)
    master_biases[date_name] = (master_amp1, master_amp2)

    out_path = date_dir_lookup[date_name] / reduced_dirname
    stitched = stitch_for_output(master_amp1, master_amp2)
    fits.writeto(out_path / "master_bias.fits", stitched, overwrite=True)

    del bias_chips_temp, keepers

print(f"\nWrote master biases to each date directory's {reduced_dirname}/ folder")


def find_nearest_bias(target_dir, master_biases_dict):
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

    bias_path = date_dir_lookup[nearest_dir] / reduced_dirname / "master_bias.fits"
    with fits.open(bias_path) as hdul:
        stitched = hdul[0].data
        unstitched = np.flipud(stitched)
        amp2, amp1 = np.array_split(unstitched, 2, axis=0)
        return (amp1, amp2)


def canonical_filter_key(upper, lower):
    """
    Treat 'clear + X' and 'X + clear' as the same filter for reduction
    purposes, since a clear element contributes no filtering. Returns
    (X, 'clear') with the non-clear filter always first, or (upper, lower)
    unchanged if neither/both slots are clear.
    """
    upper_is_clear = upper.lower() == 'clear'
    lower_is_clear = lower.lower() == 'clear'
    if lower_is_clear and not upper_is_clear:
        return (upper, 'clear')
    elif upper_is_clear and not lower_is_clear:
        return (lower, 'clear')
    else:
        return (upper, lower)


def find_backup_master_flat(filter_key, backup_dir):
    """
    Look for a master flat matching filter_key (already canonicalized) in
    backup_dir. Backup flats may have been named with either filter order,
    so both orderings of the filename are tried before falling back to
    parsing every *_master_flat.fits file in the directory.
    """
    upper, lower = filter_key
    candidates = [
        backup_dir / f"{upper}_{lower}_master_flat.fits",
        backup_dir / f"{lower}_{upper}_master_flat.fits",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in sorted(backup_dir.glob("*_master_flat.fits")):
        stem = candidate.name[: -len("_master_flat.fits")]
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        if canonical_filter_key(parts[0], parts[1]) == filter_key:
            return candidate

    return None


def load_master_flat_file(flat_path):
    """Load a previously-written master flat FITS file back into (amp1, amp2) chips."""
    with fits.open(flat_path) as hdul:
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


flats_by_filter = {}  # {canonical_filter_key: {date_name: [(amp1, amp2), ...]}}
for date_name, flats_for_date in flat_files_by_dir.items():
    for flat_file in flats_for_date:
        amp1, amp2, header = read_and_prep_image(flat_file)

        master_bias = find_nearest_bias(date_name, master_biases)
        if master_bias:
            amp1 = amp1 - master_bias[0]
            amp2 = amp2 - master_bias[1]

        upper, lower = parse_filter(header['FILTER'])
        filter_key = canonical_filter_key(upper, lower)

        if filter_key not in flats_by_filter:
            flats_by_filter[filter_key] = {}
        if date_name not in flats_by_filter[filter_key]:
            flats_by_filter[filter_key][date_name] = []

        flats_by_filter[filter_key][date_name].append((amp1, amp2))

# Shared output location for combined master flats -- flats built from the
# data can draw on more than one of the date directories passed in, so
# there's no single natural directory for them. They're written under the
# first date directory given on the command line.
master_flats_out_dir = im_dirs[0] / reduced_dirname / "master_flats"
master_flats_out_dir.mkdir(parents=True, exist_ok=True)

master_flats = {}  # {canonical_filter_key: (master_amp1, master_amp2)}

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

    if filter_key in master_flats:
        mf_amp1, mf_amp2 = master_flats[filter_key]
        stitched = stitch_for_output(mf_amp1, mf_amp2)
        out_name = f"{upper}_{lower}_master_flat.fits"
        fits.writeto(master_flats_out_dir / out_name, stitched, overwrite=True)

print(f"\nWrote master flats built from the data to {master_flats_out_dir}")

# ---- Fall back to the backup master flat directory for any filter sets ----
# ---- seen in the science data that we couldn't build a master for. --------
science_filter_keys = set()
for date_name, sci_files in science_files_by_dir.items():
    for sci_file in sci_files:
        with fits.open(sci_file, memmap=True) as hdul:
            header = hdul[0].header
            try:
                upper, lower = parse_filter(header['FILTER'])
            except (KeyError, AttributeError):
                continue
        science_filter_keys.add(canonical_filter_key(upper, lower))

missing_filter_keys = science_filter_keys - set(master_flats.keys())
for filter_key in sorted(missing_filter_keys):
    backup_path = find_backup_master_flat(filter_key, backup_flat_dir)
    if backup_path is not None:
        master_flats[filter_key] = load_master_flat_file(backup_path)
        print(f"Using backup master flat for {filter_key} from {backup_path.name}")
    else:
        print(f"No master flat available (data or backup) for {filter_key}")

# ============================================================================
# PASS 3: Reduce science images (debias + flatten only, no fringe correction)
# ============================================================================
print("\n" + "=" * 60)
print("PASS 3: REDUCING SCIENCE IMAGES")
print("=" * 60)

total_sci = sum(len(v) for v in science_files_by_dir.values())
print(f"Processing {total_sci} science images...\n")

processed_count = 0
for date_name, sci_files_for_date in science_files_by_dir.items():
    master_bias = find_nearest_bias(date_name, master_biases)

    for sci_file in sci_files_for_date:
        amp1, amp2, header = read_and_prep_image(sci_file)

        # Bias subtract
        amp1 = amp1.astype(float) - master_bias[0]
        amp2 = amp2.astype(float) - master_bias[1]

        # Flat field
        upper, lower = parse_filter(header['FILTER'])
        filter_key = canonical_filter_key(upper, lower)

        if filter_key not in master_flats:
            print(f"No master flat for {filter_key}, skipping {sci_file.name}")
            continue

        mf_amp1, mf_amp2 = master_flats[filter_key]
        amp1 /= mf_amp1
        amp2 /= mf_amp2

        # Stitch
        reduced = stitch_for_output(amp1, amp2)

        # Write
        out_dir = date_dir_lookup[date_name] / reduced_dirname
        out_name = "red_" + sci_file.name
        fits.writeto(out_dir / out_name, reduced, header, overwrite=True)

        processed_count += 1
        if processed_count % 10 == 0:
            print(f"  Processed {processed_count} science images...")

print(f"\nWrote {processed_count} reduced science images")

print("\n\nDone!\n")