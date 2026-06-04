## Memory-efficient version of reduce.py
## Two-pass approach: build masters first, then reduce science images one at a time
## python3 reduce.py path_to_day_subdirs (-t or --time)

import numpy as np 
from astropy.io import fits
from astropy.stats import sigma_clip
import sys
from pathlib import Path
from datetime import datetime
import re
import time
from itertools import combinations
from collections import defaultdict

start_time = time.time()
elapsed_before_input = 0

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

print()

time_flag = '-t' in sys.argv or '--time' in sys.argv
args = [a for a in sys.argv[1:] if a not in ('-t', '--time')]

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
    date_dir = bias_file.parts[-2]  # second-to-last part is date dir
    biases_by_dir[date_dir].append(bias_file)

unique_dirs = sorted(biases_by_dir.keys())
master_biases = {}  # {date_dir: (master_amp1, master_amp2)} or date_dir: None

for date_dir in unique_dirs:
    bias_files_for_date = biases_by_dir[date_dir]
    
    if len(bias_files_for_date) == 0:
        master_biases[date_dir] = None
        continue
    
    # Load biases one at a time, compute stats
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
    
    # Reject outliers
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
        # Clean up memory
        del bias_chips_temp
        continue
    
    if len(keepers) % 2 == 0:
        keepers = keepers[1:]
    
    # Build master bias
    master_amp1 = np.median(np.stack([c[0] for c in keepers], axis=0), axis=0)
    master_amp2 = np.median(np.stack([c[1] for c in keepers], axis=0), axis=0)
    master_biases[date_dir] = (master_amp1, master_amp2)
    
    # Write master bias
    out_path = im_dir / "reduced" / date_dir
    out_path.mkdir(parents=True, exist_ok=True)
    stitched = stitch_for_output(master_amp1, master_amp2)
    fits.writeto(out_path / "master_bias.fits", stitched, overwrite=True)
    
    # Clean up memory
    del bias_chips_temp, keepers

print(f"\nWrote master biases to {im_dir}/reduced")

# Handle missing biases by finding nearest available
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
    
    # Load from disk
    bias_path = im_dir / "reduced" / nearest_dir / "master_bias.fits"
    with fits.open(bias_path) as hdul:
        stitched = hdul[0].data
        # Unstitched: flipud to get back original orientation, split
        unstitched = np.flipud(stitched)
        amp2, amp1 = np.array_split(unstitched, 2, axis=0)
        return (amp1, amp2)

# ============================================================================
# PASS 2: Build master flats
# ============================================================================
print("\n" + "=" * 60)
print("PASS 2: BUILDING MASTER FLATS")
print("=" * 60)

# Organize flat files by date and filter
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

# Group flats by filter and collect all available nights
flats_by_filter = {}
for date_dir, flats_for_date in flats_by_dir.items():
    for flat_file in flats_for_date:
        amp1, amp2, header = read_and_prep_image(flat_file)
        
        # Subtract bias
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
        # Multiple nights available, ask user which to combine
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
                
                # Gather all raw flats from selected nights
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
# PASS 3: Reduce science images one at a time
# ============================================================================
print("\n" + "=" * 60)
print("PASS 3: REDUCING SCIENCE IMAGES")
print("=" * 60)
print(f"Processing {len(science_files)} science images...\n")

science_files_by_dir = defaultdict(list)
for sci_file in science_files:
    date_dir = sci_file.parts[-2]
    science_files_by_dir[date_dir].append(sci_file)

processed_count = 0
for date_dir in sorted(science_files_by_dir.keys()):
    sci_files_for_date = science_files_by_dir[date_dir]
    
    # Load master bias once per date
    master_bias = find_nearest_bias(date_dir, list(master_biases.keys()), master_biases)
    
    for sci_file in sci_files_for_date:
        # Load science image
        amp1, amp2, header = read_and_prep_image(sci_file)
        
        # Subtract bias
        if master_bias:
            amp1 = amp1 - master_bias[0]
            amp2 = amp2 - master_bias[1]
        
        # Apply flat field correction
        upper, lower = parse_filter(header['FILTER'])
        filter_key = (upper, lower)
        
        if filter_key not in master_flats:
            print(f"No master flat for {filter_key}, skipping {sci_file.name}")
            continue
        
        mf_amp1, mf_amp2 = master_flats[filter_key]
        
        # Flat-field each chip
        reduced_amp1 = amp1 / mf_amp1
        reduced_amp2 = amp2 / mf_amp2
        
        # Stitch and write
        reduced = stitch_for_output(reduced_amp1, reduced_amp2)
        
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

print("\n\nDone!\n")
