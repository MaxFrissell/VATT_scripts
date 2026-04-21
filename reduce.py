## This program is called like:
## python3 reduce.py path_to_day_subdirs (-t or --time)
## -t/--time flag times the reduction (ignoring waiting for user response)
## makes /path_to_subdirs/reduced directory for reduced images
## things to know:
##     turns all flats of a filter on a single night into a master flat for that filter
##     if there are flats for a given filter on multiple nights, it will ask which nights
##         to use in the master flat for that filter and throw out the other nights' flats

import numpy as np 
from astropy.io import fits
from astropy.stats import sigma_clip
import sys
from pathlib import Path
from datetime import datetime
import re
import time

start_time = time.time()
elapsed_before_input = 0

def parse_filter(filter_str):
    upper = re.search(r'upper:\s*(\S+)', filter_str).group(1)
    lower = re.search(r'lower:\s*(\S+)', filter_str).group(1)
    return upper, lower

print()

time_flag = '-t' in sys.argv or '--time' in sys.argv
args = [a for a in sys.argv[1:] if a not in ('-t', '--time')]

im_dir = Path(args[0])
im_files = list(im_dir.rglob("*.fits"))

temp = []
for file in im_files:
    name = file.parts[2]
    if 'reduced' in file.parts:
        pass
    elif (name[0] == 'm') or (name[0:4] == 'test'):
        print(f"Throwing out {file}")
    else:
        temp.append(file)

Path(f"{im_dir}/reduced").mkdir(exist_ok=True)
im_files = temp

sub_dirs = list({f.parent.relative_to(im_dir) for f in im_files if f.parent != im_dir})

print(f"\nBeginning read-in of {len(im_files)} images. . .")

# Load all fits files, keeping chips separate
# Each entry stores amp1 and amp2 independently throughout processing
all_files = []
bias_paths = []
bias_chips = []   # list of (amp1, amp2) tuples
flat_paths = []
flat_chips = []   # list of (amp1, amp2) tuples
flat_headers = []
science_chips = []  # list of (amp1, amp2) tuples
science_paths = []
science_headers = []

for im_file in im_files:
    with fits.open(im_file) as hdu_list:
        header = hdu_list[0].header

        # Read and orient each chip — keep them separate
        amp1 = np.flipud(hdu_list[1].data)   # bottom chip, flip vertically
        amp2 = hdu_list[2].data               # top chip, no transform needed

        # Remove overscan from each chip individually
        amp1 = amp1[:, :-24]
        amp2 = amp2[:, :-24]

    rel_path = im_file.relative_to(im_dir)

    if header['IMAGETYP'] == 'zero':
        bias_chips.append((amp1, amp2))
        bias_paths.append(rel_path)
    elif header['IMAGETYP'] == 'object':
        science_chips.append((amp1, amp2))
        science_paths.append(rel_path)
        science_headers.append(header)
    elif header['IMAGETYP'] == 'flat':
        flat_chips.append((amp1, amp2))
        flat_paths.append(rel_path)
        flat_headers.append(header)

paths_split = [path.parts for path in im_files]
only_dirs = [row[1] for row in paths_split]
unique_dirs = list(set(only_dirs))

bias_paths_split = [path.parts for path in bias_paths]
bias_only_dir = [row[0] for row in bias_paths_split]
bias_div = [[] for _ in unique_dirs]

for i, bias_dir in enumerate(bias_only_dir):
    dir_index = unique_dirs.index(bias_dir)
    bias_div[dir_index].append(bias_chips[i])  # appending (amp1, amp2) tuples

print("\nBeginning processing biases\n")

no_biases = []
master_biases = []  # each entry is (master_amp1, master_amp2) or []

for i, date_biases in enumerate(bias_div):
    current_dir = unique_dirs[i]
    have_biases = True

    if len(date_biases) == 0:
        no_biases.append(current_dir)
        master_biases.append([])
        print(f"No biases for {current_dir}")
    else:
        # Compute stats using combined chip data for outlier rejection,
        # but keep chips separate for the actual master
        means = [np.mean(a1 + a2) / 2 for a1, a2 in date_biases]
        stds  = [np.std(np.concatenate([a1.ravel(), a2.ravel()])) for a1, a2 in date_biases]
        med_mean = np.median(means)
        med_std  = np.median(stds)

        keep_list = []
        for j in range(len(means)):
            keep = True
            if (means[j] > med_mean * 2) or (means[j] < med_mean / 2):
                keep = False
                print(f"One bias from {current_dir} is bad")
            if (stds[j] > med_std * 2) or (stds[j] < med_std / 2):
                keep = False
                print(f"One bias from {current_dir} is bad")
            keep_list.append(keep)

        keepers = [chips for chips, keep in zip(date_biases, keep_list) if keep]

        if len(keepers) < 9:
            have_biases = False

        if not have_biases:
            print(f"For {current_dir} there are only {len(date_biases)} biases.")
            no_biases.append(current_dir)
            master_biases.append([])
        else:
            if len(keepers) % 2 == 0:
                print(f"Odd number of biases remaining for {current_dir}, dropping one, to {len(keepers) - 1} total")
                keepers = keepers[1:]

            # Build master bias for each chip independently
            master_amp1 = np.median(np.stack([c[0] for c in keepers], axis=0), axis=0)
            master_amp2 = np.median(np.stack([c[1] for c in keepers], axis=0), axis=0)
            master_biases.append((master_amp1, master_amp2))

available_bias_dirs = [d for d, mb in zip(unique_dirs, master_biases) if not isinstance(mb, list)]

def find_nearest_bias(target_dir, available_bias_dirs, master_biases, unique_dirs):
    target_date = datetime.strptime(target_dir, "%Y%m%d")
    available_dates = [datetime.strptime(d, "%Y%m%d") for d in available_bias_dirs]
    deltas = [(abs((d - target_date).days), d, i) for i, d in enumerate(available_dates)]
    deltas.sort(key=lambda x: (x[0], -x[1].timestamp()))
    nearest_dir = deltas[0][1].strftime("%Y%m%d")
    nearest_index = unique_dirs.index(nearest_dir)
    print(f"Not enough biases for {target_dir}, using master bias from {nearest_dir}")
    return master_biases[nearest_index]

for i, d in enumerate(unique_dirs):
    if d in no_biases:
        master_biases[i] = find_nearest_bias(d, available_bias_dirs, master_biases, unique_dirs)

# Write master biases — stitch chips here just for the output file
for d, mb in zip(unique_dirs, master_biases):
    out_path = im_dir / "reduced" / d
    out_path.mkdir(parents=True, exist_ok=True)
    # Stitch for writing: amp2 on top, amp1 on bottom, then flipud (same as original)
    stitched = np.flipud(np.concatenate((mb[1], mb[0]), axis=0))
    fits.writeto(out_path / "master_bias.fits", stitched, overwrite=True)

print(f"\nWrote master biases to {im_dir}/reduced")

# Sort flats and science by date dir, carrying chip tuples
flats_by_dir   = {d: [] for d in unique_dirs}
science_by_dir = {d: [] for d in unique_dirs}

for path, header, chips in zip(flat_paths, flat_headers, flat_chips):
    d = path.parts[0]
    flats_by_dir[d].append({'path': path, 'header': header, 'amp1': chips[0], 'amp2': chips[1]})

for path, header, chips in zip(science_paths, science_headers, science_chips):
    d = path.parts[0]
    science_by_dir[d].append({'path': path, 'header': header, 'amp1': chips[0], 'amp2': chips[1]})

# Subtract master bias from each chip independently
for i, d in enumerate(unique_dirs):
    mb_amp1, mb_amp2 = master_biases[i]

    for flat in flats_by_dir[d]:
        flat['amp1'] = flat['amp1'] - mb_amp1
        flat['amp2'] = flat['amp2'] - mb_amp2

    for science in science_by_dir[d]:
        science['amp1'] = science['amp1'] - mb_amp1
        science['amp2'] = science['amp2'] - mb_amp2

print("Subtracted master biases from flats and science frames")
print("\nMaking master flats")

flats_by_dir_filter = {d: {} for d in unique_dirs}

for d in unique_dirs:
    for flat in flats_by_dir[d]:
        upper, lower = parse_filter(flat['header']['FILTER'])
        filter_key = (upper, lower)
        if filter_key not in flats_by_dir_filter[d]:
            flats_by_dir_filter[d][filter_key] = []
        flats_by_dir_filter[d][filter_key].append((flat['amp1'], flat['amp2']))

master_flats = {d: {} for d in unique_dirs}

for d in unique_dirs:
    for filter_key, chip_pairs in flats_by_dir_filter[d].items():

        # Outlier rejection based on whole-frame median (average of both chips)
        good_pairs = []
        for amp1, amp2 in chip_pairs:
            med = (np.median(amp1) + np.median(amp2)) / 2
            if med < 20000 or med > 50000:
                print(f"Flat for {d} upper={filter_key[0]} lower={filter_key[1]} has median {med:.1f} counts, skipping")
            else:
                good_pairs.append((amp1, amp2))

        if len(good_pairs) == 0:
            print(f"No good flats for {d} upper={filter_key[0]} lower={filter_key[1]}, skipping master flat")
            continue

        # Make master flat from
        def make_master_chip(amp1_frames, amp2_frames):
            # Normalize each frame by the median of both chips combined
            stack_amp1 = np.stack([a1 / np.median(np.concatenate([a1.ravel(), a2.ravel()]))
                                   for a1, a2 in zip(amp1_frames, amp2_frames)], axis=0)
            stack_amp2 = np.stack([a2 / np.median(np.concatenate([a1.ravel(), a2.ravel()]))
                                   for a1, a2 in zip(amp1_frames, amp2_frames)], axis=0)
            clipped_amp1 = sigma_clip(stack_amp1, sigma=3, axis=0)
            clipped_amp2 = sigma_clip(stack_amp2, sigma=3, axis=0)
            return np.ma.mean(clipped_amp1, axis=0).data, np.ma.mean(clipped_amp2, axis=0).data

        master_amp1, master_amp2 = make_master_chip([p[0] for p in good_pairs],
                                                    [p[1] for p in good_pairs])
        master_flats[d][filter_key] = (master_amp1, master_amp2)

        print(f"Master flat for {d} upper={filter_key[0]} lower={filter_key[1]} from {len(good_pairs)} frames")

all_master_flats = {}
for d in unique_dirs:
    for filter_key, chip_pair in master_flats[d].items():
        if filter_key not in all_master_flats:
            all_master_flats[filter_key] = []
        all_master_flats[filter_key].append((d, chip_pair))

Path(f"{im_dir}/reduced/master_flats").mkdir(parents=True, exist_ok=True)

chosen_master_flats = {}

for filter_key, entries in all_master_flats.items():
    upper, lower = filter_key

    if len(entries) == 1:
        chosen = entries[0][1]
    else:
        print(f"\nMultiple master flats for upper={upper} lower={lower}:")
        for j, (d, _) in enumerate(entries):
            print(f"  {j}: {d}")

        elapsed_before_input += time.time() - start_time
        while True:
            choice = input(f"Which date's master flat to use? Enter number 0-{len(entries)-1}: ")
            if choice.isdigit() and 0 <= int(choice) < len(entries):
                chosen = entries[int(choice)][1]
                break
            print("Invalid choice, try again")
        start_time = time.time()

    chosen_master_flats[filter_key] = chosen  # (master_amp1, master_amp2)

    # Stitch chips for the output file
    mf_amp1, mf_amp2 = chosen
    stitched = np.flipud(np.concatenate((mf_amp2, mf_amp1), axis=0))
    out_name = f"{upper}_{lower}_master_flat.fits"
    fits.writeto(im_dir / "reduced" / "master_flats" / out_name, stitched, overwrite=True)

print(f"\nWrote master flats to {im_dir}/reduced/master_flats")
print("Flat field correcting science images")

for d in unique_dirs:
    for science in science_by_dir[d]:
        upper, lower = parse_filter(science['header']['FILTER'])
        filter_key = (upper, lower)

        if filter_key not in chosen_master_flats:
            print(f"No master flat for upper={upper} lower={lower}, skipping {science['path']}")
            continue

        mf_amp1, mf_amp2 = chosen_master_flats[filter_key]

        # Flat-field each chip with its own master flat
        reduced_amp1 = science['amp1'] / mf_amp1
        reduced_amp2 = science['amp2'] / mf_amp2

        # Stitch here at the very end
        reduced = np.flipud(np.concatenate((reduced_amp2, reduced_amp1), axis=0))

        out_dir = im_dir / "reduced" / d
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = "red_" + science['path'].name
        fits.writeto(out_dir / out_name, reduced, reduced_amp2.header if hasattr(reduced_amp2, 'header') else science['header'], overwrite=True)

print("\nWrote reduced science images")

if time_flag:
    total_time = elapsed_before_input + (time.time() - start_time)
    print(f"\nProcessing time: {total_time:.1f}s")

print("\n\nDone!\n")