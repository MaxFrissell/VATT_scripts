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
# import matplotlib.pyplot as plt
# from matplotlib import colors
import sys
from pathlib import Path
from datetime import datetime
import re
import time

# I'd like to do a little timing
start_time = time.time()
elapsed_before_input = 0

def parse_filter(filter_str):
    upper = re.search(r'upper:\s*(\S+)', filter_str).group(1)
    lower = re.search(r'lower:\s*(\S+)', filter_str).group(1)
    return upper, lower

# lets make some space
print()

# parse args
time_flag = '-t' in sys.argv or '--time' in sys.argv
args = [a for a in sys.argv[1:] if a not in ('-t', '--time')]

# get the directory from the args
im_dir = Path(args[0])
im_files = list(im_dir.rglob("*.fits"))  # rglob searches all subdirectories recursively

# get rid of all files that are mimage or test
temp = []
for file in im_files:
    name = file.parts[2]

    # please don't read and process any already reduced data
    if 'reduced' in file.parts:
        pass  # silently skip
    
    elif (name[0] == 'm') or (name[0:4] == 'test'):
        print(f"Throwing out {file}")
    else:
        temp.append(file)

Path(f"{im_dir}/reduced").mkdir(exist_ok=True)

# overwrite
im_files = temp

# collect unique subdirectory paths relative to the input directory
sub_dirs = list({f.parent.relative_to(im_dir) for f in im_files if f.parent != im_dir})

print(f"\nBeginning read-in of {len(im_files)} images. . .")

# load all the fits files and sort the biases, darks, etc. out
all_files = []
bias_paths = []
bias_pixs = []
flat_paths = []
flat_pixs = []
flat_headers = []
science_pixs = []
science_paths = []
science_headers = []
for im_file in im_files:
    # get the stuff
    with fits.open(im_file) as hdu_list:
        header = hdu_list[0].header

        # read and orient each chip
        amp1 = np.flipud(hdu_list[1].data)   # bottom chip, flip vertically
        amp2 = hdu_list[2].data               # top chip, no transform needed

        # stack vertically: amp2 on top, amp1 on bottom
        data = np.concatenate((amp2, amp1), axis=0)

        # I think this will orient the image on sky with North up
        data = np.flipud(data)

        # remove the overscan, I think
        data = data[:, :-24]

    all_files.append((header, data))

    # get path relative to input directory
    rel_path = im_file.relative_to(im_dir)

    # sort into biases, flats, science
    if header['IMAGETYP'] == 'zero':
        bias_pixs.append(data)
        bias_paths.append(rel_path)
    
    elif header['IMAGETYP'] == 'object':
        science_pixs.append(data)
        science_paths.append(rel_path)
        science_headers.append(header)
    
    elif header['IMAGETYP'] == 'flat':
        flat_pixs.append(data)
        flat_paths.append(rel_path)
        flat_headers.append(header)


# figure out which dirs have images
paths_split = [path.parts for path in im_files]
only_dirs = [row[1] for row in paths_split]
unique_dirs = list(set(only_dirs))

# get the date dir for each bias
bias_paths_split = [path.parts for path in bias_paths]
bias_only_dir = [row[0] for row in bias_paths_split]
bias_div = [[] for dir in unique_dirs]

# sort biases by dir
# I know this is inefficient, but I don't care
for i, bias_dir in enumerate(bias_only_dir):
    dir_index = unique_dirs.index(bias_dir)
    bias_div[dir_index].append(bias_pixs[i])

# print(science_paths[0])
# plt.imshow(science_pixs[0], cmap='grey', vmin = 2000, vmax = 10000)
# plt.show()

print("\nBeginning processing biases\n")

no_biases = []
master_biases = []
for i, date_biases in enumerate(bias_div):
    current_dir = unique_dirs[i]
    have_biases = True

    # if there are no biases, just give up
    if len(date_biases) == 0:
        no_biases.append(current_dir)
        master_biases.append([])
        print(f"No biases for {current_dir}")

    # analysis if there are any biases
    else:
        # take means and standard deviation for each bias and the median vals
        means = [np.mean(bias) for bias in date_biases]
        stds = [np.std(bias) for bias in date_biases]
        med_mean = np.median(means)
        med_std = np.median(stds)

        # see if any of the biases are 2x or 1/2 the median bias mean/std, we want to get rid of them
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
        
        # the ones we didn't throw out
        keepers = [im for im, keep in zip(date_biases, keep_list) if keep]

        # if there are not at least 9 biases left for the date, say so
        if len(keepers) < 9:
            have_biases = False
        
        # if there weren't enough biases, have to stop and note it in a list
        if not have_biases:
            print(f"For {current_dir} there are only {len(date_biases)} biases.")
            no_biases.append(current_dir)
            master_biases.append([])

        # if we did have enough biases, lets make a master
        else:

            # if we have an even number of biases, just get rid of one
            if len(keepers) % 2 == 0:
                print(f"Odd number of biases remaining for {current_dir}, dropping one, to {len(keepers) - 1} total")
                keepers = keepers[1:]

            # make master
            master = np.median(np.stack(keepers, axis=0), axis=0)
            master_biases.append(master)

# build a list of dirs that DO have master biases (not list because the masters are np arrays)
available_bias_dirs = [d for d, mb in zip(unique_dirs, master_biases) if not isinstance(mb, list)]

def find_nearest_bias(target_dir, available_bias_dirs, master_biases, unique_dirs):
    target_date = datetime.strptime(target_dir, "%Y%m%d")
    
    # compute distance in days for each available dir
    available_dates = [datetime.strptime(d, "%Y%m%d") for d in available_bias_dirs]
    deltas = [(abs((d - target_date).days), d, i) for i, d in enumerate(available_dates)]
    
    # sort by distance - in case of tie, the later date comes first since it's larger
    deltas.sort(key=lambda x: (x[0], -x[1].timestamp()))
    
    nearest_dir = deltas[0][1].strftime("%Y%m%d")
    nearest_index = unique_dirs.index(nearest_dir)
    
    print(f"Not enough biases for {target_dir}, using master bias from {nearest_dir}")
    return master_biases[nearest_index]

# fill in master_biases for the no_bias dirs
for i, d in enumerate(unique_dirs):
    if d in no_biases:
        master_biases[i] = find_nearest_bias(d, available_bias_dirs, master_biases, unique_dirs)

# write master biases out to im_dir/reduced/date/master_bias.fits
for d, mb in zip(unique_dirs, master_biases):
    out_path = im_dir / "reduced" / d
    out_path.mkdir(parents=True, exist_ok=True)
    fits.writeto(out_path / "master_bias.fits", mb, overwrite=True)

print(f"\nWrote master biases to {im_dir}/reduced")

# sort flats and science frames by unique_dir
# claude wan't to do it in this dictionary way. . . whatever
flats_by_dir = {d: [] for d in unique_dirs}
science_by_dir = {d: [] for d in unique_dirs}

# flats sorting
for path, header, data in zip(flat_paths, flat_headers, flat_pixs):
    d = path.parts[0]
    flats_by_dir[d].append({'path': path, 'header': header, 'data': data})

# science sorting
for path, header, data in zip(science_paths, science_headers, science_pixs):
    d = path.parts[0]
    science_by_dir[d].append({'path': path, 'header': header, 'data': data})

# subtract master bias from flats and science frames
for i, d in enumerate(unique_dirs):
    master_bias = master_biases[i]

    for flat in flats_by_dir[d]:
        flat['data'] = flat['data'] - master_bias

    for science in science_by_dir[d]:
        science['data'] = science['data'] - master_bias

print("Subtracted master biases from flats and science frames")
print("\nMaking master flats")
# organize flats by dir and filter combination
flats_by_dir_filter = {d: {} for d in unique_dirs}

for d in unique_dirs:
    for flat in flats_by_dir[d]:
        upper, lower = parse_filter(flat['header']['FILTER'])
        filter_key = (upper, lower)

        if filter_key not in flats_by_dir_filter[d]:
            flats_by_dir_filter[d][filter_key] = []
        flats_by_dir_filter[d][filter_key].append(flat['data'])

# make master flats for each day/filter combination
master_flats = {d: {} for d in unique_dirs}

for d in unique_dirs:
    for filter_key, frames in flats_by_dir_filter[d].items():
        
        # check median of each frame and filter out bad ones
        good_frames = []
        for frame in frames:
            med = np.median(frame)
            if med < 20000 or med > 50000:
                print(f"Flat for {d} upper={filter_key[0]} lower={filter_key[1]} has median {med:.1f} counts, skipping")
            else:
                good_frames.append(frame)

        if len(good_frames) == 0:
            print(f"No good flats for {d} upper={filter_key[0]} lower={filter_key[1]}, skipping master flat")
            continue

        stack = np.stack([frame / np.median(frame) for frame in good_frames], axis=0)
        clipped = sigma_clip(stack, sigma=3, axis=0)
        master_flat = np.ma.mean(clipped, axis=0).data
        master_flats[d][filter_key] = master_flat

        print(f"Master flat for {d} upper={filter_key[0]} lower={filter_key[1]} from {len(good_frames)} frames")

# collect all master flats across all dates, grouped by filter key
all_master_flats = {}  # filter_key -> list of (date, master_flat)

for d in unique_dirs:
    for filter_key, master_flat in master_flats[d].items():
        if filter_key not in all_master_flats:
            all_master_flats[filter_key] = []
        all_master_flats[filter_key].append((d, master_flat))

# resolve duplicates and write out
Path(f"{im_dir}/reduced/master_flats").mkdir(parents=True, exist_ok=True)

chosen_master_flats = {}  # filter_key -> chosen master flat array

for filter_key, entries in all_master_flats.items():
    upper, lower = filter_key

    if len(entries) == 1:
        chosen = entries[0][1]
    else:
        print(f"\nMultiple master flats for upper={upper} lower={lower}:")
        for j, (d, _) in enumerate(entries):
            print(f"  {j}: {d}")
        
        # pause timer while waiting for user input
        elapsed_before_input += time.time() - start_time
        while True:
            choice = input(f"Which date's master flat to use? Enter number 0-{len(entries)-1}: ")
            if choice.isdigit() and 0 <= int(choice) < len(entries):
                chosen = entries[int(choice)][1]
                break
            print("Invalid choice, try again")
        start_time = time.time()  # reset start for remaining processing

    chosen_master_flats[filter_key] = chosen
    out_name = f"{upper}_{lower}_master_flat.fits"
    fits.writeto(im_dir / "reduced" / "master_flats" / out_name, chosen, overwrite=True)

print(f"\nWrote master flats to {im_dir}/reduced/master_flats")

print("Flat field correcting science images")

# reduce all science images using the globally chosen master flat for each filter
for d in unique_dirs:
    for science in science_by_dir[d]:
        upper, lower = parse_filter(science['header']['FILTER'])
        filter_key = (upper, lower)

        if filter_key not in chosen_master_flats:
            print(f"No master flat for upper={upper} lower={lower}, skipping {science['path']}")
            continue

        reduced = science['data'] / chosen_master_flats[filter_key]

        out_dir = im_dir / "reduced" / d
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = "red_" + science['path'].name
        fits.writeto(out_dir / out_name, reduced, overwrite=True)

print("\nWrote reduced science images")

if time_flag:
    total_time = elapsed_before_input + (time.time() - start_time)
    print(f"\nProcessing time: {total_time:.1f}s")

print("\n\nDone!\n")