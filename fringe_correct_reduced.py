#!/usr/bin/env python
"""
Extract fringe patterns from reduced science images by computing sigma-clipped mean
for each day and filter combination.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
import re
from collections import defaultdict


def parse_filter(filter_str):
    """Extract upper and lower filter names from filter string."""
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


def apply_sky_background_subtraction(image_data, box_size=60):
    """Apply sky background subtraction using Background2D."""
    from photutils.background import Background2D, MedianBackground
    
    bkg = Background2D(
        image_data,
        box_size=(box_size, box_size),
        filter_size=(3, 3),
        bkg_estimator=MedianBackground()
    )
    
    return image_data - bkg.background


def extract_min_date_from_reduced_dir(reduced_dir):
    """
    Extract the minimum date from subdirectories in reduced directory.
    
    Parameters
    ----------
    reduced_dir : Path
        Path to reduced directory
    
    Returns
    -------
    str or None
        Minimum date directory name found, or None if no dates found
    """
    reduced_dir = Path(reduced_dir)
    date_dirs = [d.name for d in reduced_dir.iterdir() if d.is_dir() and len(d.name) == 8 and d.name.isdigit()]
    
    if not date_dirs:
        return None
    
    return sorted(date_dirs)[0]


def extract_fringe_frames(reduced_dir, date_label):
    """
    Create master fringe frames by 3-sigma clipping all images in each
    filter/exptime combination regardless of date.
    
    Parameters
    ----------
    reduced_dir : str or Path
        Path to the 'reduced' directory produced by reduce.py
    date_label : str
        Label for output directory (typically the earliest date in this reduced dir)
    
    Returns
    -------
    dict
        Dictionary mapping (filter_key, exptime) to fringe frame data
    """
    reduced_dir = Path(reduced_dir)
    if not reduced_dir.exists():
        raise FileNotFoundError(f"Reduced directory not found: {reduced_dir}")
    
    # Output directory
    output_dir = Path.home() / "Desktop" / "reduction_test" / f"run_starting_on_{date_label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing reduced directory: {reduced_dir}")
    print(f"Output label: {date_label}")
    print(f"Output directory: {output_dir}\n")
    
    # Find all reduced science images
    sci_files = sorted(reduced_dir.glob("*/red_*.fits"))
    
    if not sci_files:
        print(f"  No reduced science images found in {reduced_dir}\n")
        return {}
    
    print(f"Found {len(sci_files)} reduced science images\n")
    
    # Organize by filter and exptime (across all dates)
    images_by_filter_exptime = defaultdict(list)
    
    for sci_file in sci_files:
        with fits.open(sci_file) as hdul:
            header = hdul[0].header
            try:
                upper, lower = parse_filter(header['FILTER'])
                filter_key = (upper, lower)
                exptime = header.get('EXPTIME', None)
                
                if exptime is None:
                    print(f"Warning: No EXPTIME in {sci_file.name}, skipping")
                    continue
                
                key = (filter_key, exptime)
                images_by_filter_exptime[key].append((sci_file, header))
            except (KeyError, AttributeError):
                print(f"Warning: Could not parse header for {sci_file.name}, skipping")
                continue
    
    # Process each filter/exptime combination
    results = {}
    
    for (filter_key, exptime) in sorted(images_by_filter_exptime.keys()):
        upper, lower = filter_key
        image_data = images_by_filter_exptime[(filter_key, exptime)]
        
        print(f"Processing {upper} + {lower}, ExpTime={exptime}s ({len(image_data)} images)")
        
        # Load and apply sky subtraction to all images
        sky_subtracted_images = []
        
        for img_file, header in image_data:
            # Load the stitched image
            with fits.open(img_file) as hdul:
                stitched_image = hdul[0].data
            
            # Apply sky background subtraction
            sky_subtracted = apply_sky_background_subtraction(stitched_image, box_size=60)
            sky_subtracted_images.append(sky_subtracted)
        
        # Compute sigma-clipped mean across all images
        stacked = np.stack(sky_subtracted_images, axis=0)
        clipped = sigma_clip(stacked, sigma=3, axis=0)
        fringe_frame = np.ma.mean(clipped, axis=0).data
        
        # Write output
        out_subdir = output_dir / f"{upper}_{lower}_t{exptime:.0f}"
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_path = out_subdir / "master_fringe_frame.fits"
        fits.writeto(out_path, fringe_frame, header=header, overwrite=True)
        
        print(f"  Wrote master fringe frame: {out_path.relative_to(Path.home())}\n")
        
        results[(filter_key, exptime)] = fringe_frame
    
    print(f"Processed {len(results)} filter/exptime combinations for {date_label}\n")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract master fringe patterns from multiple reduced directories"
    )
    parser.add_argument(
        "reduced_dirs",
        nargs="+",
        help="One or more paths to 'reduced' directories produced by reduce.py"
    )
    
    args = parser.parse_args()
    
    # Sort reduced directories by their minimum date
    reduced_dirs_with_dates = []
    for red_dir in args.reduced_dirs:
        min_date = extract_min_date_from_reduced_dir(red_dir)
        if min_date:
            reduced_dirs_with_dates.append((min_date, red_dir))
        else:
            print(f"Warning: Could not find date directories in {red_dir}, skipping\n")
    
    # Sort by minimum date
    reduced_dirs_with_dates.sort(key=lambda x: x[0])
    
    print(f"Processing {len(reduced_dirs_with_dates)} reduced directories in date order:\n")
    for min_date, red_dir in reduced_dirs_with_dates:
        extract_fringe_frames(red_dir, min_date)
    
    # Create across_all master fringe frames
    print("\n" + "="*60)
    print("Creating master fringe frames across all directories")
    print("="*60 + "\n")
    
    create_across_all_fringe_frames(reduced_dirs_with_dates)


def create_across_all_fringe_frames(reduced_dirs_with_dates):
    """
    Create master fringe frames combining all images from all reduced directories.
    
    Parameters
    ----------
    reduced_dirs_with_dates : list of tuples
        List of (min_date, reduced_dir) tuples
    """
    # Output directory
    output_dir = Path.home() / "Desktop" / "reduction_test" / "across_all"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    # Collect images from all directories
    images_by_filter_exptime = defaultdict(list)
    
    for min_date, reduced_dir in reduced_dirs_with_dates:
        reduced_dir = Path(reduced_dir)
        sci_files = sorted(reduced_dir.glob("*/red_*.fits"))
        
        for sci_file in sci_files:
            with fits.open(sci_file) as hdul:
                header = hdul[0].header
                try:
                    upper, lower = parse_filter(header['FILTER'])
                    filter_key = (upper, lower)
                    exptime = header.get('EXPTIME', None)
                    
                    if exptime is None:
                        continue
                    
                    key = (filter_key, exptime)
                    images_by_filter_exptime[key].append((sci_file, header))
                except (KeyError, AttributeError):
                    continue
    
    # Process each filter/exptime combination
    for (filter_key, exptime) in sorted(images_by_filter_exptime.keys()):
        upper, lower = filter_key
        image_data = images_by_filter_exptime[(filter_key, exptime)]
        
        print(f"Processing {upper} + {lower}, ExpTime={exptime}s ({len(image_data)} images across all runs)")
        
        # Load and apply sky subtraction to all images
        sky_subtracted_images = []
        
        for img_file, header in image_data:
            # Load the stitched image
            with fits.open(img_file) as hdul:
                stitched_image = hdul[0].data
            
            # Apply sky background subtraction
            sky_subtracted = apply_sky_background_subtraction(stitched_image, box_size=60)
            sky_subtracted_images.append(sky_subtracted)
        
        # Compute sigma-clipped mean across all images
        stacked = np.stack(sky_subtracted_images, axis=0)
        clipped = sigma_clip(stacked, sigma=3, axis=0)
        fringe_frame = np.ma.mean(clipped, axis=0).data
        
        # Write output
        out_subdir = output_dir / f"{upper}_{lower}_t{exptime:.0f}"
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_path = out_subdir / "master_fringe_frame.fits"
        fits.writeto(out_path, fringe_frame, header=header, overwrite=True)
        
        print(f"  Wrote master fringe frame: {out_path.relative_to(Path.home())}\n")
    
    print(f"Processed {len(images_by_filter_exptime)} filter/exptime combinations across all directories")



if __name__ == "__main__":
    main()

