#!/usr/bin/env python3
"""
Script to change IMAGETYP header values in FITS images.

Usage:
    python3 change_imagetyp.py <directory> <numbers> <new_value>

Arguments:
    directory: Path to directory containing FITS files
    numbers: Comma-separated list of numbers or ranges (with or without leading zeros)
    new_value: New value for IMAGETYP header

Examples:
    change_imagetyp.py ./images 1-5 DARK
    change_imagetyp.py ./images 1-5,10-15,20 BIAS
    change_imagetyp.py ./images 5,7,8,10,100 FLAT
"""

import sys
import os
from pathlib import Path
from astropy.io import fits


def parse_ranges(ranges_str):
    """
    Parse a number specification string into a list of file numbers.
    Accepts ranges and individual numbers, with or without leading zeros.
    
    Examples:
        "1-5" -> [1, 2, 3, 4, 5]
        "1-5,10-15" -> [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        "5,7,8,10,100" -> [5, 7, 8, 10, 100]
        "5" -> [5]
    """
    numbers = set()
    
    # Split by comma for multiple ranges
    parts = ranges_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range like "1-5"
            try:
                start, end = part.split('-')
                start, end = int(start.strip()), int(end.strip())
                numbers.update(range(start, end + 1))
            except ValueError:
                print(f"Error: Invalid range format '{part}'")
                sys.exit(1)
        else:
            # Handle single number
            try:
                numbers.add(int(part))
            except ValueError:
                print(f"Error: Invalid number '{part}'")
                sys.exit(1)
    
    return sorted(numbers)


def find_fits_files(directory, numbers):
    """
    Find FITS files in directory that match the given number suffixes.
    Files are expected to end with ####.fits (4-digit numbers with leading zeros).
    Matches files of the form *####.fits (prefix is allowed).
    """
    dir_path = Path(directory)
    
    if not dir_path.is_dir():
        print(f"Error: Directory '{directory}' not found")
        sys.exit(1)
    
    files = []
    for num in numbers:
        # Format number with leading zeros (4 digits) and search with wildcard
        num_str = f"{num:04d}"
        pattern = f"*{num_str}.fits"
        
        matches = list(dir_path.glob(pattern))
        
        if len(matches) == 0:
            print(f"Warning: No file matching '{pattern}' found in '{directory}'")
        elif len(matches) > 1:
            print(f"Error: Multiple files match '{pattern}' in '{directory}':")
            for m in matches:
                print(f"  - {m.name}")
            sys.exit(1)
        else:
            files.append(matches[0])
    
    return files


def change_imagetyp(fits_files, new_value):
    """
    Change IMAGETYP header value in the given FITS files.
    """
    if not fits_files:
        print("No matching FITS files found.")
        return
    
    print(f"Changing IMAGETYP to '{new_value}' for {len(fits_files)} file(s)...")
    
    for filepath in fits_files:
        try:
            with fits.open(filepath, mode='update') as hdul:
                # Update IMAGETYP in the primary HDU
                hdul[0].header['IMAGETYP'] = new_value
            print(f"  Updated: {filepath.name}")
        except Exception as e:
            print(f"  Error updating {filepath.name}: {e}")


def main():
    if len(sys.argv) != 4:
        print("Usage: change_imagetyp.py <directory> <numbers> <new_value>")
        print("\nExamples:")
        print("  change_imagetyp.py ./images 1-5 DARK")
        print("  change_imagetyp.py ./images 1-5,10-15 BIAS")
        print("  change_imagetyp.py ./images 5,7,8,10,100 FLAT")
        sys.exit(1)
    
    directory = sys.argv[1]
    ranges_str = sys.argv[2]
    new_value = sys.argv[3]
    
    # Parse the ranges
    numbers = parse_ranges(ranges_str)
    
    # Find the corresponding FITS files
    fits_files = find_fits_files(directory, numbers)
    
    # Change IMAGETYP in the files
    change_imagetyp(fits_files, new_value)


if __name__ == '__main__':
    main()
