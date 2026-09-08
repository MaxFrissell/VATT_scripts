# VATT Scripts

Repo for various data processing scripts for the VATT4k instrument at the
Vatican Advanced Technology Telescope (VATT).

## reduce

python3 reduce.py dir (-t/--time)

Given a directory of subdirectories of dates (yyyymmdd) with images inside,
does bias subtraction, flat fielding, and defringing. All flat in a filter from a single
night are combined into a master flat, then you choose which night's filter
to use for all data in that filter. Will produce master fringe frames for
any filter set using VR, i, or I. For a given filter of this sort, it will
determing the exposure time with the highest #images * exposure time to
make a fringe frame and subtract from each image using that filter,
scaling the fringe pattern linearly for other exposure times.

-t/--time will print the time the script took at the end.

-m/--memory will print the peak memory usage by the script.

## quick_reduce

python3 quick_reduce.py date_dir1 (date_dir2 etc...) backup_master_flat_dir

Given directories of dates with images inside it will do bias subtraction and
flat fielding. If there are not enough flat images in the directories for a
given filter, it will use the master flat from the backup_master_flat_dir
directory.

date_dir# - Need at least 1. A dir with .fits named as a date in yyyymmdd format.
These images will be reduced using this script.

backup_master_flat_dir - a directory containing .fits files that are master flats
for all filter sets present in the date dirs. These files need to be named: 
upperfilter_lowerfilter_master_flat.fits (same as master flats writted by this
script and the normal reduce.py script). These master flats will be used for
reducing the images if there aren't enough good flats present in the date dirs
for the corresponding filter. clear + filter and filter + clear are treated
as the same filter.

## change_imagetyp

python3 change_imagetyp.py dir #,##-##,##-##,etc. type

Changes the IMAGETYP fits header value to the new type in all listed image
numbers specified by either a list or list of ranges.
