# VATT Scripts

Repo for various data processing scripts for the VATT4k instrument at the
Vatican Advanced Technology Telescope (VATT).

## reduce

python3 reduce.py dir (-t/--time)

Given a directory of subdirectories of dates (yyyymmdd) with images inside,
does bias subtraction and flat fielding. All flat in a filter from a single
night are combined into a master flat, then you choose which night's filter
to use for all data in that filter.

-t/--time will print the time the script took at the end.

## change_imagetyp

python3 change_imagetyp.py dir #,##-##,##-##,etc. type

Changes the IMAGETYP fits header value to the new type in all listed image
numbers specified by either a list or list of ranges.
