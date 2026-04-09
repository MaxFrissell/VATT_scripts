# VATT Data Reduction

Repo for various data reduction scripts for the Vatican Advanced Technology
Telescope (VATT).

## reduce

python3 reduce.py dir (-t/--time)

Given a directory of subdirectories of dates (yyyymmdd) with images inside,
does bias subtraction and flat fielding. All flat in a filter from a single
night are combined into a master flat, then you choose which night's filter
to use for all data in that filter.

-t/--time will print the time the script took at the end.
