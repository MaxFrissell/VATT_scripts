# VATT Scripts

Repo for various data processing scripts for the VATT4k instrument at the
Vatican Advanced Technology Telescope (VATT).

## reduce

python3 reduce.py dir (-t/--time) (-m/--memory) (--no_defringing)

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

--no_defringing skips the defringing step, so all that is done is bias
subtraction and flat fielding

## change_imagetyp

python3 change_imagetyp.py dir #,##-##,##-##,etc. type

Changes the IMAGETYP fits header value to the new type in all listed image
numbers specified by either a list or list of ranges.
