# YCLEAN

An automatic CLEAN python module.

## Requirements

The current version has been tested with:

- `python` 3.7
- `casa`/`mpicasa` 6.5
- `numpy` 1.20.2
- `scipy` 1.6.3
- `astropy` 4.2.1
- `spectral-cube` 0.6.0
- `dask` 2021.04.1

## Running YCLEAN

To run YCLEAN, import the `yclean` function into your script:
```python
from yclean_parallel import yclean
```

At the moment `yclean` runs within python and calls `mpicasa` for the parallel
execution of `tclean`. Future versions will be able to run with `mpirun`.
