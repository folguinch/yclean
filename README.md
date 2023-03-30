# YCLEAN

An automatic CLEAN python module.
This is a modified version of the original [YCLEAN](https://zenodo.org/record/1216881).

## Installation

To install YCLEAN and its dependencies run:

```bash
pip install git+https://github.com/folguinch/yclean.git --user
```

The current version has been tested with `python 3.8` and `casa 6.5.3`.
At the moment to run `tclean` in parallel it is necessary to install CASA from the `tar` file distribution (the `pip` command above should install the modular version of CASA too), and insert the CASA `bin` directory into your `PATH`, e.g.:

```bash
export PATH="/path/to/casa-6.5.3-28-py3.8/bin/:$PATH"
```

Optional dependencies:

- `psutil`: for additional feedback on the resources (memory) used during the mask making step.

## Running YCLEAN

YCLEAN can be used in 2 forms:

### Command line script

To execute YCLEAN from the command line run:

```bash
python -m yclean.run_yclean configfile uvdata
```

Additional command line options can be obtaining by running:

```bash
python -m yclean.run_yclean -h
```

To execute YCLEAN from the command line a configuration file is necessary to obtain parameters for `tclean`, e.g.:

```INI
[yclean]
field = vis_field
imsize = 1000
cell = 0.1arcsec
deconvolver = hogbom
robust = 0.5
spws = 0,1,2,3
chanranges = 0~1500
```

Note that most parameters are optional, if not present they take the default value from `tclean`.


### Modular form

To run YCLEAN, import the `yclean` function into your script:

```python
from yclean_parallel import yclean
```

Read the `yclean_parallel.yclean` function docs to see its input parameters.
