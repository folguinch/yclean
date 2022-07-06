"""Common utilities."""
from datetime import datetime
from typing import Callable, Sequence, Tuple, Optional
from pathlib import Path
import itertools
import json
import os
import subprocess

from casatasks import exportfits, tclean, casalog
from dask.diagnostics import ProgressBar
from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np

def load_images(imagename: Path,
                load: Sequence[str] = ('image', 'residual'),
                export_to: Optional[Path] = None,
                log: Callable = casalog.post) -> Tuple[SpectralCube]:
    """Load images into `SpectralCube` objects.

    Args:
      imagename: name of the image (ending with `.image`).
      load: optional; type of files to load.
      export_to: optional; same as `imagename` but to write FITS to.
      log: optional; logging function.

    Returns:
      A tuple with the images in the same order as `load`.
    """
    pbar = ProgressBar()
    pbar.register()
    images = []
    for imtype in load:
        name = imagename.with_suffix(f'.{imtype}')
        if export_to is not None:
            log(f'Exporting {imtype} to FITS')
            fitsname = export_to.with_suffix(f'.{imtype}.fits')
            exportfits(imagename=str(name), fitsimage=str(fitsname))
        image = SpectralCube.read(name, use_dask=True, format='casa')
        image.allow_huge_operations = True
        images.append(image)
        log(f'Loaded {imtype} with shape {image.shape}')

    return tuple(images)

def second_max_local(psf: 'SpectralCube'):
    """Determine the psf secondary lobe level."""
    # Central plane
    midpsf = psf.unmasked_data[np.floor(psf.shape[0]/2).astype(int),:,:]

    # Find second
    aux = np.ones(midpsf.shape)
    rng = range(-1, 2)
    for dx, dy in itertools.product(rng, rng):
        if dx**2 + dy**2 > 0:
            aux *= midpsf > np.roll(np.roll(midpsf, dx, axis=0), dy, axis=1)
    midpsf = np.sort(np.ndarray.flatten(midpsf * aux))
    second = midpsf[-2] / midpsf[-1]

    return second

def tclean_parallel(vis: Path,
                    imagename: Path,
                    nproc: int,
                    tclean_args: dict,
                    log: Callable = casalog.post):
    """Run `tclean` in parallel.

    If the number of processes (`nproc`) is 1, then it is run in a single
    processor. The environmental variable `MPICASA` is used to run the code,
    otherwise it will use the `mpicasa` and `casa` available in the system.

    A new logging file is created by `mpicasa`. This is located in the same
    directory where the program is executed.

    Args:
      vis: measurement set.
      imagename: image file name.
      nproc: number of processes.
      tclean_args: other arguments for tclean.
      log: optional; logging function.
    """
    if nproc == 1:
        tclean_args.update({'parallel': False})
        tclean(vis=str(vis), imagename=str(imagename), **tclean_args)
    else:
        # Save tclean params
        tclean_args.update({'parallel': True})
        paramsfile = imagename.parent / 'tclean_params.json'
        paramsfile.write_text(json.dumps(tclean_args, indent=4))

        # Run
        cmd = os.environ.get('MPICASA', f'mpicasa -n {nproc} casa')
        script = Path(__file__).parent / 'run_tclean_parallel.py'
        logfile = datetime.now().isoformat(timespec='milliseconds')
        logfile = f'tclean_parallel_{logfile}.log'
        cmd = (f'{cmd} --nogui --logfile {logfile} '
               f'-c {script} {vis} {imagename} {paramsfile}')
        log(f'Running: {cmd}')
        # pylint: disable=R1732
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        proc.wait()

def common_beam_cube(cube: SpectralCube, filename: Path,
                     log: Callable = casalog.post) -> None:
    """Convolve the cube to a single beam data cube.

    Args:
      cube: the spectral cube.
      filename: output FITS file.
      log: optional; logging function.
    """
    # Common beam
    common_beam = cube.beams.common_beam()
    minbeam, maxbeam =  cube.beams.extrema_beams()
    common_asec = [common_beam.minor.to(u.arcsec),
                   common_beam.major.to(u.arcsec)]
    log(f'Smallest beam: {minbeam.minor} {minbeam.major}')
    log(f'Largest beam: {maxbeam.minor} {maxbeam.major}')
    log(f'Common beam: {common_asec[0]} {common_asec[1]}')

    # Convolve
    new_cube = cube.convolve_to(common_beam)
    new_cube.write(filename)

