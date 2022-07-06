#>>> =====================================================#
#>>>             YCLEAN Version 2020
#>>>
#>>> Original from: Yanett Contreras
#>>> Adapted to python>=3.6 by Fernando Olguin
#>>> =====================================================#
"""Automasking routine for ALMA cube CLEANing."""
# CASA 6.0+
from typing import Optional, Tuple, Callable
from pathlib import Path
import os

from astropy import stats
from casatasks import casalog
from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np

# Import local modules
from .hacer_mascara import make_threshold_mask
from .utils import (load_images, tclean_parallel, common_beam_cube,
                    second_max_local)

def get_stats(cube: SpectralCube,
              residual: SpectralCube,
              secondary_lobe_level: float,
              residual_max: Optional[u.Quantity] = None,
              planes: Tuple[int] = (2, 8)) -> tuple:
    """Calculate image statistics (rms, max, limits).

    Args:
      cube: image spectral cube.
      residual: residual spectral cube.
      secondary_lobe_level: secondary lobe level.
      residual_max: optional; residual maximum value from previous iterations.
      planes: optional; channel percentiles where statistics are measured.

    Returns:
      The rms value calculated from the input planes.
      Updated maximum of the residual.
      Limit level SNR.
    """
    # Channel sample
    planes = np.arange(*planes, dtype=float)
    planes = np.floor(planes / 10 * cube.shape[0]).astype(int)
    rms = 1.42602219 * \
            stats.median_absolute_deviation(cube.unmasked_data[planes,:,:],
                                            ignore_nan=True)

    # Residual stats and SNR
    new_residual_max = residual.max()
    if residual_max is None or new_residual_max <= residual_max:
        residual_max = new_residual_max
    else:
        return rms, None, None

    # Limit level
    limit_level_snr = residual_max / rms * secondary_lobe_level

    return rms, residual_max, limit_level_snr

def yclean(vis: Path,
           imagename: str,
           nproc: int = 5,
           min_limit_level: float = 1.5,
           iter_limit: int = 10,
           log: Callable = casalog.post,
           **tclean_args) -> None:
    """Automatic CLEANing.

    The data is cleaned with an incremental mask for each iteration.
    The maximum number of iterations can be controled with the `iter_limit`
    parameter. The `tclean` threshold is determined from the previous iteration
    (starting with a dirty cube). The `iter_limit` parameter allows to control
    the minimum threshold allowed: `threshold = 2 * min_limit_level * rms`.

    Args:
      vis: visibility filename.
      imagename: imagename base name.
      nproc: optional; number of processes for parallel processing.
      min_limit_level: optional; minimum SNR limit level.
      iter_limit: optional; maximum number of yclean iterations.
      log: optional; logging function.
      tclean_args: arguments for `tclean`.
    """
    # Some useful definitions
    work_img = Path(f'{imagename}.tc0.image')
    mask_name = work_img.with_suffix('.mask')
    mask_dir = work_img.parent / 'masks'
    mask_dir.mkdir(exist_ok=True)

    # Dirty
    tclean_args.update({'parallel': True,
                        'niter': 0})
    if not work_img.is_dir():
        log('Calculating dirty image')
        work_img.parent.mkdir(exist_ok=True)
        tclean_parallel(vis, Path(f'{imagename}.tc0'), nproc, tclean_args)
        #tclean(vis=vis, imagename=f'{imagename}.tc0', **tclean_args)
    else:
        log('Dirty found ... skipping initial tclean')
    # pylint: disable=W0632
    psf, pbmap, cube, residual = load_images(work_img,
                                             load=('psf', 'pb', 'image',
                                                   'residual'),
                                             log=log)
    residual = residual * cube.unit

    # The PSF does not change in further iterations
    secondary_lobe_level = second_max_local(psf)
    log(f'Secondary Lobe PSF Level: {secondary_lobe_level}')

    # RMS calculated in a subset of channels
    rms, residual_max, limit_level_snr = get_stats(cube, residual,
                                                   secondary_lobe_level)
    log(f'Dirty rms: {rms}')
    log(f'Residual maxmimum: {residual_max}')
    log(f'Limit level SNR: {limit_level_snr}')

    # Incremental step
    it = 0
    cumulative_mask = None
    while limit_level_snr > min_limit_level:
        # Iteration limit
        if it > iter_limit:
            break
        it += 1

        # Some logging
        log((f'Iter {it}: SNR of Maximum Residual: '
             f'{limit_level_snr/secondary_lobe_level}'))
        # threshold needs to be (slightly?) above limit_level_snr
        log(f'Iter {it}: SNR of threshold: {limit_level_snr}')

        # This is one idea: the masklevel never gets below SNR=4. When the
        # threshold level is high, masklevel is close limit_level_snr*rms
        masklevel = 1.5 * np.exp(-(limit_level_snr-1.5) / 1.5)
        masklevel = (limit_level_snr + masklevel) * rms
        log(f'Iter {it}: SNR of masklevel: {masklevel/rms}')

        # Clean threshold
        rms = rms.to(u.mJy/u.beam)
        threshold = f'{2*limit_level_snr*rms.value}mJy'
        log(f'Iter {it}: Threshold: {threshold}')

        # The masks are defined based on the previous image and residuals
        os.system(f'rm -rf {mask_name}')
        new_mask_name = mask_dir / f'{imagename.name}.tc{it-1}.mask'
        cumulative_mask = make_threshold_mask(cube, residual, pbmap,
                                              masklevel, new_mask_name,
                                              beam_fraction=0.5,
                                              use_residual=True,
                                              previous_mask=cumulative_mask)

        # Run tclean
        tclean_args.update({'parallel': True,
                            'niter': 100000,
                            'threshold': threshold,
                            'startmodel': '',
                            'mask': str(new_mask_name)})
        tclean_parallel(vis, Path(f'{imagename}.tc0'), nproc, tclean_args)

        # Load new images
        export_to = Path(f'{imagename}.tc_{it}.image')
        cube, residual = load_images(work_img, export_to=export_to, log=log)
        residual = residual * cube.unit

        # New stats
        rms, residual_max, limit_level_snr = get_stats(
            cube,
            residual,
            secondary_lobe_level,
            residual_max=residual_max,
            planes=(2, 9),
        )
        if residual_max is None:
            log('Residual maximum increased, breaking ...')
            break

    # Calculate a final 3*sigma mask
    log('Main iteration cycle done')
    it += 1
    os.system(f'rm -rf {mask_name}')
    new_mask_name = mask_dir / f'{imagename.name}.tc{it-1}.mask'
    masklevel = 3. * rms
    # The mask is dilated 1 pixel in each direction, the original code dilates
    # only in the spetral direction
    cumulative_mask = make_threshold_mask(cube, residual, pbmap,
                                          masklevel, new_mask_name,
                                          beam_fraction=0.5,
                                          use_residual=True,
                                          previous_mask=cumulative_mask,
                                          dilate=1)

    # Last clean
    try:
        log(f'Last threshold: {threshold}')
    except NameError:
        pass
    threshold = f'{2.0*rms*1e3}mJy'
    log(f'Final threshold: {threshold}')
    tclean_args.update({'parallel': True,
                        'niter': 100000,
                        'threshold': threshold,
                        'startmodel': '',
                        'mask': str(new_mask_name),
                        'pblimit': 0.1})
    tclean_parallel(vis, Path(f'{imagename}.tc0'), nproc, tclean_args)

    # Export FITS
    export_to = Path(f'{imagename}.tc_final.image')
    cube, _ = load_images(work_img, load=('image', 'pb'), export_to=export_to,
                           log=log)
    common_beam_cube(cube, export_to.with_suffix('.common_beam.image.fits'),
                     log=log)
