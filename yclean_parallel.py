#>>> =====================================================#
#>>>             YCLEAN Version 2022
#>>>
#>>> Original from: Yanett Contreras
#>>> Adapted to python>=3.6 by Fernando Olguin
#>>> =====================================================#
"""Automasking routine for ALMA cube CLEANing."""
from typing import Optional, Tuple, Callable
from pathlib import Path
import os

from astropy import stats
from casatasks import casalog
from spectral_cube import SpectralCube
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Import local modules
from .hacer_mascara import make_threshold_mask, open_mask
from .utils import (load_images, tclean_parallel, common_beam_cube,
                    second_max_local, pb_crop_fits)

def get_stats(cube: SpectralCube,
              residual: SpectralCube,
              secondary_lobe_level: float,
              residual_max: Optional[u.Quantity] = None,
              planes: Tuple[int] = (2, 8),
              log: Callable = print) -> tuple:
    """Calculate image statistics (rms, max, limits).

    Args:
      cube: image spectral cube.
      residual: residual spectral cube.
      secondary_lobe_level: secondary lobe level.
      residual_max: optional; residual maximum value from previous iterations.
      planes: optional; channel percentiles where statistics are measured.
      log: optional; logging function.

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
    log(f'Image rms: {rms.value:.3e} {rms.unit}')

    # Residual stats and SNR
    #aux = residual.minimal_subcube()
    #ind = np.ones(len(residual.spectral_axis), dtype=bool)
    #ind[:5] = False
    #ind[-5:] = False
    #residual_masked = residual.mask_channels(ind)
    #new_residual_max = np.nanmax(residual_masked.filled_data[:]) * cube.unit
    aux = residual[5:-5]
    new_residual_max = np.nanmax(aux.unmasked_data[:]) * cube.unit
    log(f'Residual maximum: {new_residual_max.value:.3e} {new_residual_max.unit}')
    if residual_max is None or new_residual_max <= residual_max:
        residual_max = new_residual_max
    else:
        return rms, None, None, None

    # For test
    residual_max_pos = np.nanargmax(aux.unmasked_data[:])
    residual_max_pos = np.unravel_index(residual_max_pos, aux.shape)
    log(f'Position of residual maximum: {residual_max_pos}')

    # Limit level
    residual_max = residual_max.to(rms.unit)
    limit_level_snr = residual_max / rms * secondary_lobe_level

    return rms, residual_max, residual_max_pos, limit_level_snr

def get_threshold(limit_level_snr: u.Quantity, residual_max: u.Quantity,
                  rms: u.Quantity, log: Callable = casalog.post) -> str:
    """Calculate the `tclean` threshold.

    We use an arctan function normalized so a secondary lobe level of 0.2 will
    result in a threshold of `0.4*residual_max`.

    Args:
      limit_level_snr: limit level SNR.
      residual_max: maximum of the residual.
      rms: root mean squared.
      log: optional; logging function.
    """
    # Get original value
    limit_level = limit_level_snr * rms
    secondary_lobe_level = limit_level / residual_max
    if not secondary_lobe_level.unit.is_equivalent(u.Unit(1)):
        raise ValueError(('There is a problem with units: '
                          f'{secondary_lobe_level}'))

    # Determine peak scaling factor
    scaling_factor = 0.4 + np.arctan(secondary_lobe_level.value - 0.2)
    log(f'Scaling residual peak by: {scaling_factor}')

    # Calculate threshold
    #rms_mjy = rms.to(u.mJy/u.beam)
    #threshold = f'{2*limit_level_snr*rms_mjy.value}mJy'
    threshold = scaling_factor * residual_max
    threshold = threshold.to(u.mJy/u.beam)

    return f'{threshold.value}mJy'

def yclean(vis: Path,
           imagename: str,
           nproc: int = 5,
           min_limit_level: float = 1.5,
           iter_limit: int = 10,
           common_beam: bool = False,
           resume: bool = False,
           full: bool = False,
           pbcor: bool = False,
           pb_crop_level: Optional[float] = None,
           log: Callable = casalog.post,
           **tclean_args) -> Tuple[Path]:
    """Automatic CLEANing.

    The data is cleaned with an incremental mask for each iteration.
    The maximum number of iterations can be controled with the `iter_limit`
    parameter. The `tclean` threshold is determined from the previous iteration
    (starting with a dirty cube). The `iter_limit` parameter allows to control
    the minimum threshold allowed: `threshold = 2 * min_limit_level * rms`.

    The `pb_crop_level` parameter is used to determine the limits where the
    image cube will be cropped after the final iteration.

    Args:
      vis: visibility filename.
      imagename: imagename base name.
      nproc: optional; number of processes for parallel processing.
      min_limit_level: optional; minimum SNR limit level.
      iter_limit: optional; maximum number of yclean iterations.
      common_beam: optional; calculate common beam cube?
      resume: optional; resume computations.
      full: optional; store intermediate steps images and masks?
      pbcor: optional; compute the pbcor image after the last clean.
      pb_crop_level: optional; crop the final cube down to the given pb limit.
      log: optional; logging function.
      tclean_args: arguments for `tclean`.

    Returns:
      The path of the final image.
      The path of the final image converted to FITS.
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

    # The PSF does not change in further iterations
    secondary_lobe_level = second_max_local(psf)
    log(f'Secondary Lobe PSF Level: {secondary_lobe_level}')

    # RMS calculated in a subset of channels
    rms, residual_max, residual_max_pos, limit_level_snr = get_stats(
        cube,
        residual,
        secondary_lobe_level,
        log=log
    )
    log(f'Dirty rms: {rms}')
    log(f'Dirty residual maxmimum: {residual_max}')
    log(f'Dirty limit level SNR: {limit_level_snr}')

    # Check for resume
    mask_contents = mask_dir.glob(f'{imagename.name}.tc*.mask.fits')
    mask_contents = list(map(str, mask_contents))
    if resume and len(mask_contents) != 0:
        for i in range(0, iter_limit+1):
            aux_mask_name = mask_dir / f'{imagename.name}.tc{i}.mask.fits'
            if str(aux_mask_name) in mask_contents:
                min_it = i
                log(f'Lowest mask iteration number: tc{min_it}')
                break
    else:
        min_it = -1

    # For test
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'Residual max ({residual_max.unit})', color='r')
    ax2.set_ylabel(f'rms ({rms.unit})', color='b')

    # Incremental step
    it = 0
    ax.plot(it, residual_max.value, 'ro')
    ax2.plot(it, rms.value, 'bx')
    cumulative_mask = None
    while limit_level_snr > min_limit_level:
        # Iteration limit
        if it > iter_limit:
            break
        it += 1

        # Resume
        new_mask_name = mask_dir / f'{imagename.name}.tc{it-1}.mask'
        if resume and new_mask_name.is_dir():
            log(f'Iter {it}: skipping iteration')
            continue
        elif resume and it <= min_it:
            log(f'Iter {it}: skipping iteration')
            continue
        elif resume and it > 1 and not new_mask_name.is_dir():
            log(f'Resuming at iteration: {it}')
            cumulative_mask =  mask_dir / f'{imagename.name}.tc{it-2}.mask'
            log(f'Iter {it}: loading previous mask: {cumulative_mask}')
            cumulative_mask = open_mask(cumulative_mask)
        else:
            pass

        # Some logging
        log((f'Iter {it}: SNR of Maximum Residual: '
             f'{limit_level_snr/secondary_lobe_level}'))
        # threshold needs to be (slightly?) above limit_level_snr
        log(f'Iter {it}: SNR of threshold: {limit_level_snr}')

        # This is one idea: the masklevel never gets below SNR=4. When the
        # threshold level is high, masklevel is close limit_level_snr*rms
        masklevel = 1.5 * np.exp(-(limit_level_snr-1.5) / 1.5)
        masklevel = (limit_level_snr + masklevel) * rms
        log(f'Iter {it}: masklevel: {masklevel}')
        log(f'Iter {it}: SNR of masklevel: {masklevel/rms}')

        # Clean threshold
        threshold = get_threshold(limit_level_snr, residual_max, rms, log=log)
        log(f'Iter {it}: tclean threshold: {threshold}')

        # The masks are defined based on the previous image and residuals
        os.system(f'rm -rf {mask_name}')
        cumulative_mask = make_threshold_mask(cube, residual, pbmap,
                                              masklevel, new_mask_name,
                                              beam_fraction=0.5,
                                              use_residual=True,
                                              previous_mask=cumulative_mask)
        log(('Max residual in mask: '
             f'{cumulative_mask[residual_max_pos].compute()}'))

        # Run tclean
        tclean_args.update({'parallel': True,
                            'niter': 1000000,
                            'threshold': threshold,
                            'startmodel': '',
                            'calcpsf': False,
                            'calcres': False,
                            'mask': str(new_mask_name)})
        tclean_parallel(vis, Path(f'{imagename}.tc0'), nproc, tclean_args)

        # Load new images
        if full:
            export_to = Path(f'{imagename}.tc_{it}.image')
        else:
            export_to = None
        cube, residual = load_images(work_img, export_to=export_to, log=log)

        # New stats
        rms, residual_max, residual_max_pos, limit_level_snr = get_stats(
            cube,
            residual,
            secondary_lobe_level,
            residual_max=residual_max,
            planes=(2, 9),
            log=log
        )
        if residual_max is None:
            log('Residual maximum increased, breaking ...')
            break
        ax.plot(it, residual_max.value, 'ro')
        ax2.plot(it, rms.value, 'bx')

        # Delete old masks
        if not full and it > 2:
            old_mask = mask_dir / f'{imagename.name}.tc{it-3}.mask'
            os.system(f'rm -rf {old_mask} {old_mask}.fits')

    # Re-open previous mask if jump here
    it += 1
    if resume and cumulative_mask is None:
        log(f'Resuming at iteration: final')
        cumulative_mask =  mask_dir / f'{imagename.name}.tc{it-2}.mask'
        log(f'Iter final: Loading previous mask: {cumulative_mask}')
        cumulative_mask = open_mask(cumulative_mask)

    # Calculate a final 3*sigma mask
    log('Main iteration cycle done')
    os.system(f'rm -rf {mask_name}')
    new_mask_name = mask_dir / f'{imagename.name}.tc{it-1}.mask'
    masklevel = 3. * rms
    log(f'Final mask level: {masklevel.value:.3e} {masklevel.unit}')
    # The mask is dilated 1 pixel in each direction, the original code dilates
    # only in the spectral direction
    cumulative_mask = make_threshold_mask(cube, residual, pbmap,
                                          masklevel, new_mask_name,
                                          beam_fraction=0.5,
                                          use_residual=True,
                                          previous_mask=cumulative_mask,
                                          dilate=2)

    # Last clean
    try:
        log(f'Last threshold: {threshold}')
    except NameError:
        pass
    rms_mjy = rms.to(u.mJy/u.beam)
    threshold = f'{2.0*rms_mjy.value}mJy'
    log(f'Final threshold: {threshold}')
    tclean_args.update({'parallel': True,
                        'niter': 1000000,
                        'threshold': threshold,
                        'startmodel': '',
                        'calcpsf': False,
                        'calcres': False,
                        'pbcor': pbcor,
                        'mask': str(new_mask_name),
                        'pblimit': 0.1})
    tclean_parallel(vis, Path(f'{imagename}.tc0'), nproc, tclean_args)

    # Export FITS
    export_to = Path(f'{imagename}.tc_final.image')
    load = ('image', 'pb')
    if pbcor:
        load += ('image.pbcor',)
        cube, pb, _ = load_images(work_img, load=load,
                                  export_to=export_to, log=log)
    else:
        cube, pb = load_images(work_img, load=load,
                               export_to=export_to, log=log)
    if common_beam:
        common_beam_cube(cube,
                         export_to.with_suffix('.common_beam.image.fits'),
                         log=log)

    # Final stats
    #residual = load_images(work_img, load=('residual',), log=log)[0]
    #rms, residual_max, *_ = get_stats(
    #    cube,
    #    residual,
    #    secondary_lobe_level,
    #    planes=(2, 9),
    #    log=log
    #)
    #ax.plot(it, residual_max.value, 'ro')
    #ax2.plot(it, rms.value, 'bx')
    #fig.savefig(f'{imagename}.stats.png')

    # Crop image? Shouldn't be used with cubes that will be joined
    if pb_crop_level is not None:
        log(f'Cropping image down to pb level: {pb_crop}')
        load = ('image',)
        if pbcor:
            load += ('image.pbcor',)
        pb_crop_fits(pb, pb_crop_level, export_to, load, log=log)

    # Clean up masks
    os.system(f'rm -rf {mask_dir}/*.mask')

    return work_img, export_to.with_suffix('.image.fits')
