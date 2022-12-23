#>>> =====================================================#
#>>>             YCLEAN Version 2022
#>>>
#>>> Original from: Yanett Contreras
#>>> Adapted to python>=3.7 by Fernando Olguin
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
import numpy.typing as npt

# Import local modules
from .hacer_mascara import make_threshold_mask, open_mask
from .utils import (load_images, tclean_parallel, common_beam_cube,
                    second_max_local, pb_crop_fits, store_stats)

def get_stats(cube: SpectralCube,
              residual: SpectralCube,
              secondary_lobe_level: float,
              pbmask: Optional[SpectralCube] = None,
              planes: Tuple[int] = (2, 8),
              ignore_borders: int = 5,
              log: Callable = print) -> tuple:
    """Calculate image statistics (rms, max, limits).

    Args:
      cube: image spectral cube.
      residual: residual spectral cube.
      secondary_lobe_level: secondary lobe level.
      pbmask: optional; compute maximum only on values in mask.
      planes: optional; channel percentiles where statistics are measured.
      ignore_borders: optional; ignore this number of border channels.
      log: optional; logging function.

    Returns:
      The rms value calculated from the input planes.
      Updated maximum of the residual.
      Limit level SNR.
    """
    # Channel sample
    planes = np.arange(*planes, dtype=float)
    planes = np.floor(planes / 10 * cube.shape[0]).astype(int)
    rms = stats.mad_std(cube.filled_data[planes,:,:], ignore_nan=True)
    log(f'Image rms: {rms.value:.3e} {rms.unit}')

    # Residual stats and SNR
    if pbmask is not None:
        aux = residual.with_mask(pbmask)
        aux = residual[ignore_borders:-ignore_borders]
    else:
        aux = residual[ignore_borders:-ignore_borders]
    residual_max = np.nanmax(aux.filled_data[:]) * cube.unit
    log(f'Residual maximum: {residual_max.value:.3e} {residual_max.unit}')
    #if residual_max is None or new_residual_max <= residual_max:
    #    residual_max = new_residual_max
    #else:
    #    return rms, None, None, None

    # Position of maximum
    residual_max_pos = np.nanargmax(aux.filled_data[:])
    residual_max_pos = np.unravel_index(residual_max_pos, aux.shape)
    residual_max_pos = (residual_max_pos[0] + ignore_borders,) + \
            residual_max_pos[1:]
    log(f'Position of residual maximum: {residual_max_pos}')

    # Limit level
    residual_max = residual_max.to(rms.unit)
    limit_level_snr = residual_max / rms * secondary_lobe_level

    return rms, residual_max, residual_max_pos, limit_level_snr

def plot_yclean_step(cube_spec: u.Quantity,
                     mask_spec: npt.ArrayLike,
                     res_spec: u.Quantity,
                     dirty_spec: u.Quantity,
                     plot_name: Path,
                     threshold: Optional[str] = None,
                     masklevel: Optional[u.Quantity] = masklevel,
                     unit: Optional[u.Unit] = None) -> None:
    """Plot spectra from a `yclean` step.

    Args:
      cube_spec: spectrum from the image.
      mask_spec: mask slice.
      res_spec: spectrum from the residual.
      dirty_spec: spectrum from dirty image.
      plot_name: figure name.
      threshold: optional; plot the threshold value.
      masklevel: optional; plot the mask threshold value.
      unit: optional; intensity unit.
    """
    # Check unit
    if unit is None:
        unit = cube_spec.unit
    res_spec = res_spec.value * cube_spec.unit
    res_spec = res_spec.to(unit)
    cube_spec = cube_spec.to(unit)
    dirty_spec = dirty_spec.to(unit)

    # Normalize mask
    mask_spec_norm = mask_spec.astype(int) * np.nanmax(cube_spec)

    # Plot
    fix, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.set_xlim(0, cube_spec.size-1)
    ax.plot(cube_spec.value, 'k-', ds='steps-mid', label='image', zorder=7)
    ax.plot(mask_spec_norm.value, 'b--', ds='steps-mid', label='mask', zorder=6)
    ax.plot(res_spec.value, 'r-', ds='steps-mid', label='residual', zorder=5)
    ax.plot(dirty_spec.value, 'g:', ds='steps-mid', label='dirty', zorder=4)

    # Threshold comes in CASA notation
    if threshold is not None:
        thresh = u.Quantity(threshold) / u.beam
        thresh = thresh.to(unit)
        ax.axhline(thresh.value, ls='-', c='c', label='threshold', zorder=3)
    
    # Mask level comes as quantity
    if masklevel is not None:
        ax.axhline(masklevel.to(unit).value, ls='-', c='m', label='masklevel',
                   zorder=3)

    # Save figure
    ax.legend(loc='best')
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close()

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
    # original form: threshold = f'{2*limit_level_snr*rms_mjy.value}mJy'
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
           peak_tol: float = 0.01,
           full: bool = False,
           pbcor: bool = False,
           pb_crop_level: Optional[float] = None,
           spectrum_at: Optional[Tuple[int]] = None,
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

    The `peak_tol` determines if the peak should be scaled down. This may
    be triggered when the residual peak is outside the mask and it did not
    change in value (within the tolerance) after the tclean iteration.
    The tolerance is defined as `peak_tol * rms`.

    Args:
      vis: visibility filename.
      imagename: imagename base name.
      nproc: optional; number of processes for parallel processing.
      min_limit_level: optional; minimum SNR limit level.
      iter_limit: optional; maximum number of yclean iterations.
      common_beam: optional; calculate common beam cube?
      resume: optional; resume computations.
      peak_tol: optional; rms factor to trigger a peak correction.
      full: optional; store intermediate steps images and masks?
      pbcor: optional; compute the pbcor image after the last clean.
      pb_crop_level: optional; crop the final cube down to the given pb limit.
      spectrum_at: optional; plot results using a spectrum at this pixel.
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
        pbmask=pbmap>0.2*pbmap.unit,
        log=log
    )
    log(f'Dirty rms: {rms}')
    log(f'Dirty residual maxmimum: {residual_max}')
    log(f'Dirty limit level SNR: {limit_level_snr}')

    # Spectrum
    if spectrum_at is not None:
        dirty_spec = cube.unmasked_data[:, spectrum_at[1], spectrum_at[0]]
    else:
        dirty_spec = None

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

    # Incremental step
    it = 0
    stats_file = imagename.parent / 'statistics.dat'
    store_stats(stats_file,
                {'it': it,
                 'rms': rms,
                 'residual_max': residual_max,
                 'residual_max_pos': residual_max_pos,
                 'threshold': 'none',
                 'mask_level': 0 * rms.unit,
                 'mask_initial': 0,
                 'mask_combined': 0,
                 'mask_final': 0})
    cumulative_mask = None
    old_residual_max = residual_max
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
            resume = False

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
        cumulative_mask, mask_stats = make_threshold_mask(
            cube,
            residual,
            pbmap,
            masklevel,
            new_mask_name,
            beam_fraction=0.5,
            use_residual=True,
            previous_mask=cumulative_mask,
        )
        log(('Max residual in mask: '
             f'{cumulative_mask.is_in(tuple(residual_max_pos))}'))

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
            #residual_max=residual_max,
            pbmask=pbmap>0.2*pbmap.unit,
            planes=(2, 9),
            log=log
        )
        sti = {'it': it, 'rms': rms, 'residual_max': residual_max,
               'residual_max_pos': residual_max_pos, 'threshold': threshold,
               'mask_level': masklevel}
        sti.update(mask_stats)
        store_stats(stats_file, sti)
        if residual_max <= old_residual_max :
            tol = peak_tol * rms
            # We know residual_max is already <= old_residual_max
            if residual_max >= old_residual_max - tol:
                residual_max = 0.8 * residual_max
                if residual_max <= 2.0 * rms:
                    log('Residual max cannot be corrected, breaking ...')
                    break
                log(f'Corrected residual max: {residual_max}')
            else:
                old_residual_max = residual_max
        else:
            log('Residual maximum increased, breaking ...')
            break

        # Plot spectrum
        if spectrum_at is not None:
            cube_spec = cube.unmasked_data[:, spectrum_at[1], spectrum_at[0]]
            res_spec = residual.unmasked_data[:, spectrum_at[1], spectrum_at[0]]
            mask_spec = cumulative_mask.mask_from_position(spectrum_at)
            plot_name = f'.spec{spectrum_at[0]}_{spectrum_at[1]}.png'
            plot_name = new_mask_name.with_suffix(plot_name)
            plot_yclean_step(cube_spec, mask_spec, res_spec, dirty_spec,
                             plot_name, threshold=threshold,
                             masklevel=masklevel, unit=u.mJy/u.beam)

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
    cumulative_mask, mask_stats = make_threshold_mask(
        cube,
        residual,
        pbmap,
        masklevel,
        new_mask_name,
        beam_fraction=0.5,
        use_residual=True,
        previous_mask=cumulative_mask,
        dilate=2,
    )

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
    residual = load_images(work_img, load=('residual',), log=log)[0]
    rms, residual_max, residual_max_pos, *_ = get_stats(
        cube,
        residual,
        secondary_lobe_level,
        pbmask=pbmap>0.2*pbmap.unit,
        planes=(2, 9),
        log=log
    )
    sti = {'it': it, 'rms': rms, 'residual_max': residual_max,
           'residual_max_pos': residual_max_pos, 'threshold': threshold,
           'mask_level': masklevel}
    sti.update(mask_stats)
    store_stats(stats_file, sti)

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
