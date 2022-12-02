"""Tasks for creating and writing masks."""
from typing import Optional, TypeVar, Callable
from pathlib import Path

from astropy.io import fits
from casatasks import casalog, importfits
from spectral_cube import SpectralCube
import astropy.units as u
import dask
import dask.array as da
import numpy as np
import scipy.ndimage as ndimg
try:
    import psutil
except ImportError:
    psutil = None
try:
    from dask_image import ndmorph
except ImportError:
    ndmeasure = ndmorph = None

Array = TypeVar('Array', dask.array.core.Array, np.array)

def write_mask(mask: Array, cube: SpectralCube, output: Path) -> None:
    """Write mask to disk.

    Args:
      mask: mask array
      cube: spectral cube.
      output: file name of the CASA mask.
    """
    # Header
    header = cube.header
    del header['BUNIT']

    # FITS file
    hdu = fits.PrimaryHDU(header=header, data=mask.astype('int16'))
    fitsmask = output.with_suffix('.mask.fits')
    hdu.writeto(fitsmask, overwrite=True)

    # CASA file
    importfits(fitsimage=str(fitsmask), imagename=str(output),
               defaultaxes=True, defaultaxesvalues = ['','','','I'])

def remove_small_masks(mask: Array,
                       beam_area: float,
                       beam_fraction: float,
                       dilate: Optional[int] = None,
                       log: Callable = casalog.post) -> Array:
    """Remove small masks pieces.

    Args:
      mask: mask array.
      beam_area: beam area in pixels.
      beam_fraction: fraction of the beam area for small masks.
      dilate: optional; number of iteration to dilate the final mask.
      log: optional; logging function.

    Returns:
      A `dask` array.
    """
    # Some information first
    beams = np.array(beam_area, dtype=int)
    unique_beams = np.unique(beams)
    log(f'Percentage of RAM: {psutil.virtual_memory().percent}')
    log(f'Beam area range: {np.min(beams)} - {np.max(beams)}')
    log(f'Number of unique beams: {len(unique_beams)}')

    # Label mask
    structure = ndimg.generate_binary_structure(mask.ndim, 1)
    labels, nlabels = ndimg.label(mask, structure=structure)
    component_sizes = np.bincount(labels.ravel())
    log(f'Labeled {nlabels} mask structures')
    if psutil is not None:
        log(f'Percentage of RAM: {psutil.virtual_memory().percent}')

    # Iterate over unique beam values
    new_mask = mask.flatten()
    for beam in unique_beams:
        # Search where beam areas are equal
        ind = beams == beam

        # Dilate ind to include one extra channel in the borders
        ind = ndimg.binary_dilation(ind, iterations=1)
        log((f'Removing small mask structures for {np.sum(ind)} channels '
             f'with beam area: {beam} pixels'))

        # Filter small
        small_mask = component_sizes < beam * beam_fraction
        small_mask = small_mask[labels]
        # pylint: disable=E1130
        small_mask[~ind] = False
        log(f'Fitered out {np.sum(small_mask)} pixels in small masks')
        new_mask[small_mask.ravel()] = False
        if psutil is not None:
            log(f'Percentage of RAM: {psutil.virtual_memory().percent}')
    new_mask = new_mask.reshape(mask.shape)

    # Dilate
    if dilate is not None and dilate > 0:
        log(f'Dilating mask {dilate} iteration(s)')
        if psutil is not None:
            log(f'Percentage of RAM: {psutil.virtual_memory().percent}')
        if ndmorph is not None:
            new_mask = ndmorph.binary_dilation(new_mask, structure=structure,
                                               iterations=dilate)
        else:
            new_mask = ndimg.binary_dilation(new_mask, structure=structure,
                                             iterations=dilate)

    return new_mask

def make_threshold_mask(cube: SpectralCube,
                        residual: SpectralCube,
                        pbmap: SpectralCube,
                        mask_threshold: u.Quantity,
                        output_mask: Path,
                        previous_mask: Optional[Array] = None,
                        beam_fraction: float = 1.,
                        dilate: Optional[int] = None,
                        use_residual: bool = True,
                        log: Callable = casalog.post) -> Array:
    """Creates a mask from flux threshold.

    The `residual` image is used if `use_residual=True`, which is the default.
    Otherwise, it takes `cube` image. It calculates a mask with 1s over the
    `mask_threshold`. The task will remove connected components smaller than a
    fraction (could be > 1) of the beam size. Lets call this mask `MM`. If
    existing image(s) name(s) is(are) given in `previous_mask`, the task will
    redefine `MM` by combining it, using a logical `OR`, with that mask.  Mask
    `MM` is recorded with the the name `output_mask`.

    The mask can be dilated in all directions (spatial and spectral). The
    number of iterations to dilate is given by the value of the input parameter
    `dilate`.

    Args:
      cube: image spectral cube.
      residual: residual spectral cube.
      pbmap: primary beam spectral cube.
      mask_threshold: threshold level.
      output_mask: output mask file name.
      previous_mask: optional; mask to combine with the new mask.
      beam_fraction: optional; fraction of the beam to reject small masks.
      dilate: optional; number of dilation iterations.
      use_residual: optional; use residual image?
      log: optional; logging function.

    Returns:
      A binary mask.
      A dictionary with mask statistics.

    Notes:
      Version 28 Dic 2017.
      Updated to python 3 on May 2021.
      Improved on June 2022.
    """
    # Create the threshold mask
    log('Calculating threshold mask')
    if use_residual:
        threshold = mask_threshold.to(cube.unit).value * residual.unit
        mask = (pbmap > 0.2*pbmap.unit) & (residual > threshold)
    else:
        mask = (pbmap > 0.2*pbmap.unit) & (cube > mask_threshold)

    # Operate over dask mask
    stats = {}
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        mask = mask.include()

        # Join with previous mask
        stats['mask_initial'] = np.sum(mask.compute())
        log(f"Inital valid data in mask: {stats['mask_initial']}")
        if previous_mask is not None:
            log('Combining masks')
            log(f'Previous mask valid data: {np.sum(previous_mask.compute())}')
            mask = mask | previous_mask
            stats['mask_combined'] = np.sum(mask.compute())
            log(f"Valid data after combining masks: {stats['mask_combined']}")
        else:
            stats['mask_combined'] = 0

        # Filter out small mask pieces
        log('Removing small masks')
        mask = remove_small_masks(mask, cube.pixels_per_beam, beam_fraction,
                                  dilate=dilate, log=log)
        stats['mask_final'] = np.sum(mask.compute())
        log(f"Final number of valid data: {stats['mask_final']}")
        
        # Write mask
        log('Writing mask')
        write_mask(mask, cube, output_mask)

    return mask, stats

def open_mask(mask_name: Path):
    """Open a mask and load the array."""
    mask = SpectralCube.read(mask_name, use_dask=True, format='casa')
    mask.allow_huge_operations = True
    mask.use_dask_scheduler('threads', num_workers=12)
    mask = mask.unmasked_data[:].value
    mask = da.from_array(mask.astype(bool), chunks='auto')

    return mask
