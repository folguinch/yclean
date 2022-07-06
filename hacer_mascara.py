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
                       beam_area: u.Quantity,
                       beam_fraction: float,
                       dilate: Optional[int] = None,
                       log: Callable = casalog.post) -> Array:
    """Remove small masks pieces.

    Args:
      cube: spectral cube.
      header: image header.
      beamarea: beam area array or float.
      output_mask_name: output mask file name.
      beam_fraction_real: beam fraction.
      log: optional; logging function.

    Returns:
      A `dask` array.
    """
    # Some information first
    beams = np.array(beam_area, dtype=int)
    unique_beams = np.unique(beams)
    log(f'Beam area range: {np.min(beams)} - {np.max(beams)}')
    log(f'Number of unique beams: {len(unique_beams)}')
    mask_array = mask.compute()

    # Label mask
    structure = ndimg.generate_binary_structure(mask_array.ndim, 1)
    labels, nlabels = ndimg.label(mask_array, structure=structure)
    component_sizes = np.bincount(labels.ravel())
    log(f'Labeled {nlabels} mask structures')

    # Iterate over unique beam values
    for beam in unique_beams:
        # Search where beam areas are equal
        ind = beams == beam

        # Dilate ind to include one extra channel in the borders
        ind = ndimg.binary_dilation(ind, iterations=1)
        log((f'Removing small mask structures for {np.sum(ind)} channels '
             f'with beam area: {beam} pixels'))

        # Filter small
        small = component_sizes < beam * beam_fraction
        log(f'Fitered out {np.sum(small)} small mask structures')
        small_mask = small[labels]
        # pylint: disable=E1130
        small_mask[~ind] = False
        mask_array[small_mask] = False

    # Dilate
    if dilate is not None and dilate > 0:
        log(f'Dilating mask {dilate} iteration(s)')
        mask_array = ndimg.binary_dilation(mask_array, structure=structure,
                                           iterations=dilate)

    return da.from_array(mask_array, chunks=mask.chunks)

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

    Notes:
      Version 28 Dic 2017.
      Updated to python 3 on May 2021.
      Improved on June 2022.
    """
    # Create the threshold mask
    log('Calculating threshold mask')
    if use_residual:
        mask = (pbmap > 0.2*pbmap.unit) & (residual > mask_threshold)
    else:
        mask = (pbmap > 0.2*pbmap.unit) & (cube > mask_threshold)

    # Operate over dask mask
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        mask = mask.include()

        # Join with previous mask
        if previous_mask is not None:
            log('Combining masks')
            mask = mask | previous_mask

        # Filter out small mask pieces
        log('Removing small masks')
        mask = remove_small_masks(mask, cube.pixels_per_beam, beam_fraction,
                                  dilate=dilate, log=log)

        # Write mask
        log('Writing mask')
        write_mask(mask, cube, output_mask)

    return mask
