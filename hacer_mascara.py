"""Tasks for creating and writing masks."""
from typing import Optional, TypeVar, Callable, Tuple
from pathlib import Path

from astropy.io import fits
from casatasks import casalog, importfits
from spectral_cube import SpectralCube
import astropy.units as u
import dask
import dask.array as da
import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndimg
try:
    import psutil
except ImportError:
    psutil = None

class IndexedMask:
    """Store a mask by tracking `True` indices.

    Attributes:
      indices: structured array containing the indices along each axis.
      shape: shape of the mask.
      origin: indices origin.
      is_shifted: origin shifting status.
    """

    def __init__(self, indices: npt.ArrayLike, shape: Tuple[int],
                 origin: Tuple[int], is_shifted: bool = False) -> None:
        self.indices = indices
        self.shape = shape
        self.origin = origin
        self.is_shifted = is_shifted

    @classmethod
    def from_array(cls, mask: npt.ArrayLike):
        """Create an indexed mask from an array."""
        # Get indices
        indices = cls.indices_from_array(mask)

        # Find origin and shift
        origin = cls.get_origin(indices)
        #origin = tuple(np.min(indices[field])
        #               for field in indices.dtype.fields)
        #origins = tuple()
        #for field in indices.dtype.fields:
        #    origin = np.min(indices[field])
        #    indices[field] = indices[field] - origin
        #    origins += (origin,)

        return cls(indices, mask.shape, origin)

    @property
    def size(self) -> int:
        return len(self.indices)

    @staticmethod
    def indices_from_array(mask: npt.ArrayLike,
                           dtype='int32') -> npt.ArrayLike:
        """Get mask indices from mask array."""
        indices = np.nonzero(mask)
        indices = np.array(list(zip(*indices)),
                           dtype=[('chan', dtype),
                                  ('y', dtype),
                                  ('x', dtype)])
        if (np.any(indices['chan'] < 0) or np.any(indices['x'] < 0) or
            np.any(indices['y'] < 0)):
            raise ValueError(f'Wrong dtype: {dtype}')

        return indices

    @staticmethod
    def get_origin(indices: npt.ArrayLike) -> Tuple[int]:
        """Get the origin of structured array of indices.

        The origin is defined as the lowest index along each axis (field) in
        the array.
        """
        return tuple(np.min(indices[field]) for field in indices.dtype.fields)

    def get_max(self, shift: int = 0) -> Tuple[int]:
        """Get the maximum along each field."""
        return tuple(np.max(self.indices[field]) + shift
                     for field in self.indices.dtype.fields)

    def update_origin(self):
        """Find and update the `origin` from the current stored indices."""
        if self.is_shifted:
            raise ValueError('Cannot update origin of already shifted indices')
        self.origin = IndexedMask.get_origin(self.indices)

    def shift_to_origin(self):
        """Shift indices so origin is zero."""
        if self.is_shifted:
            raise ValueError('Indices already shifted')

        # Update origin just in case someone forgot to do it
        self.update_origin()
        for i, field in enumerate(self.indices.dtype.fields):
            self.indices[field] = self.indices[field] - self.origin[i]

        # Update status
        self.is_shifted = True

    def shift_back(self):
        """Shift indices so origin is the origin back to original mask."""
        # Check shift status
        if not is_shifted:
            raise ValueError('Cannot shift back unshifted indices')

        # Shift indices
        for i, field in enumerate(self.indices.dtype.fields):
            self.indices[field] = self.indices[field] + self.origin[i]

        # Update status
        self.is_shifted = False

    def update_to(self, mask: npt.ArrayLike,
                  shift_back: bool = False) -> None:
        """Replace the indices using input mask.
        
        If `shift_back` is `True` then the indices derived from `mask` are
        shifted using the stored `origin`, i.e. it assumes the input mask has
        the same origin as the stored indices and that these were shifted so
        origin is zero. 
        """
        # Replace indices
        self.indices = IndexedMask.indices_from_array(mask)

        # Shift indices
        if shift_back:
            self.shift_back()

        # Update origin if needed
        if not self.is_shifted:
            self.update_origin()

    def merge_with(self, mask: 'IndexedMask') -> None:
        """Union of 2 masks."""
        # Check shift status
        if self.is_shifted:
            raise ValueError('Cannot merge shifted indices')
        elif mask.is_shifted:
            raise ValueError('Cannot merge with shifted input mask')
        if self.shape != mask.shape:
            raise ValueError(('Masks with different shape: '
                              f'{self.shape} {mask.shape}'))

        # Update indices
        indices = np.concatenate((self.indices, mask.indices))
        self.indices = np.array(list(set(map(tuple, indices))),
                                dtype=self.indices.dtype)

        # Update origin
        self.update_origin()

    def minimal_mask(self, shift_to_origin: bool = False) -> npt.ArrayLike:
        """Return a mask with the minimum shape containg all valid point."""
        if shift_to_origin:
            self.shift_to_origin()

        # Get the shape
        shape = self.get_max(shift=1)

        # Generate array
        mask = np.zeros(shape, dtype=bool)
        mask[tuple(np.array(list(map(list, self.indices))).T)] = True

        return mask

    def to_array(self) -> npt.ArrayLike:
        """Build a mask array."""
        # Check shift status
        if self.is_shifted:
            raise ValueError('Cannot create array from shifted indices')

        # Create array
        mask = np.zeros(self.shape, dtype=bool)
        mask[tuple(np.array(list(map(list, self.indices))).T)] = True

        return mask

    def is_in(self, val: Tuple[int]) -> bool:
        """Is position in mask?"""
        return np.isin(np.array([val], dtype=self.indices.dtype)[0],
                       self.indices)

    def mask_from_position(self, pix: Tuple[int]) -> npt.ArrayLike:
        """Recover the mask at a given position."""
        ind = (self.indices['x'] == pix[0]) & (self.indices['y'] == pix[1])
        chans = self.indices['chan'][ind]
        mask = np.zeros(len(ind), dtype=bool)
        mask[chans] = True

        return mask

def write_mask(mask: npt.ArrayLike, cube: SpectralCube,
               output: Path) -> None:
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

def remove_small_masks(mask: IndexedMask,
                       beam_area: float,
                       beam_fraction: float,
                       log: Callable = casalog.post) -> IndexedMask:
    """Remove small masks pieces.

    Args:
      mask: mask array.
      beam_area: beam area in pixels.
      beam_fraction: fraction of the beam area for small masks.
      log: optional; logging function.

    Returns:
      An `IndexedMask` object.
    """
    # Some information first
    working_mask = mask.minimal_mask(shift_to_origin=True)
    valid_beam_area = beam_area[mask.origin[0]:][:working_mask.shape[0]]
    beams = np.array(valid_beam_area, dtype=int)
    unique_beams = np.unique(beams)
    if psutil is not None:
        log(f'Percentage of RAM: {psutil.virtual_memory().percent}')
    log(f'Beam area range: {np.min(beams)} - {np.max(beams)}')
    log(f'Number of unique beams: {len(unique_beams)}')

    # Label mask
    log(f'Working mask shape: {working_mask.shape}')
    structure = ndimg.generate_binary_structure(working_mask.ndim, 1)
    labels, nlabels = ndimg.label(working_mask, structure=structure)
    component_sizes = np.bincount(labels.ravel())
    log(f'Labeled {nlabels} mask structures')
    if psutil is not None:
        log(f'Percentage of RAM: {psutil.virtual_memory().percent}')

    # Iterate over unique beam values
    #new_mask = mask.flatten()
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
        #new_mask[small_mask.ravel()] = False
        working_mask[small_mask] = False
        if psutil is not None:
            log(f'Percentage of RAM: {psutil.virtual_memory().percent}')
    #new_mask = new_mask.reshape(mask.shape)

    # Store in the mask and restore indices
    mask.update_to(working_mask, shift_back=True)

    return mask

def make_threshold_mask(cube: SpectralCube,
                        residual: SpectralCube,
                        pbmap: SpectralCube,
                        mask_threshold: u.Quantity,
                        output_mask: Path,
                        previous_mask: Optional[IndexedMask] = None,
                        beam_fraction: float = 1.,
                        dilate: Optional[int] = None,
                        use_residual: bool = True,
                        log: Callable = casalog.post) -> IndexedMask:
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

    # Compute mask
    stats = {}
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        mask = mask.include().compute()

    # Join with previous mask
    mask = IndexedMask.from_array(mask)
    stats['mask_initial'] = mask.size
    log(f"Initial valid data in mask: {stats['mask_initial']}")
    log(f'Initial mask origin: {mask.origin}')
    if previous_mask is not None:
        log('Combining masks')
        log(f'Previous mask valid data: {previous_mask.size}')
        mask.merge_with(previous_mask)
        stats['mask_combined'] = mask.size
        log(f"Valid data after combining masks: {stats['mask_combined']}")
        log(f'Combined mask origin: {mask.origin}')
    else:
        stats['mask_combined'] = mask.size

    # Filter out small mask pieces
    log('Removing small masks')
    mask = remove_small_masks(mask, cube.pixels_per_beam, beam_fraction,
                              log=log)

    # Build mask array
    mask_array = mask.to_array()

    # Dilate
    if dilate is not None and dilate > 0:
        log(f'Dilating mask {dilate} iteration(s)')
        structure = ndimg.generate_binary_structure(mask_array.ndim, 1)
        if psutil is not None:
            log(f'Percentage of RAM: {psutil.virtual_memory().percent}')
        mask_array = ndimg.binary_dilation(mask_array,
                                           structure=structure,
                                           iterations=dilate)
        mask.update_to(mask_array)

    # Final stats
    stats['mask_final'] = mask.size
    log(f"Final number of valid data: {stats['mask_final']}")
    log(f'Final mask origin: {mask.origin}')
    
    # Write mask
    log('Writing mask')
    write_mask(mask_array, cube, output_mask)

    return mask, stats

def open_mask(mask_name: Path):
    """Open a mask and load the array."""
    mask = SpectralCube.read(mask_name, use_dask=True, format='casa')
    #mask.allow_huge_operations = True
    #mask.use_dask_scheduler('threads', num_workers=12)
    mask = mask.unmasked_data[:].value.astype(bool)
    #mask = da.from_array(mask.astype(bool), chunks='auto')
    mask = IndexedMask.from_array(mask)

    return mask
