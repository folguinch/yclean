"""Tasks for creating a mask."""
from collections import Counter
from typing import Sequence
from pathlib import Path
import os
import sys
import time

from casatasks import casalog, imhead
from spectral_cube import SpectralCube
from spectral_cube.io.casa_masks import make_casa_mask
import numpy as np
import scipy.ndimage as ndimg

def get_nchans(header: dict) -> int:
    """Return the number of channels."""
    freq_axis = np.where(header['axisnames']=='Frequency')[0][0]
    return header['shape'][freq_axis]

def copy_pbimg(image_name: str, copy_name: Path = Path('pbimage.im')) -> Path:
    """Copy the pb image.

    Args:
      image_name: image name.
      copy_name: optional; destination.

    Returns:
      The path of the copy.
    """
    pbimg = Path(f'{image_name}.pb')
    casalog.post(f'Copying {pbimg} to {copy_name}')
    os.system(f'rm -rf {copy_name}')
    os.system(f'cp -r {pbimg} {copy_name}')

    return copy_name

def get_beam_area(header: dict) -> Union[np.array, float]:
    """Calculate the beam area."""
    multi_beams = 'perplanebeams' in header
    nchans = get_nchans(header)

    # The following checks whether there are multiple beams or not. Defines
    # an array of major and minor beamlengths
    if multi_beams:
        major = []
        minor = []
        beams = header['perplanebeams']['beams']
        for ch in range(0, nchans):
            # in arcsec by default, apparently
            major.append(beams[f'*{ch}']['*0']['major']['value'])
            minor.append(beams[f'*{ch}']['*0']['minor']['value'])
        major = np.array(major)
        minor = np.array(minor)
    else:
        # in arcsec by default, apparently
        major = header['beammajor']['value']
        minor = header['beamminor']['value']

    unit_cdelt = header['cdelt2']['unit']
    if unit_cdelt == 'rad':
        # in the header, these CDELT values are in radians
        pixelsize = header['cdelt2']['value'] / pi*180*3600
    else:
        raise NotImplementedError(f'Pixelsize with unit: {unit_cdelt}')
    # beamarea in pixels' area
    beamarea = (major * minor * pi / (4 * log(2))) / (pixelsize**2)

    return beamarea

def generate_masked_cube(image_name:str, mask_threshold: float,
                         use_residual: bool) -> SpectralCube:
    """Create a masked cube with a threshold mask.

    Args:
      image_name: image name.
      mask_threshold: flux threshold.
      use_residual: use residual or image data.
    """
    # Copy the pb image
    pbimg = copy_pbimg(image_name)

    # El mero mero del asunto. It creates the mask 'output_mask_name' (this
    # string is the same as mm) with the mask_threshold. The gridding of the
    # mask is equivalent to that of the image.
    if use_residual:
        cubename = Path(f'{image_name}.residual')
        #immath(imagename = [imageName+'.residual'],
        #outfile = mm,
        #expr = 'iif(IM0 > '+str(mask_threshold) +',1.0,0.0)',
        #mask=myflux+'>0.2')
    else:
        cubename = Path(f'{image_name}.image')
        #immath(imagename = [myimage],
        #outfile = mm,
        #expr = 'iif(IM0 > '+str(mask_threshold) +',1.0,0.0)',
        #mask=myflux+'>0.2')
    pbmap = SpectralCube.read(pbimg, format='casa_image')
    cube = SpectralCube.read(cubename, format='casa_image')
    mask = (pbmap > 0.2) & (cube > mask_threshold)
    cube = cube.with_mask(mask)
    #make_casa_mask(cube, output_mask_name)

    return cube

def remove_small_masks(cube: SpectralCube, header: dict,
                       beamarea: Union[np.array, float],
                       output_mask_name: Path,
                       beam_fraction_real: float) -> SpectralCube:
    """Remove small masks pieces.

    Args:
      cube: spectral cube.
      header: image header.
      beamarea: beam area array or float.
      output_mask_name: output mask file name.
      beam_fraction_real: beam fraction.
    """
    # Some useful definitions
    mask = cube.mask
    nchans = get_nchans(header)

    #ia.open(mm)    # Open the mask
    #mask=ia.getchunk() # Get the data in an array, usually of dimension 4
                       # (spatial x spatial x pol x frequency/velocity
                       # or (spatial x spatial x frequency/velocity x pol)
    if nchans > 1: # In case of multiple beams
        # Mask size limit
        sizelimit = beamarea * beam_fraction_real

        # remove extra redundant dimension (polarization?)
        mask = np.squeeze(mask)
        casalog.post(f'Mask shape: {mask.shape}')

        # The neighboring structure to give to a cube. Otherwise it will assume
        # that neighboring channels are not independent.
        #logfile = open('logfilemasking.txt','w')
        neighbor_structure = [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]

        # Separate and label connected components
        labeled, j = ndimg.label(mask, structure=neighbor_structure)
        label_stack = set(range(1, j+1))
        casalog.post(f'Labeled mask: {j} pieces')
        for canal in range(nchans):
            #if (canal%20==0):
            #casalog.post("Channel # " + str(canal))
            counts = Counter(np.ndarray.flatten(labeled[:,:,canal]).tolist())
            labels_in_channel = counts.keys()
            del labels_in_channel[0]
            #print set(labelsInChannel)
            #casalog.post('Possible issue: %r' %
            #        (set(labelsInChannel)<=label_stack,))
            if set(labels_in_channel) <= label_stack:
                label_stack = label_stack - set(labels_in_channel)
            else:
                raise Exception('Error in labeling')
    
            # Remove small connected components
            sumapixels = np.bincount(labeled[:,:,canal].flatten())
            sumapixels = sumapixels[labeled[:,:,canal]]
            aux = (sumapixels < sizelimit[canal]) & (labeled[:,:,canal] != 0)
            mask[:,:,canal][aux] = 0
        
        # The following gets the mask to the original dimensions of the image
        # It is assumed that 'Stokes' axis has no dimension
        pol_axis_position = np.where(header['axisnames'] == 'Stokes')[0][0]
        axes_order = [1, 2, 3]
        axes_order.insert(pol_axis_position, 0)
        mask = np.transpose(mask[None,:], tuple(axes_order))
                
    else: # In case of single beam
        neighbor_structure = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        labeled, j = ndimg.label(mask, structure=neighbor_structure)
        casalog.post('Labeled mask')
        for i in range(1, j):
            sumapixels = np.sum((labeled == j).astype(int), axis=(0, 1))
            if sumapixels < beamarea * beam_fraction_real:
                mask[labeled==i] = 0
           
    cube = cube.with_mask(mask)
    make_casa_mask(cube, output_mask_name)
    #ia.putchunk(mask)
    #ia.done()
    #ia.close()
    #logfile.close()

    return cube

def combine_masks(output_mask_name: Path, combine_mask: Sequence[Path],
                  imagename: Path, multi_beams: bool):
    """Combine input masks.
    
    Args:
      output_mask_name: output mask.
      combine_mask: masks to combine.
      imagename: image name.
      multi_beams: has the input image multiple beams?
    """
    ##  Combine the main mask with the rest 
    inpmask = [output_mask_name]
    #inpmask.append(imageName+'.mask') if(os.path.exists(imageName+'.mask')) else 1
    #inpmask.append(imageName+'.pb') if(os.path.exists(imageName+'.pb')) else 1
    for cmask in combine_mask:
        if cmask.is_dir():
            inpmask.append(cmask)
        else:
            casalog.post(f'Mask does not exists: {cmask}')
    #if isinstance(combine_mask, basestring):
    #    inpmask.append(combine_mask) if(os.path.exists(combine_mask)) else 1   
    #elif all(isinstance(item, basestring) for item in combine_mask):
    #    for cM in combine_mask:
    #        inpmask.append(cM) if(os.path.exists(cM)) else 1 
    #else:
    #    raise Exception("Error in combine_mask keyword")
    casalog.post(f'Masks to combine: {inpmask}')
    
    # If multiBeams, delete all beams to run makemask
    if multi_beams:
        # Masks restoring beams
        rb = {}
        for mascara in inpmask:
            #print mascara
            ia.open(mascara)
            rbaux = ia.restoringbeam()
            if rbaux:# check it has beam information
                rb[mascara] = rbaux
                ia.setrestoringbeam(remove=True) # chan!
            ia.close()

        # Image restoring beam
        ia.open(imagename)
        rbaux = ia.restoringbeam()
        if rbaux:
            rb[imagename] = rbaux
            ia.setrestoringbeam(remove=True) # chan!
        else:
            ia.close()
            raise Exception('Image must have multi beams')
        ia.close()
        
        # Security feature, save the beams just in case
        tag = str(int(round(time.time()))) 
        np.save(f'_beams_{tag}.npy', rb)
        
    # makemask. Despite the name, it is not 'that' important. Maybe
    # convenient to combine with previously defined masks.
    makemask(mode='copy', inpimage=output_mask_name, inpmask=inpmask, 
             output=output_mask_name, overwrite=True)
    
    # Recover the erased multibeams. This means that all beams will be lost
    # if the code does not run until the end of this block. But we have to
    # be brave
    if multi_beams: 
        for mascara in inpmask:
            if mascara in rb:
                ia.open(mascara)
                for ch in range(rb[mascara]['nChannels']):
                    ia.setrestoringbeam(
                        channel=ch, 
                        beam=rb[mascara]['beams'][f'*{ch}']['*0'],
                    )
                ia.close()
        
        ia.open(imagename)
        for ch in range(rb[imagename]['nChannels']):
            ia.setrestoringbeam(channel=ch,
                                beam=rb[imagename]['beams'][f'*{ch}']['*0'])
        ia.close()
            	        	        
def hacer_mascara(image_name: str,
                  mask_threshold: float,
                  output_mask_name: Path,
                  beam_fraction_real: float = 1.,
                  combine_mask: Sequence[Path] = (),
                  use_residual: bool = True) -> None:
    """Creates a mask from flux threshold.
    
    Takes image `image_name.residual` if `use_residual=True` --- which is the
    default. Otherwise, it takes `image_name.image`. It calculates a mask
    with 1s over the `mask_threshold` (assumed a number in the same units as
    image). The task will remove connected components smaller than a
    'fraction' (could be > 1) of the beam size. Lets call this mask `MM`. If
    existing image(s) name(s) is(are) given in `combine_mask`, the task will
    redefine `MM` by combining it -- using a logical `OR` -- with the mask(s)
    of all values greater than zero in (each mask in) `combine_mask`.  Mask
    `MM` is recorded with the the name `output_mask_name`.
    
    IMPORTANT: For the current implementation, it is assumed that the
    associated `.mask`, `.pb`, and `.flux` files have the same grid as
    `image_name.image` and the same multibeam structure.

    Args:
      image_name: image base name.
      mask_threshold: threshold level.
      output_mask_name: output mask file name.
      beam_fraction_real: optional; fraction of the beam to reject small masks.
      combine_mask: optional; masks to combine.
      use_residual: optional; use residual image?

    Notes:
      Version 28 Dic 2017, updated to python 3 on May 2021.
    """
    # If "imageName.mask" exists, it will redefine MM by combination with
    # it. If "imageName.pb" exists, it will redefine MM by combination with
    # it.
    
    #pass
    ## START OF SUBROUTINE CODE
    
    ## Necessary definitions. Determine properties of image/cube. Save the
    ## header, determine whether there is a single beam or multiple. Save
    ## beam areas in an array in the latter case. One important issue is
    ## that the residuals left by tclean DOES NOT have beam
    ## information. Therefore we need to use the header of the .image file.
    
    imagename = Path(f'{image_name}.image')
    casalog.post(f'Image name: {imagename}')
    header = imhead(imagename=imagename)
    multi_beams = 'perplanebeams' in header
    
    # Switches
    step1 = True
    step2 = True
    step3 = True
    step4 = True
    
    # Steps
    if step1:
        beamarea = get_beam_area(header)
    if step2:
        cube = generate_masked_cube(mask_threshold, image_name, use_residual)
        casalog.post('Created mask_threshold mask')
    if step3:
        cube = remove_small_masks(cube, header, beamarea, output_mask_name,
                                  beam_fraction_real)
        casalog.post('Small masks removed')
    if step4:
        combine_masks(output_mask_name, combine_mask, imagename, multi_beams)
        casalog.post(f'Created {output_mask_name}\nGood day')
