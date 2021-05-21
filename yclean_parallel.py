#>>> =====================================================#
#>>>             YCLEAN Version 2020
#>>>
#>>> Original from: Yanett Contreras
#>>> Adapted to python>=3.6 by Fernando Olguin
#>>> =====================================================#
"""Automasking routine for ALMA cube CLEANing."""
# CASA 6.0+
from typing import Optional, Tuple
from pathlib import Path
import os

from casatasks import casalog, tclean, imhead, imstat, exportfits, makemask
from casatools import image
import numpy as np

# Import local modules
from hacer_mascara import hacer_mascara
from second_max_local import second_max_local

def get_stats(imagename: str, secondary_lobe_level: float, nchan: int,
              xstat: Optional[dict] = None,
              planes: Tuple[int] = (2, 8),
              nit: int = 0) -> tuple:
    """Calculate image statistics (rms, limits).

    Args:
      imagename: input image.
      secondary_lobe_level: secondary lobe level.
      nchan: number of channels in image.
      xstat: input image statistics.
      planes: channels between which statistics are measured.
      nit: iteration number.

    Returns:
      rms value calculated in the input planes.
      new statistics.
      limit level SNR.
    """
    # Channel sample
    planes = np.arange(*planes, dtype=float)
    planes = list((np.floor(planes / 10 * nchan)).astype(int))
    planes = ';'.join(map(str, planes))
    for dummycounter in range(5):
        try:
            rms = 1.42602219 * \
                imstat(f'{imagename}.tc{nit}.image',
                       chans=planes)['medabsdevmed'][0]
            break
        except TypeError:
            casalog.post(f'Trying imstat again: {dummycounter}')

    # Residual stats and SNR
    for dummycounter in range(5):
        try:
            newxstat = imstat(f'{imagename}.tc{nit}.residual')
            casalog.post(f"Stats max: {xstat['max']}")
            break
        except TypeError:
            casalog.post(f'Trying imstat again: {dummycounter}')

    if xstat is None or newxstat['max'] <= xstat['max']:
        xstat = newxstat
    else:
        return rms, None, None

    # Limit level
    limit_level_snr = xstat['max'] / rms * secondary_lobe_level

    return rms, xstat, limit_level_snr

def yclean(vis: Path, imagename: str, **tclean_args) -> None:
    """Automatic CLEANing.

    Args:
      vis: visibility filename.
      imagename: imagename base name.
      tclean_args: arguments for `tclean`.
    """
    # Close and wipe out clean previous files
    img = image()
    img.close()
    img.done()

    # Dirty
    it = 0
    aux = Path(f'{imagename}.tc0.image')
    tclean_args.update({'parallel': True,
                        'niter': 1})
    if not aux.is_dir():
        tclean(vis=vis, imagename=f'{imagename}.tc0', **tclean_args)
    else:
        casalog.post('Data found ... skipping first tclean')

    # Useful values
    h0 = imhead(imagename=f'{imagename}.tc0.image')
    nchan = h0['shape'][np.where(h0['axisnames']=='Frequency')[0][0]]

    # The PSF does not change in further iterations
    secondary_lobe_level = second_max_local(f'{imagename}.tc0.psf')
    casalog.post(f'Secondary Lobe PSF Level: {secondary_lobe_level}')

    # RMS calculated in a subset of channels
    rms, xstat, limit_level_snr = get_stats(imagename, secondary_lobe_level,
                                            nchan, nit=it)

    #### BEGINNING OF WHILE
    while limit_level_snr > 1.5:
        # Iteration limit
        if it > 10:
            break
        it += 1

        # Some logging
        casalog.post((f'Iter {it}: SNR of Maximum Residual:'
                      f'-----{limit_level_snr/secondary_lobe_level}'))
        # threshold needs to be (slightly?) above limit_level_snr
        threshold = f'{limit_level_snr * 2 * rms * 1.e3}mJy'
        casalog.post(f'Iter {it}: SNR of threshold:-----{limit_level_snr}')

        # This is one idea: the masklevel never gets below SNR=4. When the
        # threshold level is high, masklevel is close limit_level_snr*rms
        masklevel = (limit_level_snr + \
                     1.5 * np.exp(-(limit_level_snr - 1.5) / 1.5)) * rms
        casalog.post(f'Iter {it}: SNR of masklevel: -----{masklevel/rms}')

        # The masks are defined based on the previous image and residuals
        output_mask_name = Path(f"{tclean_args['field']}MASCARA.tc{it-1}.m")
        if not output_mask_name.is_dir():
            aux = Path(f"{tclean_args['field']}MASCARA.tc{it-2}.m")
            combine_mask = [aux] if it > 1 else []
            hacer_mascara(f'{imagename}.tc0', masklevel, output_mask_name,
                          beam_fraction_real=0.5, use_residual=True,
                          combine_mask=combine_mask)

            os.system(f'rm -rf {imagename}.tc0.workdirectory/auto*.n*.mask')

            # Run tclean
            tclean_args.update({'parallel': True,
                                'niter': 100000,
                                'threshold': threshold,
                                'startmodel': '',
                                'mask': output_mask_name})
            tclean(vis=vis, imagename=f'{imagename}.tc0', **tclean_args)
            #startmodel=imagename+'.tc'+str(it-1)+'.model',
        else:
            # In case of resuming
            casalog.post(f'Skipping mask {output_mask_name}')

        ##### RMS calculated in a subset of channels
        rms, xstat, limit_level_snr = get_stats(imagename,
                                                secondary_lobe_level,
                                                nchan,
                                                xstat=xstat,
                                                planes=(2, 9),
                                                nit=0)
        if xstat is None:
            break

        # Convert to fits
        fitsimage = Path(f'{imagename}.tc_{it}.image.fits')
        if not fitsimage.exists():
            for ext in ['image', 'residual', 'mask']:
                exportfits(f'{imagename}.tc0.{ext}',
                           fitsimage=f'{imagename}.tc_{it}.{ext}.fits',
                           velocity=True)
    #### END OF WHILE

    casalog.post('Reached limit, cleaning to 2. rms, masklevel = 4 sigma')
    it += 1
    output_mask_name = f"{tclean_args['field']}MASCARA.tc{it-1}.m"
    if it > 1:
        combine_mask = [Path(f"{tclean_args['field']}MASCARA.tc{it-2}.m")]
    else:
        combine_mask = []
    hacer_mascara(f'{imagename}.tc0', 3. * rms, output_mask_name,
                  beam_fraction_real=0.5, use_residual=True,
                  combine_mask=combine_mask)

    ## Extiende en un par de canalcitos mas a las mascaras. Las lineas no
    ## terminan tan abruptamente, pero ciertamente bajan de 4 sigma lejos de la
    ## vlsr. (POR HACER: UNA FUNCION APARTE QUE HAGA ESTO)
    img.open(output_mask_name)
    lz = img.getchunk()
    img.close()
    lz = list(np.nonzero(np.amax(np.squeeze(lz), axis=(0,1)))[0])
    inpfreqs = []
    outfreqs = []
    if min(lz) > 0 or max(lz) < nchan-1:
        if min(lz) > 0:
            inpfreqs.append(np.asscalar(min(lz)))
            outfreqs += [np.asscalar(min(lz))-1, np.asscalar(min(lz))]
        if max(lz) < nchan-1:
            inpfreqs.append(np.asscalar(max(lz)))
            outfreqs += [np.asscalar(max(lz)), np.asscalar(max(lz))+1]
        makemask(mode='expand', inpimage=output_mask_name,
                 inpmask=output_mask_name, inpfreqs=inpfreqs,
                 outfreqs=outfreqs, output=output_mask_name,
                 overwrite=True)

    # Last clean
    try:
        casalog.post(f'Last threshold: {threshold}')
    except NameError:
        pass
    threshold = f'{2.0*rms*1e3}mJy'
    casalog.post(f'New threshold: {threshold}')
    os.system(f'rm -rf {imagename}.tc0.workdirectory/auto*.n*.mask')
    tclean_args.update({'parallel': True,
                        'niter': 100000,
                        'threshold': threshold,
                        'startmodel': '',
                        'mask':output_mask_name,
                        'pblimit':0.1})
    tclean(vis=vis, imagename=f'{imagename}.tc0', **tclean_args)
        #startmodel=imagename+'.tc'+str(it-1)+'.model',

    # Export FITS
    for ext in ['image', 'pb']:
        exportfits(f'{imagename}.tc0.{ext}',
                   fitsimage=f'{imagename}.tc_final.{ext}.fits', velocity=True)

    # Clean up
    os.system(f"rm -rf {tclean_args['field']}MASCARA.tc*.m")
