#>>> =====================================================#
#>>>             YCLEAN Version 2020
#>>>
#>>> Original from: Yanett Contreras
#>>> Adapted to python>=3.6 by Fernando Olguin
#>>> =====================================================#

# CASA 6.0+
import os
from pathlib import Path

from casatasks import casalog, tclean, imhead, imstat
from casatools import image
import numpy as np

# Import local modules
from .hacer_mascara import hacerMascara
from .second_max_local import secondMaxLocal 

def get_stats(imagename: str, secondary_lobe_level: float, 
        xstat: float = None, planes: tuple = (2, 8), it=0):
    planes = np.arange(*planes, dtype=float)
    planes = list((np.floor(planes/10 * nchan)).astype(int))
    planes = ';'.join(map(str, planes))
    for dummycounter in range(5):
        try:
            rms = 1.42602219 * \
                    imstat(f'{imagename}.tc{it}.image', 
                            chans=planes)['medabsdevmed'][0]
            break
        except TypeError:
            casalog.post(f'Trying imstat again: {dummycounter}')

    # Residual stats and SNR
    for dummycounter in range(5):
        try:
            newxstat = imstat(f'{imagename}.tc{it}.residual')
            casalog.post(f"Stats max: {xstat['max']}")
            break
        except TypeError:
            casalog.post(f'Trying imstat again: {dummycounter}')

    if xstat is None or newxstat['max'] <= xstat['max']:
        xstat = newxstat
    else:
        return rms, None, None

    # Limit level
    limitLevelSNR = xstat['max']/rms * secondary_lobe_level

    return rms, xstat, limitLevelSNR

def yclean(vis: str, imagename: str, **tclean_args):
    # Close and wipe out clean previous files
    img = image()
    img.close()
    img.done()

    # Dirty
    it = 0
    aux = Path(f'{imagename}.tc0.image')
    tclean_args.update({'parallel': True, 'niter': 1})
    if aux.is_dir() == False:
        tclean(vis=vis, imagename=f'{imagename}.tc0', **tclean_args)
    else:
        casalog.post('Data found... skipping first tclean')

    # Useful values
    h0 = imhead(imagename=f'{imagename}.tc0.image')
    nchan = h0['shape'][np.where(h0['axisnames']=='Frequency')[0][0]]

    # The PSF does not change in further iterations
    secondary_lobe_level = secondMaxLocal(f'{imagename}.tc0.psf')
    casalog.post(f'Secondary Lobe PSF Level: {secondary_lobe_level}')

    # RMS calculated in a subset of channels
    rms, xstat, limitLevelSNR = get_stats(imagename, secondary_lobe_level, it=it)

    #### BEGINNING OF WHILE
    while limitLevelSNR > 1.5:
        # Iteration limit
        if it > 10: 
            break
        it += 1
        
        # Some logging
        casalog.post(f"Iter {it}: SNR of Maximum Residual:-----{limitLevelSNR/secondary_lobe_level}")
        # threshold needs to be (slightly?) above limitLevelSNR 
        threshold = f'{limitLevelSNR * 2 * rms * 1.e3}mJy' 
        casalog.post(f"Iter {it}: SNR of threshold:-----{limitLevelSNR}")

        # This is one idea: the masklevel never gets below SNR=4. When the
        # threshold level is high, masklevel is close limitLevelSNR*rms
        masklevel = (limitLevelSNR + 1.5 * np.exp(-(limitLevelSNR-1.5)/1.5)) * rms
        casalog.post(f"Iter {it}: SNR of masklevel: -----{masklevel/rms}")
        
        # The masks are defined based on the previous image and residuals    
        outputMaskName = Path(f"{tclean_args['field']}MASCARA.tc{it-1}.m")
        if not outputMaskName.is_dir():
            aux = f"{tclean_args['field']}MASCARA.tc{it-2}.m"
            combineMask = [aux] if it > 1 else ''
            hacerMascara(imageName=f'{imagename}.tc0',
                    maskThreshold=masklevel, beamFractionReal=0.5,
                    outputMaskName=outputMaskName, useResidual=True, 
                    combineMask=combineMask)
        
            os.system(f'rm -rf {imagename}.tc0.workdirectory/auto*.n*.mask')

            # Run tclean
            tclean_args.update({'parallel': True, 'niter': 100000, 
                'threshold': threshold, 'startmodel': '', 'mask':outputMaskName})
            tclean(vis=vis, imagename=f'{imagename}.tc0', **tclean_args)
            #startmodel=imagename+'.tc'+str(it-1)+'.model',
        else:
            # In case of resuming
            casalog.post(f'Skipping mask {outputMaskName}')
        
        ##### RMS calculated in a subset of channels
        rms, xstat, limitLevelSNR = get_stats(imagename, secondary_lobe_level, 
                xstat=xstat, planes=(2, 9), it=0)
        if xstat is None:
            break

        # Convert to fits
        fitsimage = Path(f'{imagename}.tc_{it}.image.fits')
        if not fitsimage.exists():
            for ext in ['image', 'residual', 'mask']:
                exportfits(f'{imagename}.tc0.{ext}',
                        fitsimage=f'{imagename}.tc_{it}.{ext}.fits', velocity=True)
    #### END OF WHILE

    casalog.post("Reached limit, cleaning to 2. rms, masklevel = 4 sigma")
    it += 1
    outputMaskName = f"{tclean_args['field']}MASCARA.tc{it-1}.m"
    combineMask = [f"{tclean_args['field']}MASCARA.tc{it-2}.m"] if(it>1) else ''
    hacerMascara(imageName=f'{imagename}.tc0', maskThreshold=3.*rms, 
            beamFractionReal=0.5, outputMaskName=outputMaskName,
            useResidual=True, combineMask=combineMask)

    ## Extiende en un par de canalcitos mas a las mascaras. Las lineas no
    ## terminan tan abruptamente, pero ciertamente bajan de 4 sigma lejos de la
    ## vlsr. (POR HACER: UNA FUNCION APARTE QUE HAGA ESTO)
    img.open(outputMaskName)
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
        makemask(mode='expand', inpimage=outputMaskName, inpmask=outputMaskName, 
                inpfreqs=inpfreqs, outfreqs=outfreqs, output=outputMaskName, 
                overwrite=True)  

    # Last clean
    try:
        casalog.post(f"Last threshold: {threshold}")
    except NameError:
        pass
    threshold = f'{2.0*rms*1e3}mJy'
    casalog.post(f'New threshold: {threshold}')
    os.system(f'rm -rf {imagename}.tc0.workdirectory/auto*.n*.mask')
    tclean_args.update({'parallel': True, 'niter': 100000, 
        'threshold': threshold, 'startmodel': '', 'mask':outputMaskName,
        'pblimit':0.1})

    tclean(vis=vis, imagename=f'{imagename}.tc0', **tclean_args)
        #startmodel=imagename+'.tc'+str(it-1)+'.model',

    for ext in ['image', 'pb']:
        exportfits(f'{imagename}.tc0.{ext}',
                fitsimage=f'{imagename}.tc_final.{ext}.fits', velocity=True)

    # Clean up
    os.system(f"rm -rf {tclean_args['field']}MASCARA.tc*.m")

