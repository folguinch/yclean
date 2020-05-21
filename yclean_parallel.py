#>>> =====================================================#
#>>>             YCLEAN Version ENERO 2016
#>>> =====================================================#

# CASA 4.7

import os
import scipy
import numpy as np


#### INFO DIRECTORIES & SUBROUTINES

#diryclean = os.path.expanduser('~/binary_project/yclean/final_scripts')
execfile(os.path.join(diryclean, 'hacerMascara.py'))
execfile(os.path.join(diryclean, 'secondMaxLocal.py'))
execfile(os.path.join(diryclean, "checkExtremeChannels.py"))
##### CUBE DATA ######


# close and wipe out clean previous files
ia.close()
ia.done()
#rmtables(source+"*tc*")
#os.system("rm -f "+source+"*tc*/table.lock")
#rmtables(source+"*tc*")
#os.system("rm -rf "+source+"*tc*")

if os.path.isdir(imagename+'.tc0.image')==False:
    tclean(vis = vis,
           imagename = imagename+'.tc0',
           field = source,
           spw=spwline,
           gridder = gridder,
           #wprojplanes = wprojplanes, 
           specmode = specmode,
           outframe = outframe,
           width = width, # 
           start = start,
           nchan = nchan,
           restfreq = restfreq, 
           interpolation = interpolation,
           interactive = interactive,
           niter=1,
           imsize = imsize,
           cell = cell,
           weighting = weighting,
           robust=robust, 
           deconvolver = deconvolver, 
           scales = scales,
           phasecenter = phasecenter,
           uvtaper=uvtaper,
           pblimit=pblimit,
           parallel=True)

else:
    print "Data found... skipping first tclean"


h0=imhead(imagename=imagename+'.tc0.image')
nchan=h0['shape'][np.where(h0['axisnames']=='Frequency')[0][0]]

# The PSF does not change in further iterations
secondary_lobe_level=secondMaxLocal(imagename+'.tc0.psf')
print "Secondary Lobe PSF Level:" ,secondary_lobe_level



##### RMS calculated in a subset of channels
planes=list((np.floor(np.array([2,3,4,5,6,7],dtype=float)/10*nchan)).astype(int))
planes=';'.join(map(str,planes))
rms=1.42602219*imstat(imagename+'.tc'+str(it)+'.image', chans=planes)['medabsdevmed'][0]
####
#rms=4e-3


for dummycounter in range(5):
    try:
        xstat = imstat(imagename+'.tc'+str(it)+'.residual')
        casalog.post('Stats max: %r' % (xstat['max'],))
        break
    except TypeError:
        casalog.post('Trying imstat again: %i' % dummycounter)
limitLevelSNR=float(xstat['max']/rms*secondary_lobe_level)


#### BEGINNING OF WHILE
while limitLevelSNR>1.5:
    if it>10: break
    it=it+1
    #sys.exit("EXIT HERE")    
    
    print "Iter "+str(it)+": SNR of Maximum Residual: -----"+str(limitLevelSNR/secondary_lobe_level)    
    # threshold needs to be (slightly?) above limitLevelSNR 
    threshold=str(limitLevelSNR*2*rms*1.e3)+'mJy' 
    print "Iter "+str(it)+": SNR of threshold: -----"+str(limitLevelSNR)

    # This is one idea: the masklevel never gets below SNR=4. When the
    # threshold level is high, masklevel is close limitLevelSNR*rms
    masklevel=(limitLevelSNR+1.5*exp(-(limitLevelSNR-1.5)/1.5))*rms
    print "Iter "+str(it)+": SNR of masklevel: -----"+str(masklevel/rms)
    
    # The masks are defined based on the previous image and residuals    
    combineMask=[source+'MASCARA.tc'+str(it-2)+'.m'] if(it>1) else ''
    hacerMascara(imageName=imagename+'.tc0',
                 maskThreshold=masklevel,beamFractionReal=0.5, 
                 outputMaskName=source+'MASCARA.tc'+str(it-1)+'.m',
                 useResidual=True, combineMask=combineMask)
   
    os.system('rm -rf '+imagename+'.tc0.workdirectory/auto*.n*.mask')


    tclean(vis = vis,
           imagename = imagename+'.tc0',
           field = source,
           spw=spwline,
           gridder = gridder,
           #wprojplanes = wprojplanes, 
           specmode = specmode,
           outframe = outframe,
           width = width, #
           start = start,
           nchan = nchan,
           restfreq = restfreq, 
           interpolation = interpolation,
           interactive = interactive,
           niter=100000,
           imsize = imsize,
           cell = cell,
           weighting = weighting,
           robust=robust, 
           deconvolver = deconvolver, 
           scales = scales,
           phasecenter = phasecenter, 
           threshold = threshold,
           startmodel='',
           #startmodel=imagename+'.tc'+str(it-1)+'.model',
           mask=source+'MASCARA.tc'+str(it-1)+'.m',
           uvtaper=uvtaper,
           pblimit=pblimit,
           parallel=True)
    
    ##### RMS calculated in a subset of channels
    planes=list((np.floor(np.array([2,3,4,5,6,7,8],dtype=float)/10*nchan)).astype(int))
    planes=';'.join(map(str,planes))
    rms=1.42602219*imstat(imagename+'.tc0.image', chans=planes)['medabsdevmed'][0]
    ####
    #rms=4e-3

    for dummycounter in range(5):
        try:
            xstatnew = imstat(imagename+'.tc0.residual')
            casalog.post('New stats max: %r' % (xstatnew['max'],))
            break
        except TypeError:
            casalog.post('Trying imstat again: %i' % dummycounter)

    if xstatnew['max']>xstat['max']:
        break
    else:
        xstat=xstatnew
    limitLevelSNR=float(xstat['max']/rms*secondary_lobe_level)
    exportfits(imagename+'.tc0.image', fitsimage=imagename+'.tc_'+str(it)+'.fits', velocity=True)
    exportfits(imagename+'.tc0.residual', fitsimage=imagename+'.tc_'+str(it)+'.residual.fits', velocity=True)
    exportfits(imagename+'.tc0.mask', fitsimage=imagename+'.tc_'+str(it)+'.mask.fits', velocity=True)
    


#### END OF WHILE


print "Reached limit, cleaning to 2. rms, masklevel = 4 sigma"
it+=1
combineMask=[source+'MASCARA.tc'+str(it-2)+'.m'] if(it>1) else ''
hacerMascara(imageName=imagename+'.tc0',
             maskThreshold=3.*rms,beamFractionReal=0.5, 
             outputMaskName=source+'MASCARA.tc'+str(it-1)+'.m',
             useResidual=True, combineMask=combineMask)


## Extiende en un par de canalcitos mas a las mascaras. Las lineas no
## terminan tan abruptamente, pero ciertamente bajan de 4 sigma lejos de la
## vlsr. (POR HACER: UNA FUNCION APARTE QUE HAGA ESTO)
ia.open(source+'MASCARA.tc'+str(it-1)+'.m')
lz=ia.getchunk()
ia.close()
lz=list(np.nonzero(np.amax(np.squeeze(lz),axis=(0,1)))[0])
inpfreqs=[]
outfreqs=[]
if(min(lz)>0 or max(lz)<nchan-1):
    if(min(lz)>0):
        inpfreqs.append(np.asscalar(min(lz)))
        outfreqs+=[np.asscalar(min(lz))-1,np.asscalar(min(lz))]
    if(max(lz)<nchan-1):
        inpfreqs.append(np.asscalar(max(lz)))
        outfreqs+=[np.asscalar(max(lz)),np.asscalar(max(lz))+1]
    makemask(mode='expand', inpimage=source+'MASCARA.tc'+str(it-1)+'.m', 
             inpmask=source+'MASCARA.tc'+str(it-1)+'.m', 
             inpfreqs=inpfreqs, outfreqs=outfreqs,
             output=source+'MASCARA.tc'+str(it-1)+'.m', overwrite=True)  

print "Last threshold: ", threshold
print "New threshold: ", str(2.0*rms*1e3)+'mJy'

os.system('rm -rf '+imagename+'.tc0.workdirectory/auto*.n*.mask')

tclean(vis = vis,
       imagename = imagename+'.tc0',
       field = source,
       gridder = gridder,
       spw=spwline,
       #wprojplanes = wprojplanes, 
       specmode = specmode,
       outframe = outframe,
       width = width, #
       start = start,
       nchan = nchan,
       restfreq = restfreq, 
       interpolation = interpolation,
       interactive = interactive,
       niter=100000,
       imsize = imsize,
       cell = cell,
       weighting = weighting,
       robust=robust,
       deconvolver = deconvolver, 
       scales = scales,
       phasecenter = phasecenter, 
       threshold = str(2.0*rms*1e3)+'mJy',
       startmodel='',
       #startmodel=imagename+'.tc'+str(it-1)+'.model',
       mask=source+'MASCARA.tc'+str(it-1)+'.m',
       uvtaper=uvtaper,
       pblimit=0.1,
       parallel=True)


exportfits(imagename+'.tc0.image', fitsimage=imagename+'.tc_final.fits', velocity=True)
exportfits(imagename+'.tc0.pb', fitsimage=imagename+'.tc_final.pb.fits', velocity=True)
os.system('rm -rf '+source+'MASCARA.tc*.m')
