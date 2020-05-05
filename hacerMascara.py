import scipy.ndimage
import sys
import numpy as np
from collections import Counter
import time

def hacerMascara(imageName, maskThreshold, outputMaskName, beamFractionReal=1, combineMask='',useResidual=True,):
    """hacerMascara(imageName, maskThreshold, n, outputMaskName): Takes image
    "imageName.residual" if useResidual=True --- which is the
    default. Otherwise, it takes "imageName.image". It calculates a mask
    with 1s over the maskThreshold (assumed a number in the same units as
    image). The task will remove connected components smaller than a
    'fraction' (could be >1) of the beamsize. Lets call this mask MM. If
    existing image(s) name(s) is(are) given in 'combineMask', the task will
    redefine MM by combining it -- using a logical OR -- with the mask(s)
    of all values greater than zero in (each mask in) 'combineMask'.  Mask
    MM is recorded with the the name 'outputMaskName'. IMPORTANT: For the
    current implementation, it is assumed that the associated '.mask',
    '.pb', and '.flux' files have the same grid as 'imageName.image' and
    the same multibeam structure.  Version 28 Dic 2017

    """
    # If "imageName.mask" exists, it will redefine MM by combination with
    # it. If "imageName.pb" exists, it will redefine MM by combination with
    # it.
    
    pass
    ## START OF SUBROUTINE CODE
    
    ## Necessary definitions. Determine properties of image/cube. Save the
    ## header, determine whether there is a single beam or multiple. Save
    ## beam areas in an array in the latter case. One important issue is
    ## that the residuals left by tclean DOES NOT have beam
    ## information. Therefore we need to use the header of the .image file.
    
    myimage=imageName+'.image'
    myfluxim=imageName+'.pb'
    print myimage, myfluxim
    os.system('rm -rf pbimage.im')
    os.system('cp -r '+myfluxim+' pbimage.im')
    myflux='pbimage.im'
    header=imhead(imagename=myimage)
    mm = outputMaskName
    freqAxis=np.where(header['axisnames']=='Frequency')[0][0]
    nChannels=header['shape'][freqAxis]
    multiBeams='perplanebeams' in header
    
    BLOCK1=True
    BLOCK2=True
    BLOCK3=True
    BLOCK4=True
    
    if(BLOCK1):
            ## Block 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
            # The following checks whether there are multiple beams or not. Defines
	    # an array of major and minor beamlengths
	    if(multiBeams):
	        major=[]
	        minor=[]
	        for ch in range(0,nChannels):
	            # in arcsec by default, apparently
	            major.append(header['perplanebeams']['beams']['*'+str(ch)]['*0']['major']['value']) 
	            minor.append(header['perplanebeams']['beams']['*'+str(ch)]['*0']['minor']['value']) 
	        major=np.array(major)
	        minor=np.array(minor)
	    else:
	        major=imhead(imagename=myimage,mode='get',hdkey='beammajor')['value'] # in arcsec by default, apparently
	        minor=imhead(imagename=myimage,mode='get',hdkey='beamminor')['value'] # in arcsec by default, apparently
	
	    unitCDELT=imhead(imagename=myimage,mode='get',hdkey='cdelt2')['unit']
	    if unitCDELT=='rad':
	        pixelsize=(imhead(imagename=myimage,mode='get',hdkey='cdelt2')['value'])/pi*180*3600 # in the header, these CDELT values are in radians
	    beamarea=(major*minor*pi/(4*log(2)))/(pixelsize**2) # beamarea in pixels' area
	    
	    ## End Of Block 1   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
    if(BLOCK2):
	    ## Block 2  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	    # El mero mero del asunto. It creates the mask 'outputMaskName' (this
	    # string is the same as mm) with the maskThreshold. The gridding of the
	    # mask is equivalent to that of the image.
	    
	    if(useResidual):
	        immath(imagename = [imageName+'.residual'],outfile = mm,expr = 'iif(IM0 > '+str(maskThreshold) +',1.0,0.0)', mask=myflux+'>0.2')
	    else:
	        immath(imagename = [myimage],outfile = mm,expr = 'iif(IM0 > '+str(maskThreshold) +',1.0,0.0)', mask=myflux+'>0.2')
	    #immath(imagename = [mm],outfile = mm,expr = 'iif(IM0 > '+str(thresh) +',1.0,0.0)', mask=myflux+'>0.2') # clean vs tclean
	    
	    print 'Created maskThreshold mask.'
	    ## End of Block 2  <<<<<<<<<<<<<<<<<<<<<<
	    
    if(BLOCK3):
    
	    ## Block 3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	    # Remove small masks pieces
	    
	    # The neighboring structure to give to a cube. Otherwise it will assume
	    # that neighboring channels are not independent.
            logfile=open('logfilemasking.txt','w')
	    if nChannels>1: 
	        neighborStructure=[[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[0,1,0],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]]
	    else:
	        neighborStructure=[[0,1,0],[1,1,1],[0,1,0]]
	
	    print mm
	    ia.open(mm)    # Open the mask
	    mask=ia.getchunk() # Get the data in an array, usually of dimension 4
	                       # (spatial x spatial x pol x frequency/velocity
	                       # or (spatial x spatial x frequency/velocity x pol)
	    if nChannels>1: # In case of multiple beams
	        polAxisPosition=np.where(header['axisnames']=='Stokes')[0][0]
	        mask=np.squeeze(mask) # remove extra redundant dimension (polarization?) 
                print 'ShapeMASK',np.shape(mask)
	        # separate and label connected components
                print mask.ndim, np.array(neighborStructure).ndim
	        labeled,j=scipy.ndimage.label(mask,structure=neighborStructure) 
	        labelStack=set(range(1,j+1))
	        print "Labeled mask. "+str(j)+" pieces"
	        for canal in range(nChannels):
	            #if (canal%20==0):
                    print "Channel # "+str(canal)
	            counts=Counter(np.ndarray.flatten(labeled[:,:,canal]).tolist())
	            labelsInChannel=counts.keys()
	            del labelsInChannel[0]
                    #print set(labelsInChannel)
                    print 'possible issue:',(set(labelsInChannel)<=labelStack)
	            if(set(labelsInChannel)<=labelStack):
	               labelStack= labelStack - set(labelsInChannel)
	            else:
	               raise Exception("Error in labeling") 
	            for i in labelsInChannel:
	                sumapixels=counts[i]
                        logfile.write(str(canal)+'....'+str(sumapixels)+'..'+str(beamarea[canal]*beamFractionReal)+'..'+str(sumapixels<beamarea[canal]*beamFractionReal)+'\n')
	                # Zap small masks
	                ###if multiBeams:
                        if(sumapixels<(beamarea[canal]*beamFractionReal)):
                            mask[:,:,canal]*=(1-(labeled[:,:,canal]==i))
	                ###else:
	                ###    if(sumapixels<beamarea*beamFractionReal):
	                ###        mask[:,:,canal]*=(1-(labeled[:,:,canal]==i))        
	
	        # The following gets the mask to the original dimensions of the image
	        # It is assumed that 'Stokes' axis has no dimension
	        axesOrder=[1,2,3]
	        axesOrder.insert(polAxisPosition,0)
	        mask=np.transpose(mask[None,:],tuple(axesOrder)) 
	                
	    else: # In case of single beam
	        labeled,j=scipy.ndimage.label(mask,structure=neighborStructure)
	        print "Labeled mask"
	        for i in range(1,j):
	            sumapixels=np.sum((labeled==j).astype(int),axis=(0,1))
	            if sumapixels<beamarea*beamFractionReal:
	                mask[labeled==i]=0
	           
	    ia.putchunk(mask)
	    ia.done()
	    ia.close()
            logfile.close()
	    print 'Small masks removed'
	    ## End of Block 3 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	    ## The mask without the small pieces should be updated in outputMaskName
	    
    if(BLOCK4):
	    ##  Block 4 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	    ##  Combine the main mask with the rest 
	    inpmask=[mm]
	    #inpmask.append(imageName+'.mask') if(os.path.exists(imageName+'.mask')) else 1
	    #inpmask.append(imageName+'.pb') if(os.path.exists(imageName+'.pb')) else 1
	    if isinstance(combineMask, basestring):
	        inpmask.append(combineMask) if(os.path.exists(combineMask)) else 1   
	    elif all(isinstance(item, basestring) for item in combineMask):
	        for cM in combineMask:
	            inpmask.append(cM) if(os.path.exists(cM)) else 1 
	    else:
                raise Exception("Error in combineMask keyword")
            print "Masks to combine:",inpmask
            #sys.exit()
	    
	    # If multiBeams, delete all beams to run makemask
            if(multiBeams):
	        rb={}
	        for mascara in inpmask:
	            #print mascara
	            ia.open(mascara)
                    rbaux=ia.restoringbeam()
                    if(rbaux):# check it has beam information
                        rb[mascara]=rbaux
                        ia.setrestoringbeam(remove=True) # chan!
	            ia.close()
	        ia.open(myimage)
	        rbaux=ia.restoringbeam()
                if(rbaux):
                    rb[myimage]=rbaux
                    ia.setrestoringbeam(remove=True) # chan!
                else:
                    ia.close()
                    raise Exception("Image must have multiBeams")
	        ia.close()
	    
                # security feature, save the beams just in case
                tag=str(int(round(time.time()))) 
                np.save('_beams_'+tag+'.npy',rb) 
            
            # makemask. Despite the name, it is not 'that' important. Maybe
	    # convenient to combine with previously defined masks.
	    makemask(mode='copy',inpimage=mm,inpmask=inpmask,output=mm,overwrite=True)
	    
	    # Recover the erased multibeams. This means that all beams will be lost
	    # if the code does not run until the end of this block. But we have to
	    # be brave
	    if(multiBeams): 
	        for mascara in inpmask:
                    if(mascara in rb):
                        ia.open(mascara)
                        for ch in range(rb[mascara]['nChannels']):
                            ia.setrestoringbeam(channel=ch,beam=rb[mascara]['beams']["*"+str(ch)]["*0"])
                        ia.close()
	        
	        ia.open(myimage)
                for ch in range(rb[myimage]['nChannels']):
	            ia.setrestoringbeam(channel=ch,beam=rb[myimage]['beams']["*"+str(ch)]["*0"])
	        ia.close()
	        	        	        
            print "Created "+mm+'\nGood day'
	    ##  End of Block 4 <<<<<<<<<<<<<<<<<<<<<<<

## END OF SUBROUTINE CODE
