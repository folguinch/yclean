import scipy.ndimage

def domask_lines(imagename, thresh,n, fl=False, useimage=False):

    print n, n=='tc0', n=='0'
    min_number_of_pixels = 10 # number of pixels from which I believe that a source is real (I am using half of the synthesized beam, other people use the synthesized beam)
    if n!='tc1':
        mymask=imagename+'.'+str(n)+'.mask'
    else:
        mymask=imagename+'.tc0.fullmask.pb2.min'
    myimage=imagename+'.'+str(n)+'.image'
    #myflux=imagename+'.'+str(n)+'.flux'
    myflux=imagename+'.'+str(n)+'.pb'
    myresidual=imagename+'.'+str(n)+'.residual'
    maskim=imagename+'.'+str(n)+'.fullmask'
    
    print myflux, myimage
    print "Using threshold : ", thresh
    print 'rm -rf '+imagename+'_threshmask.' +str(n)
    os.system('rm -rf '+imagename+'_threshmask.' +str(n))
    os.system('rm -rf '+maskim+'.pb2')
    os.system('rm -rf '+maskim+'.pb2.min')
    #os.system('rm -rf '+maskim+'.pb2.mintest')
    os.system('rm -rf '+imagename+'.'+str(n)+'.fullmask')
    #print "deleted "+maskim+'.pb*'
    
    threshmask = imagename+'_threshmask.' +str(n)
    pixelmin=min_number_of_pixels
    #print "Using pixelmin=",pixelmin
    try:
        major=imhead(imagename=myimage,mode='get',hdkey='beammajor')['value']
        minor=imhead(imagename=myimage,mode='get',hdkey='beamminor')['value']
    except:
        print 'trying this instead', myimage
        major=imhead(imagename=myimage)['perplanebeams']['beams']['*0']['*0']['major']['value']
        minor=imhead(imagename=myimage)['perplanebeams']['beams']['*0']['*0']['minor']['value']

    print "Major, minor axis: ", major, minor
    pixelsize=float(cell.split('arcsec')[0])
    beamarea=(major*minor*pi/(4*log(2)))/(pixelsize**2)
    
    if useimage==True:
        print "Using Image"
        immath(imagename = [myimage],outfile = threshmask,expr = 'iif(IM0 > '+str(thresh) +',1.0,0.0)')
        #immath(imagename = [myimage],outfile = threshmask,expr = 'iif(IM0 > '+str(thresh) +',1.0,0.0)', mask=myflux+'>0.2')
    else:
        immath(imagename = [myresidual],outfile = threshmask,expr = 'iif(IM0 > '+str(thresh) +',1.0,0.0)')
        #immath(imagename = [myresidual],outfile = threshmask,expr = 'iif(IM0 > '+str(thresh) +',1.0,0.0)', mask=myflux+'>0.2')
        
    if (n!='tc0') and (fl==False):
        print 'Combining with previous mask..'
        os.system('cp -r '+threshmask+' thresh_temp')
        os.system('cp -r '+mymask+' mymask_temp')
        os.system('cp -r '+myflux+' myflux_temp')
        makemask(mode='copy',inpimage=myimage,inpmask=['thresh_temp','mymask_temp'],output=maskim)
        print "Flux",myflux 
        imsubimage(imagename=maskim, mask='myflux_temp>'+str(0.2),outfile=maskim+'.pb2')     
        print 'Combined mask ' +maskim+' generated.'
        os.system('rm -rf *_temp')
            
            
    
    elif (n=='tc0') or (fl==True):
        print "Creating mask from image: ",myimage
        os.system('cp '+myimage+' myimage_temp')
        immath(imagename = [myimage],outfile = threshmask,expr = 'iif(IM0 > '+str(thresh) +',1.0,0.0)')
        
        print 'Making fresh new mask from image/residual'
        os.system('cp -r '+threshmask+' '+maskim+'.pb2')
        print 'This is the first loop'
        os.system('rm -rf *_temp')
    
        
                
    # Remove small masks
    os.system('cp -r '+maskim+'.pb2 ' +maskim+'.pb2.min')
    #os.system('cp -r '+maskim+'.pb2 ' +maskim+'.pb2.mintest')
    #print 'cp -r '+maskim+'.pb2 ' +maskim+'.pb2.min'
    maskfile=maskim+'.pb2.min'
    #print maskfile
    ia.open(maskfile)
    mask=ia.getchunk()           
    labeled,j=scipy.ndimage.label(mask)                     
    myhistogram = scipy.ndimage.measurements.histogram(labeled,0,j+1,j+1)
    object_slices = scipy.ndimage.find_objects(labeled)
    threshold=beamarea*pixelmin
    #print threshold
    for i in range(j):
        if myhistogram[i+1]<threshold:
            mask[object_slices[i]] = 0
            
            
    ia.putchunk(mask)
    ia.done()
    print 'Small masks removed and ' +maskim +'.pb2.min generated.'

