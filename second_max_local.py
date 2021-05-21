import pyfits as pf
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

def secondMaxLocal(psffile):
    ia.open(psffile)
    bim=ia.getchunk()
    if len(ia.shape())==3:
        (nx,ny,c)=ia.shape()
    else:
        (nx,ny,dd,c)=ia.shape()
    ia.close()
    ia.done()
    bim=bim.reshape(nx,ny,c)
    bim=np.squeeze(bim)
    
    bim=bim[:,:,np.floor(c/2).astype(int)]
    bul=np.ones(bim.shape)
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            if(dx**2+dy**2>0):
                bul*=(bim>np.roll(np.roll(bim,dx,axis=0),dy,axis=1))
    
    bim=np.sort(np.ndarray.flatten(bim*bul))           
    second=bim[-2]/bim[-1]
    return second

    
def return_secondmax(fitsfile):
    auxfits=pf.open(fitsfile)
    plane=np.shape(auxfits[0].data)[0]/2

    neighborhood_size = 5
    threshold = 0.01

    data = auxfits[0].data[plane,:,:]

    data_max = filters.maximum_filter(data, neighborhood_size)
   
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y,value = [], [],[]
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
        value.append(data[y_center,x_center])

    #plt.imshow(data)
    #plt.savefig('/tmp/data.png', bbox_inches = 'tight')
        
    #plt.autoscale(False)
    #plt.plot(x,y, 'ro')
    #plt.savefig('/tmp/result.png', bbox_inches = 'tight')
        

    secondmax=np.sort(value)[-2]
    return secondmax
