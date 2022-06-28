"""Find the secondary lobe level."""
import itertools

from casatools import image
#from spectral_cube import SpectralCube
import numpy as np

def second_max_local(psffile: 'Path'):
    """Determine the psf secondary lobe level."""
    # Open image
    img = image()
    img.open(psffile)
    psf = img.getchunk()
    #img = SpectralCube.read(psffile, format='casa_image', use_dask=True)
    #img
    #psf = 

    # Data shape
    if len(img.shape()) == 3:
        nx, ny, nc = img.shape()
    else:
        nx, ny, dd, nc = img.shape()
    img.close()
    img.done()
    psf = psf.reshape(nx, ny, nc)
    psf = np.squeeze(psf)
    psf = psf[:, :, np.floor(nc/2).astype(int)]

    # Find second
    aux = np.ones(psf.shape)
    rng = range(-1, 2)
    for dx, dy in itertools.product(rng, rng):
        if dx**2 + dy**2 > 0:
            aux *= psf > np.roll(np.roll(psf, dx, axis=0), dy, axis=1)
    psf = np.sort(np.ndarray.flatten(psf * aux))
    second = psf[-2] / psf[-1]

    return second

#def return_secondmax(fitsfile):
#    auxfits=pf.open(fitsfile)
#    plane=np.shape(auxfits[0].data)[0]/2
#
#    neighborhood_size = 5
#    threshold = 0.01
#
#    data = auxfits[0].data[plane,:,:]
#
#    data_max = filters.maximum_filter(data, neighborhood_size)
#   
#    maxima = (data == data_max)
#    data_min = filters.minimum_filter(data, neighborhood_size)
#    diff = ((data_max - data_min) > threshold)
#    maxima[diff == 0] = 0
#
#    labeled, num_objects = ndimage.label(maxima)
#    slices = ndimage.find_objects(labeled)
#    x, y,value = [], [],[]
#    for dy,dx in slices:
#        x_center = (dx.start + dx.stop - 1)/2
#        x.append(x_center)
#        y_center = (dy.start + dy.stop - 1)/2    
#        y.append(y_center)
#        value.append(data[y_center,x_center])
#
#    #plt.imshow(data)
#    #plt.savefig('/tmp/data.png', bbox_inches = 'tight')
#        
#    #plt.autoscale(False)
#    #plt.plot(x,y, 'ro')
#    #plt.savefig('/tmp/result.png', bbox_inches = 'tight')
#        
#
#    secondmax=np.sort(value)[-2]
#    return secondmax
