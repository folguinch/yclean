import numpy as np
# 04 Ene 2017
def checkExtremeChannels(cubename,numberOfChannels=5,side='Both'):
    """Check the statistics of the numberOfChannels extreme channels of
    cube. Large deviations from the general statistics will be indicated. A
    suggested start and channel number will be the output. """

    if not os.path.exists(cubename):
        raise Exception(cubename+" does not exist.")
        
    header=imhead(imagename=cubename)
    allChannels=header['shape'][np.where(header['axisnames']=='Frequency')[0][0]]
    collapseAxes=list(np.where((header['axisnames']=='Frequency')==0)[0])
    cubeStats=imstat(imagename=cubename, axes=collapseAxes)

    quotient3=cubeStats['q3']/np.median(cubeStats['rms'])
    quotientRms=cubeStats['rms']/np.median(cubeStats['rms'])
    # channels with large noise or large signal
    nchtot=2*numberOfChannels if(side=='Both') else numberOfChannels
    marked=set(np.argsort(-quotientRms)[range(nchtot)]) & set(np.argsort(-quotient3)[range(nchtot)])
    # channels with zero signal
    marked=marked | set(np.where(cubeStats['rms']==0)[0])
    # outliers according with Tukey's criterion
    marked=marked & (set(outliersIndexesTukey(quotient3)) | set(outliersIndexesTukey(quotientRms)))
    
    startchan=0
    endchan=allChannels-1
    #print endchan
    if side=='Beginning' or side=='Both':
        checkChannels=set(range(numberOfChannels))
        channelsAux=np.sort(list(marked & checkChannels))
        if len(channelsAux):
            startchan=channelsAux[-1]+1
    if side=='End' or side=='Both':
        checkChannels=set(range(allChannels-numberOfChannels,allChannels))
        channelsAux=np.sort(list(marked & checkChannels))
        if len(channelsAux):
            endchan=channelsAux[0]-1
            
    return str(startchan)+"~"+str(endchan)
        
        
def outliersIndexesTukey(data, kValue=1.5):
    q1=np.percentile(data,25)
    q3=np.percentile(data,75)
    rT=q1+np.array([-1,1])*kValue*(q3-q1)
    return np.where((rT[0] > data)| (data > rT[1]))[0]
    
