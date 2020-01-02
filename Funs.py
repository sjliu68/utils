import numpy as np
import scipy.stats as st

'''
['BrightnessTemperature_M12',
 'BrightnessTemperature_M13',
 'BrightnessTemperature_M15', 
 'BrightnessTemperature_M16', 
 'DNB_At_Sensor_Radiance_500m',  4
 'Glint_Angle', 
 'Granule', 
 'Lunar_Azimuth', 
 'Lunar_Zenith', 
 'Moon_Illumination_Fraction', 
 'Moon_Phase_Angle', 
 'QF_Cloud_Mask',  11
 'QF_DNB',  12
 'QF_VIIRS_M10', 
 'QF_VIIRS_M11',
 'QF_VIIRS_M12',
 'QF_VIIRS_M13', 
 'QF_VIIRS_M15', 
 'QF_VIIRS_M16', 
 'Radiance_M10', 
 'Radiance_M11', 
 'Sensor_Azimuth', 
 'Sensor_Zenith', 
 'Solar_Azimuth', 
 'Solar_Zenith', 
 'UTC_Time', 25]
'''
def read_vnp(filename,idx=4):
    f = h5py.File(filename,'r')
    
    a = list(f.keys())[0]
    b = list(f[a].keys())[1]
    c = list(f[a][b].keys())[0]
    d = list(f[a][b][c].keys())[0]
    e = list(f[a][b][c][d].keys()) # 26 files in this list
    
    data = f[a][b][c][d]
    data = data[e[idx]]
    data = data[:]
    
    return data

# Generate a Gaussian kernel 
# rbf = gkern(5,2.5)*273
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

# plot confusion matrix
def plot_confusion_matrix(cfm,name,dpi=80):
    lcz = ['LCZ-1','LCZ-2','LCZ-3','LCZ-4','LCZ-5','LCZ-6','LCZ-7','LCZ-8','LCZ-9',
           'LCZ-10','LCZ-A','LCZ-B','LCZ-C','LCZ-D','LCZ-E','LCZ-F','LCZ-G']
    plt.figure(figsize=(12,10),dpi=dpi)
    imx = cfm.shape[0]
    cm = np.zeros([imx,imx])
    for i in range(imx):
        cm[:,i] = cfm[:,i]/cfm[:,i].sum()*100
    plt.imshow(cm,interpolation='nearest')
    plt.title(name)
    plt.colorbar()
    
    tick_marks=np.arange(imx)
    plt.xticks(tick_marks,np.arange(imx)+1,fontsize=6,rotation=45)
    plt.yticks(tick_marks,np.arange(imx)+1,fontsize=6,rotation=45)
    plt.ylabel('Predicted Label')
    plt.xlabel('Reference Label')
    fmt = '.0f'
    thresh = cm.max() / 2.
    for i in range(imx):
        for j in range(imx):
            plt.text(j, i, format(cfm[i,j], fmt),
                    ha="center", va="center",fontsize=10,
                    color="white" if cm[i,j] < thresh else "black")
    plt.xlim(-0.5,imx-0.5)
    plt.ylim(imx-0.5,-0.5)
    plt.xticks(np.arange(0,17),lcz)
    plt.yticks(np.arange(0,17),lcz)
    plt.tight_layout()
    plt.savefig(name+'.pdf')
    plt.show()
    
def setGeo(geotransform,bgx,bgy):
    reset0 = geotransform[0] + bgx*geotransform[1]
    reset3 = geotransform[3] + bgy*geotransform[5]
    reset = (reset0,geotransform[1],geotransform[2],
             reset3,geotransform[4],geotransform[5])
    return reset

