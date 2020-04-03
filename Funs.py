import numpy as np
import scipy.stats as st
import h5py
import matplotlib.pyplot as plt

# save remote sensing image classification maps
def save_cmap(img, cmap, fname):
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi = height) 
    plt.close()
    
# visualize RS data to RGB figure
def strimg255(im, perc=0.5):
    maxx = np.percentile(im,100-perc)
    minn = np.percentile(im,perc)
    im[im>maxx] = maxx
    im[im<minn] = minn 
    im_new = np.fix((im-minn)/(maxx-minn)*255).astype(np.uint8)
    return im_new

# read vnp46a1 data
'''
dnb = 4
utc = -1
moon phase angle = 10
m11 radiance cloud = 20 
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

# gdal: given latitude and longitude, check its pixel location 
def latlng2pix(geo,lat,lng):
    imx = (lat+geo[5]/2-geo[3])/geo[5] 
    imy = (lng+geo[1]/2-geo[0])/geo[1]
    return int(imx),int(imy)

# read vnp46a1 data
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

# Generate a Gaussian kernel, rbf = gkern(5,2.5)*273
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

# plot confusion matrix, lcz-17
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
    
# gdal, set GeoTransform with new bgx and bgy
def setGeo(geotransform,bgx,bgy):
    reset0 = geotransform[0] + bgx*geotransform[1]
    reset3 = geotransform[3] + bgy*geotransform[5]
    reset = (reset0,geotransform[1],geotransform[2],
             reset3,geotransform[4],geotransform[5])
    return reset

# save classification maps
if False:
    c_paviaC = ['#000000','#0000FF','#228B22','#7BFC00', '#FF0000', '#724A12', '#C0C0C0',
              '#00FFFF', '#FF8000', '#FFFF00']
    c_salinas = ['#000000','#DCB809','#03009A','#FE0000','#FF349B','#FF66FF',
              '#0000FD','#EC8101','#00FF00','#838300','#990099','#00F7F1',
              '#009999','#009900','#8A5E2D','#67FECB','#F6EF00']
    c_indian = ['#000000','#FFFC86','#0037F3','#FF5D00','#00FB84','#FF3AFC',
              '#4A32FF','#00ADFF','#00FA00','#AEAD51','#A2549E','#54B0FF',
              '#375B70','#65BD3C','#8F462C','#6CFCAB','#FFFC00']
    c_paviaU = ['#000000','#CACACA','#02FF00','#00FFFF','#088505','#FF00FE','#AA562E','#8C0085','#FD0000', '#FFFF00']
    c_pu12 = ['#000000','#CACACA','#02FF00','#00FFFF','#088505', '#FF00FE', '#AA562E', '#8C0085',
              '#FD0000','#FFFF00','#858688','#7D5E4C','#3D3733']
def save_cmap_pc(img, cmap, fname):
    colors = ['#000000','#0000FF','#228B22','#7BFC00', '#FF0000', '#724A12', '#C0C0C0',
              '#00FFFF', '#FF8000', '#FFFF00']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=9)
    plt.savefig(fname, dpi = height)
    plt.close()
    
def save_cmap_salinas(img,cmap,fname):
    colors = ['#000000','#DCB809','#03009A','#FE0000','#FF349B','#FF66FF',
              '#0000FD','#EC8101','#00FF00','#838300','#990099','#00F7F1',
              '#009999','#009900','#8A5E2D','#67FECB','#F6EF00']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=16)
    plt.savefig(fname, dpi = height)
    plt.close()
    
def save_cmap_indian(img,cmap,fname):
    colors = ['#000000','#FFFC86','#0037F3','#FF5D00','#00FB84','#FF3AFC',
              '#4A32FF','#00ADFF','#00FA00','#AEAD51','#A2549E','#54B0FF',
              '#375B70','#65BD3C','#8F462C','#6CFCAB','#FFFC00']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=16)
    plt.savefig(fname, dpi = height)
    plt.close()

def save_cmap(img, cmap, fname):
    colors = ['#000000','#CACACA','#02FF00','#00FFFF','#088505','#FF00FE','#AA562E','#8C0085','#FD0000', '#FFFF00']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=9)
    plt.savefig(fname, dpi = height)
    plt.close()
    
def save_cmap_pu12(img, cmap, fname):
    colors = ['#000000','#CACACA','#02FF00','#00FFFF','#088505', '#FF00FE', '#AA562E', '#8C0085',
              '#FD0000','#FFFF00','#858688','#7D5E4C','#3D3733']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=12)
    plt.savefig(fname, dpi = height)
    plt.close()

