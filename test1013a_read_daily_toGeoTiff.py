# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:45:13 2019

@author: light_pollusion_team

Read daily VIIRS data VNP46A1

"""

import h5py
import glob
import matplotlib.pyplot as plt
import numpy as np
import gdal


def save_tif(data,name,geo,prj,dtype='float'):
    if dtype=='float':
        dtype = gdal.GDT_Float32
    elif dtype=='int':
        dtype = gdal.GDT_UInt16
        
    if len(data.shape)==2:
        imx,imy = data.shape
        imz = 1
    elif len(data.shape)==3:
        imx,imy,imz = data.shape
    
    outdata = gdal.GetDriverByName('GTiff').Create(name+'.tif',imy,imx,imz,dtype)
    outdata.SetGeoTransform(geo)
    outdata.SetProjection(prj)
    for i in range(imz):
        outdata.GetRasterBand(i+1).WriteArray(data[:,:,i])
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    
def setGeo(geotransform,bgx,bgy,x_offset=0):
    if x_offset==0:
        x_offset = geotransform[1]
        y_offset = geotransform[5]
    else:
        x_offset = x_offset
        y_offset = -x_offset
    reset0 = geotransform[0] + bgx*geotransform[1]
    reset3 = geotransform[3] + bgy*geotransform[5]
    reset = (reset0,x_offset,geotransform[2],
             reset3,geotransform[4],y_offset)
    return reset


def read_vnp(filename,idx=4):
    f = h5py.File(filename,'r')

#    print("Keys: %s" % f.keys())

    a = list(f.keys())[0]
    b = list(f[a].keys())[1]
    c = list(f[a][b].keys())[0]
    d = list(f[a][b][c].keys())[0]
    e = list(f[a][b][c][d].keys()) # 26 files in this list
    
    data = f[a][b][c][d]
    data = data[e[idx]]
    data = data[:]
    
    return data

def strimg255(im, perc=0.5):
    maxx = np.percentile(im,100-perc)
    minn = np.percentile(im,perc)
    im[im>maxx] = maxx
    im[im<minn] = minn
    im_new = np.fix((im-minn)/(maxx-minn)*255).astype(np.uint8)
    return im_new

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

#%%
q11b = ['000','001','010','011','101']
q11c = ['00','01','10','11']
q11d = ['00','01','10','11']

#%%
files = glob.glob('D:/research/t18/*.h5')
count = 0
data = []

n_records = 364

# 2018: 26 days mosaic images
x1,x2,y1,y2 = 1750,1900,800,1050
for file in files:
#    _data = read_vnp(file,4)
    _data = read_vnp(file,-1)
#    _data = read_vnp(file,11)
#    _data = read_vnp(file,12)
    data.append(_data[1750:1900,800:1050])
    count += 1
    if count > n_records:
        break
    
#data = np.array(data)
        
day = 0
days = []

for i in range(n_records):
#    im = strimg255(data[i])
#    im = strimg255(data[i][1750:1900,850:1100])
#    im = data[i][1750:1900,850:1100]
    im = data[i]
#     [1400:2000,500:1200] # Greater Bay Area
    # [1750:1900,850:1100] # hk
    plt.imshow(im)
    if (im.max()-im.min())>0.3:
        day += 1
        days.append(i)
#    plt.title(str(im.min())+',  '+str(im.max()))
#    plt.show()
    
#%% concat daily images
data = np.array(data)
data = np.transpose(data,[1,2,0])    
plt.imshow(data[:,:,2])    

# set data geo prj
geo = (110.0, 10/2400, 0.0, 30.0, 0.0, -10/2400)
prj = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
newgeo = setGeo(geo,y1,x1,x_offset=0)
    
#name = 'viirs_daily_y18c'
#name = 'viirs_daily_y18_cloud'
#name = 'viirs_daily_y18_quality'
name = 'viirs_daily_y18_time'
#save_tif(data,name,newgeo,prj,dtype='int')
#save_tif(data,name,newgeo,prj,dtype='float')
