# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:03:41 2020

@author: sj

read raw NEF image, save as tif

"""

import numpy as np
import gdal
import rawpy
from PIL import Image

afile = 'raw/iss042e152950.NEF'

def raw2tif(afile,bfile=0):
    raw = rawpy.imread(afile)
    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    im1x,im1y,imz = im.shape
    
    if bfile==0:
        bfile = afile[:-4]+'_16bit.tif'
    else:
        bfile = bfile+'.tif'
    
    outdata = gdal.GetDriverByName('GTiff').Create(bfile, im1y, im1x, 3, gdal.GDT_UInt16)
    outdata.GetRasterBand(1).WriteArray(im[:,:,0])
    outdata.GetRasterBand(2).WriteArray(im[:,:,1])
    outdata.GetRasterBand(3).WriteArray(im[:,:,2])
    outdata.FlushCache() ##saves to disk!!
    print('Output:'+bfile+'.tif')
    outdata = None

if True:
    raw2tif(afile)