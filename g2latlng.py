# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 07:10:55 2020

@author: sj
"""

import numpy as np

# left corner as (0,0), US in West and Asia in East
# only validated in Asia, North and East
#x as latitude (180), y as longitude (360),  
# x = 114288 # latitude, filepath
# y = 214078 # longitude, filename
# z = zoom level
# x,y,z = 114288,214078,18
def g2latlng(x,y,z=18,vbs=0):
    x,y = x+0.5,y+0.5 # to center
    n = np.power(2,z)
    lng = y / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * x / n)))
    lat = lat_rad * 180.0 / np.pi
    if vbs:
        print(x,lat,y,lng)
    return lat,lng

def to60(lat,lng,vbs=0):
    lat0 = np.floor(lat)
    tmp = (lat - lat0) * 60
    lat1 = np.floor(tmp)
    lat2 = (tmp - lat1) * 60
    lat = int(lat0),int(lat1),lat2
        
    lng0 = np.floor(lng)
    tmp = (lng - lng0) * 60
    lng1 = np.floor(tmp)
    lng2 = (tmp - lng1) * 60
    lng = int(lng0),int(lng1),lng2
    if vbs:
        print(lat,lng)
    return lat,lng

if __name__ == '__main__':
    if True:
        x,y,z = 114453,213769,18
        lat,lng = g2latlng(x,y,z,vbs=1)
        to60(lat,lng,vbs=1)
        
    
    
    