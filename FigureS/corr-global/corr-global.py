#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 09:28:02 2023

@author: fuzhenghang
"""


# In[0]
import spreadconvec_temp as SC
import mean_cesmconvec_temp as SSTC



import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr  
import numpy as np
import cmaps
from cartopy.io.shapereader import Reader

# In[1]

mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.linewidth"] = 0.8
plt.rcParams['ytick.direction'] = 'out'


d1 = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/temp_geo_5monthly9_1979_2022.nc')
lon = d1.variables['longitude'][:]
lat = d1.variables['latitude'][:]

proj = ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=(9,5),dpi=600)
ax=[]
x1 = [0.076,0.463,0,0,0]
yy = [0.72,0.72,0.35,0.00]
dx = [0.38,0.38,0.92,0.92,0.92]
dy = [0.36,0.36,0.3,0.3,0.3]
loc = [[0.84, 0.74, 0.012, 0.32],[0.84, 0.53, 0.012, 0.25],[0.84, 0.36, 0.012, 0.28],[0.84, 0.012, 0.012, 0.28],[0.335, 0.028, 0.0085, 0.32],[0.903, 0.028, 0.0085, 0.32]]
data = [SSTC.corr1,SC.corr1]
datap = [SSTC.corrp1,SC.corrp1]
level = [SSTC.levels,SC.levels]

lev =[np.arange(-2.7,2.71,0.3),np.arange(-1.8,1.8001,0.2)]
ti = [np.arange(-2.4,2.4001,1.2),np.arange(-1.6,1.6001,0.8)]
for i in range(2,4):
    ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]],projection = proj))
tit1 = ['A','B','c','d']
tit2 = ['Ensemble mean','Ensemble spread','H200: SST-forced mode','H200: Atmospheric internal mode']
cr = ['0.24','0.46**']
for i in range(2):
    leftlon, rightlon, lowerlat, upperlat = (1,359,-10,70)
    gl=ax[i].gridlines(draw_labels=True, linewidth=0.35, color='k', alpha=0.5, linestyle='--',zorder=8)
    gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
    gl.ylocator = mticker.FixedLocator(np.arange(-10,80,20))
    #ax[i].add_feature(cfeature.OCEAN.with_scale('50m'),zorder=1,color='w',lw=0)
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.3,color='k',zorder=2)
    ax[i].text(-188,72,tit1[i],fontsize=12,fontweight='bold')
    ax[i].text(-176,72,tit2[i],fontsize=10)
        
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    
    if i in [0]:
        gl.bottom_labels    = False 
    
    gl.top_labels    = False    
    gl.right_labels  = False
    gl.ypadding=2
    gl.xpadding=2


    shp_path1 = r'/Users/fuzhenghang/Documents/python/Pakistan/tibetan/tibetan.shp'
    reader = Reader(shp_path1)
    tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none',zorder=2)
    ax[i].add_feature(tpfeat, linestyle='--',linewidth=0.8)
    cb=ax[i].contourf(SSTC.cycle_lon,SSTC.lats,data[i], levels=level[i],cmap=cmaps.hotcold_18lev ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
    for l1 in range(80,160,5):
        for l2 in range(0,360,5):
            if datap[i][l1,l2]<0.05 and data[i][l1,l2]<0:
                ax[i].text(l2-180,l1-90,'.',fontsize=8,fontweight = 'bold',color = 'steelblue')
            if datap[i][l1,l2]<0.05 and data[i][l1,l2]>0:
                ax[i].text(l2-180,l1-90,'.',fontsize=8,fontweight = 'bold',color = 'darkred')   
    poly1 = plt.Polygon(SC.xy1,edgecolor='red',linestyle='--',fc="none", lw=1, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
    ax[i].add_patch(poly1)
    poly2 = plt.Polygon(SC.xy2,edgecolor='b',linestyle='--',fc="none", lw=1, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
    ax[i].add_patch(poly2)
    position1=fig.add_axes(loc[i+2])#位置[左,下,长度,宽度]
    cbar=plt.colorbar(cb,cax=position1,orientation='vertical',ticks=np.arange(-0.6,0.6001,0.3),
                     aspect=20,shrink=0.2,pad=0.06)
    cbar.ax.tick_params(length=1.8,width=0.4,pad=1.5)
    
    ax[i].text(181,-13,'gpm')
plt.savefig('/Users/fuzhenghang/Documents/python/Pakistan/figs/figs4.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)
  

    