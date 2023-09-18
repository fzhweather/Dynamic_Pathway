#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 10:07:07 2023

@author: fuzhenghang
"""
# In[0]
import CRU_corr as CC

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
from scipy.stats import pearsonr


# In[1]
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 6
mpl.rcParams["axes.linewidth"] = 0.8
plt.rcParams['ytick.direction'] = 'out'


d1 = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/temp_geo_5monthly9_1979_2022.nc')
lon = d1.variables['longitude'][:]
lat = d1.variables['latitude'][:]

proj = ccrs.PlateCarree()  #中国为左
fig = plt.figure(figsize=(10,4),dpi=600)
ax=[]
x1 = [0,0,0.3,0.49,0.49,0.7,0.7,0.7]
yy = [0.77,0.42,0.42,-0.068,0.67,0.35,0.04,0.67,0.35,0.04]
dx = [0.45,0.28,0.15,0.19]
dy = [0.6,0.36,0.36,0.36]
loc = [[0.46, 0.87, 0.0085, 0.4],[0.46, 0.24, 0.0085, 0.22],[0.535, 0, 0.18, 0.012],[0.742, 0, 0.18, 0.012]]

for i in range(3):
    if i == 0:
        ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]],projection = proj))
    else:
        ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]]))


for i in range(2):
    if i not in [1,3]:
        if i == 0:
            leftlon, rightlon, lowerlat, upperlat = (60,140,15,45)
            data = CC.corr
            datap = CC.corrp
        
        
        lon_formatter = cticker.LongitudeFormatter()
        lat_formatter = cticker.LatitudeFormatter()
        ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
        ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k',zorder=2)
        #ax[i].add_feature(cfeature.OCEAN.with_scale('50m'),zorder=1,color='w',lw=0)
        
        gl=ax[i].gridlines(draw_labels=True, linewidth=0.35, color='k', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(np.arange(60,181,20))
        gl.ylocator = mticker.FixedLocator(np.arange(10,80,10))
        gl.top_labels    = False    
        gl.right_labels  = False
        gl.ypadding=2
        gl.xpadding=2
        if i == 0:
            shp_path1 = r'/Users/fuzhenghang/Documents/python/Pakistan/tibetan/tibetan.shp'
            reader = Reader(shp_path1)
            tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none',zorder=2)
            ax[i].add_feature(tpfeat, linestyle='--',linewidth=0.8)
            cb=ax[i].contourf(CC.cycle_lon,lat,data, levels = np.arange(-0.7,0.7001,0.1),cmap = cmaps.hotcold_18lev ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
            poly1 = plt.Polygon(CC.xy1,edgecolor='r',linestyle='--',fc="none", lw=2, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
            poly2 = plt.Polygon(CC.xy2,edgecolor='b',linestyle='--',fc="none", lw=2, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
            ax[i].add_patch(poly1)
            ax[i].add_patch(poly2)
            for l1 in range(46,70,2):
                for l2 in range(60,140,2):
                    if datap[l1,l2]<0.05 and data[l1,l2]<0:
                        ax[i].text(l2,90-l1,'.',fontsize=15,fontweight = 'bold',color = 'navy')
                    if datap[l1,l2]<0.05 and data[l1,l2]>0:
                        ax[i].text(l2,90-l1,'.',fontsize=15,fontweight = 'bold',color = 'darkred')
            position=fig.add_axes(loc[0])
            cbar=plt.colorbar(cb,cax=position,orientation='vertical',ticks=np.arange(-0.6,0.6001,0.3),
                             aspect=20,shrink=0.2,pad=0.06)
            cbar.set_label('Corr.',labelpad=1)
            ax[i].text(56,46,'A',fontsize=10,fontweight='bold')
        
                
    else:
        ax[0].text(56,10,'B',fontsize=10,fontweight='bold')
        x = [1979+i for i in range(44)]
        ax[i].set_xlim(1978.5,2022.5)
        ax[i].set_ylim(-3,4.2)
        ax[i].set_ylabel('Normalized Value',labelpad=1)
        ax[i].set_yticks([-2,0,2,4])
        ax[i].tick_params(length=2,width=0.4,pad=1.5)
        ax[i].axhline(y=0,  linestyle='-',linewidth = 0.35,color='black',alpha=1,zorder=0)
        if i == 1:
            ax[i].bar(x,CC.pren,color='lightskyblue',lw=1.2,label='Precipitation')
            ax[i].plot(x,CC.hwn,'-o',color='tomato',lw=1.2,ms=3,label='Heatwave')
            ax[i].axhline(y=1,  linestyle='dashed',linewidth = 0.3,color='black',alpha=1,zorder=0)
            ax[i].axhline(y=-1,  linestyle='dashed',linewidth = 0.3,color='black',alpha=1,zorder=0)
            ax[i].axvline(x=2013,  linestyle='-',linewidth = 1.5,color='lightgray',alpha=0.2,zorder=0)
            const1,p1 = pearsonr(CC.pren, CC.hwn)
            ax[i].text(1980,3.3,'correlation = %.2f**'%const1,color='k',fontsize=6)
            ax[i].legend(frameon=False,loc='lower left',ncol=3,fontsize=6)
            ax[i].text(2013,-2.7,'2013',ha='center',fontsize=6,fontweight = 'bold')
            #ax[i].text(2012.5,2.5,'Year 2022: 3.99→',fontsize=9,c='r')
            ax[i].annotate('Year 2022: 3.99', xy=(2022, 3.99), xytext=(2010,2.99), arrowprops=dict(arrowstyle="->", color="r", hatch='*',),fontsize=6,fontweight = 'bold')

tit=['A      ','B      ','C     ','d      ']
xl = ['Niño-3.4 index','Niño-3.4 index','Niño-3.4 index','Niño-3.4 index']
yl = ['YRV-hwatwave, day/year','PNWI-precipitation, mm/month','YRV-hwatwave','PNWI-pre.']
x = [CC.index[78:],CC.index[78:],CC.index[78:],CC.index[78:]]
y = [CC.hw,CC.pre,CC.hw,CC.pre]
co = ['salmon','royalblue','salmon','royalblue']
coo = ['tomato','royalblue','red','blue']
cor = ['r','b','r','b']
words = ['r = ','r = ','r = -0.28','r = -0.32*']
const1,p1 = pearsonr(CC.index[78:], CC.pre)
const2,p2 = pearsonr(CC.index[78:], CC.hw)
print(const1,const2)
print(p1,p2)
xloc = [0.7,-1.6,0.5,0.5]
xt = [-2.9,-2.9,-2.4,-2.5]
yt = [9.45,2.1,18,91]
ytt = [17,35]
from sklearn.linear_model import LinearRegression
for i in range(2,3):
    ax[i].yaxis.tick_right()
    ax[i].yaxis.set_label_position("right")
    ax[i].text(xt[i],yt[i],tit[i],fontweight='bold',fontsize=10)
    ax[i].set_ylim(-17,17)
    ax[i].text(xloc[i],yt[i]/1.5,words[i],fontsize=7)
    ax[i].grid('--',linewidth=0.3,zorder=1)
    ax[i].scatter(x[i],y[i],s=12,color=co[i],alpha=0.66,linewidth=0,zorder=2)
    ax[i].scatter(x[i][-1],y[i][-1],s=18,color=coo[i],alpha=0.66,linewidth=0,zorder=2)
    ax[i].scatter(x[i][-10],y[i][-10],s=18,color=coo[i],alpha=0.66,linewidth=0,zorder=2)
    regressor = LinearRegression()
    regressor = regressor.fit(np.reshape(x[i],(-1, 1)),np.reshape(y[i],(-1, 1)))
    print(regressor.coef_)
    ax[i].plot(np.reshape(x[i],(-1,1)), regressor.predict(np.reshape(x[i],(-1,1))),color=cor[i],linewidth=1.5)
    ax[i].set_xlabel(xl[i])
    ax[i].set_ylabel(yl[i])
    ax[i].set_xlim(-2,2)
    ax[i].text(-0.3,ytt[i-2]*0.43,'2013',color='k',fontweight='bold',fontsize=6)
    ax[i].text(-1,yt[i]*0.76,'2022',color='k',fontweight='bold',fontsize=6)


#plt.savefig('/Users/fuzhenghang/Documents/python/Pakistan/figs/fig1.png', format="png", bbox_inches = 'tight',pad_inches=0,dpi=1000)

plt.savefig('/Users/fuzhenghang/Documents/python/Pakistan/figs/fig1.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)

        