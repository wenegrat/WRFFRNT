######################################################
#
# Script to view preliminary results
# RSA 6-9-17
#
#####################################################
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fnt
import matplotlib.dates as mdates
import datetime as dt

#inputs
savefig = True
figfile = 'sst_along.png'

dirall = '/p/lscratchh/arthur7/front/forced_d188.7_dz5_2Tispinup_isfflx_2'
wrfout = '/wrfout_d01_0001-01-02_11:00:00'

fpn = dirall + wrfout
print fpn
data = Dataset(fpn,mode='r')

#get variables from netcdf
print 'getting data'

nx = data.dimensions['west_east'].size
ny = data.dimensions['south_north'].size
dx = getattr(data,'DX')/1000
dy = getattr(data,'DY')/1000
x = np.linspace(dx/2,nx*dx-dx/2,nx)
y = np.linspace(dy/2,ny*dy-dy/2,ny)

sst = data.variables['TSK'][-1,:,:]
sst_new = np.zeros((ny+1,nx+1))
print sst_new.shape, sst.shape
sst_new[:-1,:-1] = sst
sst_new[-1,:] = sst_new[-2,:]
sst_new[:,-1] = sst_new[:,-2]
xnew = np.append(x,x[-1]+dx)
ynew = np.append(y,y[-1]+dy)

#plot
plt.figure(figsize=(15,8))
plt.pcolormesh(xnew-dx/2,ynew-dx/2,sst_new,cmap='RdYlBu_r')
cax = plt.colorbar(ticks=np.arange(299.92,300.16,0.08))
# cax.set_label(label='$SST\ \mathrm{[K]}$',fontsize=24)
plt.clim(299.92,300.08)
cax.ax.set_yticklabels(['$SST_0-\Delta SST/2$','$SST_0$','$SST_0+\Delta SST/2$'],fontsize=24)
plt.xlabel('$x\ \mathrm{[km]}$',fontsize=24)
plt.ylabel('$y\ \mathrm{[km]}$',fontsize=24)
plt.xlim(0,10)
plt.ylim(0,3)
plt.tick_params(labelsize=16)

if savefig:
    plt.savefig(figfile,dpi=100)
else:
    plt.show()

#close data
data.close()
