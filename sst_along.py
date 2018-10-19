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
savefig = False
figfile = 'sst_along.png'

dirall = '/media/ExtDriveFolder/WRFRUNS/WeakRun1'
wrfout = '/wrfout_d01_0001-01-02_11_00_00'

fpn = dirall + wrfout
print(fpn)
data = Dataset(fpn,mode='r')

#get variables from netcdf
print('getting data')

nx = data.dimensions['west_east'].size
ny = data.dimensions['south_north'].size
dx = getattr(data,'DX')/1000
dy = getattr(data,'DY')/1000
x = np.linspace(dx/2,nx*dx-dx/2,nx)
y = np.linspace(dy/2,ny*dy-dy/2,ny)

sst = data.variables['TSK'][-1,:,:]
sst_new = np.zeros((ny+1,nx+1))
print(sst_new.shape, sst.shape)
sst_new[:-1,:-1] = sst
sst_new[-1,:] = sst_new[-2,:]
sst_new[:,-1] = sst_new[:,-2]
xnew = np.append(x,x[-1]+dx)
ynew = np.append(y,y[-1]+dy)

#plot
#%%
fs = 20
plt.figure(figsize=(12,5))
pc = plt.pcolormesh(xnew-dx/2,ynew-dx/2,sst_new,cmap='RdYlBu_r')
pc.set_edgecolor('face')
cax = plt.colorbar(ticks=np.arange(299.92,300.16,0.08))
cax.solids.set_edgecolor('face')
# cax.set_label(label='$SST\ \mathrm{[K]}$',fontsize=24)
plt.clim(299.92,300.08)
cax.ax.set_yticklabels(['$SST_0-\Delta SST/2$','$SST_0$','$SST_0+\Delta SST/2$'],fontsize=fs)
plt.xlabel('$x$',fontsize=fs)
plt.ylabel('$y$',fontsize=fs)
#plt.axis('equal')
plt.gca().set_xlim(0,10)
plt.gca().set_ylim(0,3)
plt.xticks([0, 5, 10])
plt.gca().set_xticklabels(['$-L^x/2$', '0', '$L^x/2$'])
plt.yticks([0, 3])
plt.gca().set_yticklabels(['0', '$L^y$'])
plt.tick_params(labelsize=fs)
#plt.gca().annotate(' ', xy=(-2.25,-1), xytext = (-2.75, -1),xycoords='data',
#              arrowprops=dict(arrowstyle="<->", color='b'))
#plt.annotate('', xy=(2.25/10,-0.05), xytext = (2.75/10, -0.05),xycoords='axes fraction',
#              arrowprops=dict(arrowstyle="|-|", color='b'))
plt.annotate('$L_{SST}$', xy=(2.5/10,-0.05), xytext = (2.5/10, -0.11),xycoords='axes fraction',
            fontsize=fs, ha='center', va='top',
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=0.7, lengthB=0.5', lw=2.0))
if savefig:
    plt.savefig(figfile,dpi=100)
else:
    plt.show()

plt.tight_layout()
#close data
#data.close()
#plt.savefig('/home/jacob/Dropbox/wrf_fronts/ATMOSMS/Supporting Information/sst_plan.pdf', bbox_inches='tight')

