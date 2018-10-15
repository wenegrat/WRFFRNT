######################################################
#
# Script to plot SLP
# over time
# JW 10-8-18
#
######################################################
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import scipy.stats as stats
from scipy import ndimage

#%%
#inputs
savefig = False

#dz=5
zplot = 50

#FORCED#####################################################################################
dirall = '/media/ExtDriveFolder/WRFRUNS'
wrfout = ['/wrfout_d01_0001-01-02_', \
          '/wrfout_d01_0001-01-02_', \
          '/wrfout_d01_0001-01-02_', \
          '/wrfout_d01_0001-01-02_', \
          '/wrfout_d01_0001-01-03_', \
          '/wrfout_d01_0001-01-03_']
hrs = ['11','14','18','21','01','04']
mins = ['00','30','00','30','00','30']

nt = 0


zplot = 10


for n,hr in enumerate(hrs):
    fpn = dirall + wrfout[n] + hr + '_' + mins[n] + '_00'
    print(fpn)
    data = Dataset(fpn,mode='r')

    #get variables from netcdf
    print('getting data')
    nt = nt + data.dimensions['Time'].size
    
    if n==0:
        ph = data.variables['PH'][:,:,0,0]
        phb = data.variables['PHB'][:,:,0,0]
        z = (ph[0,0:-1] + ph[0,1:] + phb[0,0:-1] + phb[0,1:]) / 2 / 9.81
        k = np.argwhere(z>=zplot)[0][0]
        print(k,z[k])  

    ph = data.variables['PH'][:,:,:,:]
    phb = data.variables['PHB'][:,:,:,:]

    z = (ph[:,0:-1,1,1] + ph[:,1:,1,1] + phb[:,0:-1,1,1] + phb[:,1:,1,1]) / 2 / 9.81
    zs = (ph[:,:,1,1] + phb[:,:,1,1]) / 9.81
    z = np.expand_dims(np.expand_dims(z,axis=2), 3) #so broadcasting works
    
    #budget variables
    if n==0:
        U = data.variables['U'][:,k,:,:]
        V = data.variables['V'][:,k,:,:]
        sst = data.variables['TSK'][:,:,:]
        tau = data.variables[']
#        phd = data.variables['PH'][:,0,:,:] + data.variables['PHB'][:,0,:,:]
    else:
        U = np.concatenate((U,data.variables['U'][:,k,:,:]))
        V = np.concatenate((V, data.variables['V'][:,k,:,:]))
        sst = np.concatenate((sst, data.variables['TSK'][:,:,:]))
#        phd = np.concatenate((phd, data.variables['PH'][:,0,:,:] + data.variables['PHB'][:,0,:,:]))

#phd_avg = np.mean(phd, axis=1)
#%%
Ue = 0.5*(U[:,:,1:]+U[:,:,:-1])
Ve = 0.5*(V[:,1:,:]+V[:,:-1,:])
fsizey = 10
fsizex= 10
Ve = ndimage.uniform_filter(Ve,mode='wrap',size=(1,fsizey,fsizex))

nt, ny, nx = Ue.shape

uvec = np.reshape(Ue, (nt, ny*nx))
vvec = np.reshape(Ve, (nt, ny*nx))
sstvec = np.reshape(sst, (nt, ny*nx))


mag = np.sqrt(uvec**2 + vvec**2)

maga = np.reshape(mag, nt*ny*nx)
va = np.reshape(Ve, nt*ny*nx)
ssta = np.reshape(sstvec, nt*ny*nx)
ssta = ssta-300
#%%
ti = 12

plt.figure()
plt.scatter(sstvec[ti,:], mag[ti,:])

#%%
ti = 100
plt.figure()
plt.hist2d(ssta, va, 20, vmin=0, vmax=1e4)
plt.colorbar()
slope, intercept, r_value, p_value, stderr  = stats.linregress(ssta, va)
plt.title('Slope: %f'%slope)