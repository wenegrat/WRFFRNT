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
import scipy.integrate as integrate
from cmocean import cm as cmo
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
#inputs
savefig = False

#%%
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
nk = 100




for n,hr in enumerate(hrs):
    fpn = dirall + wrfout[n] + hr + '_' + mins[n] + '_00'
    print(fpn)
    data = Dataset(fpn,mode='r')

    #get variables from netcdf
    print('getting data')
    nt = nt + data.dimensions['Time'].size
    

    ph = data.variables['PH'][:,:,:,:]
    phb = data.variables['PHB'][:,:,:,:]

    z = (ph[:,0:-1,1,1] + ph[:,1:,1,1] + phb[:,0:-1,1,1] + phb[:,1:,1,1]) / 2 / 9.81
    zs = (ph[:,:,1,1] + phb[:,:,1,1]) / 9.81
    z = np.expand_dims(np.expand_dims(z,axis=2), 3) #so broadcasting works
    
    #budget variables
    if n==0:
        u = data.variables['U'][:,:,:,:]
        ua = np.mean(u, axis=2)
        slp = data.variables['PSFC'][:,:,:]
        P = (data.variables['P'][:,:,:,:] + data.variables['PB'][:,:,:,:])/9.8
#        phd = data.variables['PH'][:,0,:,:] + data.variables['PHB'][:,0,:,:]
        sft = data.variables['T'][:,0,:,:]
        t = data.variables['T'][:,:,:,:]
        tmp = (data.variables['T'][:,1:,:,:] - data.variables['T'][:,:-1,:,:])/(z[:,1:,:,:] - z[:,:-1,:,:])
        tz = np.mean(tmp, axis=2)
    else:
        u = data.variables['U'][:,:,:,:]
        ua = np.concatenate((ua, np.mean(u, axis=2)))
        slp = np.concatenate((slp,data.variables['PSFC'][:,:,:]))
#        phd = np.concatenate((phd, data.variables['PH'][:,0,:,:] + data.variables['PHB'][:,0,:,:]))
        tmp = (data.variables['T'][:,1:,:,:] - data.variables['T'][:,:-1,:,:])/(z[:,1:,:,:] - z[:,:-1,:,:])
        tmp2 = np.mean(tmp, axis=2)
        sft = np.concatenate((sft,data.variables['T'][:,0,:,:]))
        tz = np.concatenate((tz, tmp2))
        t = np.concatenate((t, data.variables['T'][:,:,:,:]))
        P = np.concatenate((P, (data.variables['P'][:,:,:,:] + data.variables['PB'][:,:,:,:])/9.8))
slp_avg = np.mean(slp, axis=1)
sft_avg = np.mean(sft, axis=1)
#phd_avg = np.mean(phd, axis=1)
#%%
t_avg = np.mean(t, axis=2)

t_int = integrate.cumtrapz(t_avg, x=np.squeeze(z), axis=1)
t_avg = t_int/z[:,1:,0,:]

#%% CALCULATE INVISCID SEA-BREEZE PRESSURE ACCEL
R = 287
p0 = 1000
p1 = 900 # need to check these values (first look at overturning streamfunction)
h = 1000
L = 5000
t2mt1 = 0.005 # look at t_avg over warm vs cold pool (need to pick a height)

presaccel = R*np.log(p0/p1)*t2mt1/(2*(h+L))
#%%
ti = 12
zi = 50
plt.figure()
#plt.plot(np.mean(slp[ti,:,:], axis=0)-1e5)
plt.plot(-t_int[ti, zi, :])

#%% Calculate overturning stream function.
def flip(m, axis):
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]

psi = flip(integrate.cumtrapz(flip(ua, axis=1), x=flip(np.squeeze(z), axis=0), axis=1), axis=1)
#%% MAKE SLICE PLOTS
pmean = np.mean(np.mean(P[ti, :,:,:], axis=2), axis=1)
pmean = pmean[:,np.newaxis]

#%%
plt.rcParams.update({'font.size': 22})
plt.rcParams['contour.negative_linestyle'] = 'solid'
tr = range(40, 70)
cm = cmo.thermal
#plt.figure()
#plt.subplot(1,2,1)
#im = plt.pcolor(np.linspace(0, 10, 500), z[0,:,0,0], np.mean(t_avg[tr,:,:], axis=0))
#plt.contour(np.linspace(0, 10, 500), z[0,:,0,0], np.mean(np.mean(P[tr,:,:,:], axis=0), axis=1), 4)
#plt.clim((-0.05, 0.05))
#plt.colorbar(im)
#plt.ylim((0, 1000))
#
#
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10,8))
ax =  axs.reshape(-1)
offset = 18

cl = 0.04
contst = np.linspace(-cl, cl, 40)
conts = np.linspace(-1395, -850, 30)
conts = np.linspace(-1401, -850, 25)
conts = np.linspace(-1401, -625, 15)
conts = np.linspace(-1401, -626, 15)
#conts = np.linspace(-1401.5, -641, 30)
conts = np.array(list(range(-1401, -600, 25)))-0.1
for i in range(0, 4):
    tr = range(i*offset, i*offset + offset)



    im = ax[i].contourf(np.linspace(0, 10, 500), z[0,1:,0,0], np.mean(t_avg[tr,:,:], axis=0),contst, cmap=cm, extend='both')
    for a in im.collections:
        a.set_edgecolor('face')
    ax[i].contour(np.linspace(0, 10, 501), z[0,1:,0,0], np.mean(psi[tr,:,:], axis=0), conts, colors='k', linewidths=0.75)
    #im = plt.pcolor(np.linspace(0, 10, 501), z[0,:,0,0], np.mean(flip(ua[tr,:,:], axis=1), axis=0))
    
    #plt.contour(np.linspace(0, 10, 500), z[0,:,0,0], np.mean(P[ti,:,:,:], 1), 10)
    im.set_clim((-cl, cl))
#    plt.colorbar(im)
    ax[i].set_ylim((0, 1200))
    ax[i].set_yticks([0, 400, 800, 1200])
    if (i % 2) == 0:
        ax[i].set_ylabel('z [m]')
    if i>1:
        ax[i].set_xlabel('x [km]')
#    ax[i].set_title('Hour: %i - %i' %((i*offset/6), (i*offset/6 + offset/6)))
    ax[i].text(7, 1100, 'Hour: %i - %i' %((i*offset/6), (i*offset/6 + offset/6)), bbox={'facecolor':'white'}, fontsize=12)
plt.subplots_adjust(wspace=.10, hspace=0.10)    
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
cb = fig.colorbar(im, cax=cbar_ax, ticks=[-cl, 0, cl], label='$^\circ$ K')
cb.ax.tick_params(labelsize=16)
cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=16)
cb.solids.set_edgecolor("face")
#plt.savefig('/home/jacob/Dropbox/wrf_fronts/ATMOSMS/working directory/T_Psi.pdf', bbox_inches='tight')

#plt.tight_layout()
#%%
ti
plt.figure()
plt.pcolor(np.linspace(0, 10, 500), z[0,:,0,0], t_int[ti,:,:])
plt.clim((0, 20))
plt.colorbar()
#%%

plt.figure()
plt.pcolor(np.linspace(0, 10, 500), z[0,:,0,0], np.mean(np.mean(t[12:18,:,:,:], axis=0), axis=1), vmin=-0.03, vmax=0.03)
plt.colorbar()
plt.ylim((0, 300))
#%%
plt.figure()
plt.plot(tz[10,:])
#%%
plt.figure()
plt.pcolor(np.linspace(0, 10, 500), z[0,:,0,0], np.mean(tz[6:12,:,:], axis=0), vmin=-5e-4, vmax=5e-4)
plt.colorbar()
plt.ylim((0, 300))

#%%
plt.figure()
plt.pcolor(slp_avg)
plt.colorbar()
#%%
tr = range(12, 18)
plt.figure()
plt.subplot(1,2,1)
plt.plot(np.mean(slp_avg[tr,:], axis=0))
#plt.plot(np.mean(phd_avg[0:10,:], axis=0))

plt.subplot(1,2,2)
plt.plot(np.mean(sft_avg[tr,:], axis=0))