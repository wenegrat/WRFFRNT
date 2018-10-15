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
from scipy import ndimage
from cmocean import cm as cmo


plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
#%%
#inputs

savefig = True
figfile = 'v_zeta_filt_x2km_y2km_hov.png'
zplot = 10

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
f = 1e-4
for n,hr in enumerate(hrs):
    fpn = dirall + wrfout[n] + hr + '_' + mins[n] + '_00'
    print(fpn)
    data = Dataset(fpn,mode='r')

    #get variables from netcdf
    print('getting data')
    nt = nt + data.dimensions['Time'].size

    nx = data.dimensions['west_east'].size
    ny = data.dimensions['south_north'].size
    nxs = data.dimensions['west_east_stag'].size
    nys = data.dimensions['south_north_stag'].size
    dx = getattr(data,'DX')
    dy = getattr(data,'DY')
    x = np.linspace(dx/2,nx*dx-dx/2,nx)/1000
    y = np.linspace(dy/2,ny*dy-dy/2,ny)/1000
    xs = np.linspace(0,nx*dx,nxs)/1000
    ys = np.linspace(0,ny*dy,nys)/1000

    if n==0:
        ph = data.variables['PH'][:,:,0,0]
        phb = data.variables['PHB'][:,:,0,0]
        z = (ph[0,0:-1] + ph[0,1:] + phb[0,0:-1] + phb[0,1:]) / 2 / 9.81
        k = np.argwhere(z>=zplot)[0][0]
        print(k,z[k])

    # u10 = data.variables['U10'][:]
    # v10 = data.variables['V10'][:]
    uk = data.variables['U'][:,k,:,:]
    vk = data.variables['V'][:,k,:,:]
    fsizex = 25 #250
    fsizey = 25 #150
    vk_filt = ndimage.uniform_filter(vk,mode='wrap',size=(1,fsizey,fsizex))
    uk_filt = ndimage.uniform_filter(uk,mode='wrap',size=(1,fsizey,fsizex))
    ukc = 0.5 * (uk[:,:,1:] + uk[:,:,:-1])
    vkc = 0.5 * (vk[:,1:,:] + vk[:,:-1,:])
    dudy = (uk[:,1:,1:-1] - uk[:,:-1,1:-1]) / dy
    dvdx = (vk_filt[:,1:-1,1:] - vk_filt[:,1:-1,:-1]) / dx
    zeta =  dvdx #- dudy
    dudx = (uk_filt[:,:,1:] - uk_filt[:,:,:-1])/dx
    #along-front avg
    if n==0:
        temp = np.mean(data.variables['T'][:,k,:,:], axis=1)
        uplot = np.mean(ukc,axis=1)
        vplot = np.mean(vkc,axis=1)
        zetaplot = np.mean(zeta,axis=1)
        divplot = np.mean(dudx, axis=1)
    else:
        uplot = np.concatenate((uplot,np.mean(ukc,axis=1)))
        vplot = np.concatenate((vplot,np.mean(vkc,axis=1)))
        zetaplot = np.concatenate((zetaplot,np.mean(zeta,axis=1)))
        divplot = np.concatenate((divplot, np.mean(dudx, axis=1)))
        temp = np.concatenate((temp, np.mean(data.variables['T'][:,k,:,:], axis=1)))
#plot
data.close()
#%%
plt.rcParams['contour.negative_linestyle'] = 'solid'
tStart = dt.datetime(2000,1,1,00,00,00)
t = np.array([tStart + dt.timedelta(minutes=n*10) for n in range(nt)])
# tout = 10
# t = np.arange(0,nt*tout,tout)/60.0 #hours
# t = t / 17.5
cmv = cmo.balance
cm = cmo.curl
conts = np.linspace(-10, 10, 30)
contsu = np.linspace(-1, 1, 30)
contsv = np.linspace(0, 5, 30)

nc = np.linspace(-0.03, 0.03, 5)

L = 10.0
plt.figure(figsize=(16,12))


plt.subplot(2,2,1)
im = plt.contourf(x,t,uplot,contsu, cmap=cmv, extend='both')
for a in im.collections:
        a.set_edgecolor('face')
cb = plt.colorbar(ticks=[contsu[0], 0, contsu[-1]])
cb.ax.tick_params(labelsize=16)
cb.solids.set_edgecolor("face")

#plt.clim(-1, 1)
plt.contour(x, t, temp, nc, colors='k')
plt.plot([L/4,L/4],[t[0],t[-1]],'w--',linewidth=1)
plt.plot([L/4-0.25,L/4+0.25],[t[-1],t[-1]],'w--',linewidth=10)
#plt.plot([L/4-0.25,L/4+0.25],[t[0]+dt.timedelta(minutes=-30),t[0]+dt.timedelta(minutes=-30)],'w--',linewidth=10)
#plt.annotate('', xy=(0,-0.1), xytext = (1, -0.1), xycoords='axes fraction',
#              arrowprops=dict(arrowstyle="<->", color='b'))
plt.plot([3*L/4,3*L/4],[t[0],t[-1]],'w--',linewidth=1)
plt.plot([3*L/4-0.25,3*L/4+0.25],[t[-1],t[-1]],'w--',linewidth=10)
plt.title('$U_{10}\ \mathrm{[m\ s^{-1}]}$',fontsize=24)
plt.xlabel('$x\ \mathrm{[km]}$',fontsize=24)
# plt.ylabel('$t/T$',fontsize=24)
plt.ylabel(r'$\mathrm{Hour}$',fontsize=24)
plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H'))
plt.tick_params(labelsize=16)

plt.subplot(2,2,2)
im = plt.contourf(x,t,vplot,contsv, cmap=cmv, extend='both')
for a in im.collections:
        a.set_edgecolor('face')
cb = plt.colorbar(ticks=[contsv[0], 2.5, contsv[-1]])
cb.ax.tick_params(labelsize=16)
cb.solids.set_edgecolor("face")

#plt.clim(0, 5)
plt.contour(x, t, temp, nc, colors='k')
plt.plot([L/4,L/4],[t[0],t[-1]],'w--',linewidth=1)
plt.plot([L/4-0.25,L/4+0.25],[t[-1],t[-1]],'w--',linewidth=10)
#plt.plot([L/4-0.25,L/4+0.25],[t[0]+dt.timedelta(minutes=-30),t[0]+dt.timedelta(minutes=-30)],'w--',linewidth=10)
#plt.annotate('', xy=(0,-0.1), xytext = (1, -0.1), xycoords='axes fraction',
#              arrowprops=dict(arrowstyle="<->", color='b'))
plt.plot([3*L/4,3*L/4],[t[0],t[-1]],'w--',linewidth=1)
plt.plot([3*L/4-0.25,3*L/4+0.25],[t[-1],t[-1]],'w--',linewidth=10)
plt.title('$V_{10}\ \mathrm{[m\ s^{-1}]}$',fontsize=24)
plt.xlabel('$x\ \mathrm{[km]}$',fontsize=24)
# plt.ylabel('$t/T$',fontsize=24)
plt.ylabel(r'$\mathrm{Hour}$',fontsize=24)
plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H'))
plt.tick_params(labelsize=16)

plt.subplot(2,2,3)
im = plt.contourf(x,t,divplot/1e-4,conts, cmap=cm, extend='both')
for a in im.collections:
        a.set_edgecolor('face')
cb = plt.colorbar(ticks=[conts[0], 0, conts[-1]])
cb.ax.tick_params(labelsize=16)


plt.clim(-10, 10)
cb.solids.set_edgecolor("face")
plt.contour(x, t, temp, nc, colors='k')
plt.plot([L/4,L/4],[t[0],t[-1]],'w--',linewidth=1)
plt.plot([L/4-0.25,L/4+0.25],[t[-1],t[-1]],'w--',linewidth=10)
#plt.plot([L/4-0.25,L/4+0.25],[t[0]+dt.timedelta(minutes=-30),t[0]+dt.timedelta(minutes=-30)],'w--',linewidth=10)
#plt.annotate('', xy=(0,-0.1), xytext = (1, -0.1), xycoords='axes fraction',
#              arrowprops=dict(arrowstyle="<->", color='b'))
plt.plot([3*L/4,3*L/4],[t[0],t[-1]],'w--',linewidth=1)
plt.plot([3*L/4-0.25,3*L/4+0.25],[t[-1],t[-1]],'w--',linewidth=10)
plt.title('$(\\nabla_h \\cdot {\widetilde{U}_{10}})/f$',fontsize=24)
plt.xlabel('$x\ \mathrm{[km]}$',fontsize=24)
# plt.ylabel('$t/T$',fontsize=24)
plt.ylabel(r'$\mathrm{Hour}$',fontsize=24)
plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H'))
plt.tick_params(labelsize=16)

plt.subplot(2,2,4)
im = plt.contourf(xs[1:-1],t,zetaplot/1e-4,conts, cmap=cm, extend='both')
for a in im.collections:
        a.set_edgecolor('face')
cb = plt.colorbar(ticks=[conts[0], 0, conts[-1]])
cb.ax.tick_params(labelsize=16)

plt.clim(-10, 10)
cb.solids.set_edgecolor("face")
plt.contour(x, t, temp, nc, colors='k')

#cb.ax.set_yticklabels(['$-6f$','$-4f$','$-2f$','$0$','$2f$','$4f$','$6f$'],fontsize=24)
plt.plot([L/4,L/4],[t[0],t[-1]],'w--',linewidth=1)
plt.plot([L/4-0.25,L/4+0.25],[t[-1],t[-1]],'w--',linewidth=10)
#plt.plot([L/4-0.25,L/4+0.25],[t[0]+dt.timedelta(minutes=-30),t[0]+dt.timedelta(minutes=-30)],'w--',linewidth=10)
#plt.annotate('', xy=(0,-0.1), xytext = (1, -0.1), xycoords='axes fraction',
#              arrowprops=dict(arrowstyle="<->", color='b'))
plt.plot([3*L/4,3*L/4],[t[0],t[-1]],'w--',linewidth=1)
plt.plot([3*L/4-0.25,3*L/4+0.25],[t[-1],t[-1]],'w--',linewidth=10)
# plt.title('$\zeta_{10}=\partial V_{10}/\partial x-\partial U_{10}/\partial y\ \mathrm{[s^{-1}]}$',fontsize=24)
plt.title('$\\widetilde{\zeta}_{10}/f }$',fontsize=24)
plt.xlabel('$x\ \mathrm{[km]}$',fontsize=24)
#plt.gca().set_yticklabels('')
plt.ylabel(r'$\mathrm{Hour}$',fontsize=24)
plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H'))
plt.tick_params(labelsize=16)
plt.tight_layout()
if savefig:
    plt.savefig(figfile,dpi=100)
else:
    plt.show()

#plt.savefig('/home/jacob/Dropbox/wrf_fronts/ATMOSMS/working directory/UpdatedHov.pdf', bbox_inches='tight')
#close data

