######################################################
#
# Script to plot momentum and temperature budget terms
# over time
# RSA 11-13-17
#
######################################################
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

#inputs
savefig = False
outfig_u = 'budget_hov_u_200m_new.png'
outfig_w = 'budget_hov_w_200m.png'
outfig_T = 'budget_hov_T_200m.png'

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

    #budget variables
    if n==0:
        adv_u_u = data.variables['ADV_U_U'][:,k,:,:]
        adv_u_w = data.variables['ADV_U_W'][:,k,:,:]
        pres_u = data.variables['PRES_U'][:,k,:,:]
        cor_u = data.variables['COR_U'][:,k,:,:]
        hdiff_u = data.variables['HDIFF_U'][:,k,:,:]
        vdiff_u = data.variables['VDIFF_U'][:,k,:,:]
    else:
        adv_u_u = np.concatenate((adv_u_u,data.variables['ADV_U_U'][:,k,:,:]))
        adv_u_w = np.concatenate((adv_u_w,data.variables['ADV_U_W'][:,k,:,:]))
        pres_u = np.concatenate((pres_u,data.variables['PRES_U'][:,k,:,:]))
        cor_u = np.concatenate((cor_u,data.variables['COR_U'][:,k,:,:]))
        hdiff_u = np.concatenate((hdiff_u,data.variables['HDIFF_U'][:,k,:,:]))
        vdiff_u = np.concatenate((vdiff_u,data.variables['VDIFF_U'][:,k,:,:]))

    if n==0:
        tmp = data.variables['ADV_W_U'][:,k:k+2,:,:]
        adv_w_u = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
        tmp = data.variables['ADV_W_W'][:,k:k+2,:,:]
        adv_w_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
        tmp = data.variables['PRES_W'][:,k:k+2,:,:]
        pres_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
        tmp = data.variables['COR_W'][:,k:k+2,:,:]
        cor_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
        tmp = data.variables['HDIFF_W'][:,k:k+2,:,:]
        hdiff_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
        tmp = data.variables['VDIFF_W'][:,k:k+2,:,:]
        vdiff_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
    else:
        tmp = data.variables['ADV_W_U'][:,k:k+2,:,:]
        adv_w_u = np.concatenate((adv_w_u,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
        tmp = data.variables['ADV_W_W'][:,k:k+2,:,:]
        adv_w_w = np.concatenate((adv_w_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
        tmp = data.variables['PRES_W'][:,k:k+2,:,:]
        pres_w = np.concatenate((pres_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
        tmp = data.variables['COR_W'][:,k:k+2,:,:]
        cor_w = np.concatenate((cor_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
        tmp = data.variables['HDIFF_W'][:,k:k+2,:,:]
        hdiff_w = np.concatenate((hdiff_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
        tmp = data.variables['VDIFF_W'][:,k:k+2,:,:]
        vdiff_w = np.concatenate((vdiff_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))

    if n==0:
        adv_t_u = data.variables['ADV_T_U'][:,k,:,:]
        adv_t_w = data.variables['ADV_T_W'][:,k,:,:]
        hdiff_t = data.variables['HDIFF_T'][:,k,:,:]
        vdiff_t = data.variables['VDIFF_T'][:,k,:,:]
    else:
        adv_t_u = np.concatenate((adv_t_u,data.variables['ADV_T_U'][:,k,:,:]))
        adv_t_w = np.concatenate((adv_t_w,data.variables['ADV_T_W'][:,k,:,:]))
        hdiff_t = np.concatenate((hdiff_t,data.variables['HDIFF_T'][:,k,:,:]))
        vdiff_t = np.concatenate((vdiff_t,data.variables['VDIFF_T'][:,k,:,:]))

    #raw variables for turbulent calculations
    ph = data.variables['PH'][:,k:k+2,:,:]
    phb = data.variables['PHB'][:,k:k+2,:,:]
    u = data.variables['U'][:,k-1:k+2,:,:]
    w = data.variables['W'][:,k:k+2,:,:]
    T = data.variables['T'][:,k-1:k+2,:,:]
    fnm = data.variables['FNM'][:,k:k+2]
    fnm = np.expand_dims(np.expand_dims(fnm,axis=2),axis=3) #for broadcasting
    fnp = data.variables['FNP'][:,k:k+2]
    fnp = np.expand_dims(np.expand_dims(fnp,axis=2),axis=3) #for broadcasting

    #time tendency terms
    if n==0:
        u_allt = u[:,1,:,:]
        # dudt = u[:,1,:,:] / (10*60)
        # dwdt = 0.5 * (w[:,0,:,:] + w[:,1,:,:]) / (10*60)
        # dTdt = T[:,1,:,:] / (10*60)
    else:
        u_allt = np.concatenate((u_allt,u[:,1,:,:]))
        # dudt = np.concatenate((dudt,u[:,1,:,:]/(10*60)))
        # dudt = np.concatenate((dwdt,0.5*(w[:,0,:,:]+w[:,1,:,:])/(10*60)))
        # dTdt = np.concatenate((dTdt,T[:,1,:,:]/(10*60)))

    dx = getattr(data,'DX')
    nx = data.dimensions['west_east'].size
    nxs = data.dimensions['west_east_stag'].size
    x = np.linspace(dx/2,nx*dx-dx/2,nx)/1000
    xs = np.linspace(0,nx*dx,nxs)/1000

    nz = data.dimensions['bottom_top'].size
    nzs = data.dimensions['bottom_top_stag'].size
    z = (ph[:,0:-1,1,1] + ph[:,1:,1,1] + phb[:,0:-1,1,1] + phb[:,1:,1,1]) / 2 / 9.81
    zs = (ph[:,:,1,1] + phb[:,:,1,1]) / 9.81
    zs = np.expand_dims(zs,axis=1) #so broadcasting works

    ######################################################################################
    #calculate mean and turbulent stress divergences for non-cross terms (u,u) and (w,w)
    #also flux divergences for T

    #project T to u and w points
    Tu = 0.5*(T[:,:,:,1:] + T[:,:,:,:-1])
    Tw = fnm[:,1:] * T[:,1:,:,:] + fnp[:,1:] * T[:,:-1,:,:]

    #swap axes so that broadcasting works
    #index is now n,i,j,k instead of n,k,j,i
    uswap = np.swapaxes(u,1,3)
    wswap = np.swapaxes(w,1,3)
    Tswap = np.swapaxes(T,1,3)
    Tu = np.swapaxes(Tu,1,3)
    Tw = np.swapaxes(Tw,1,3)

    #calculate primes as departure from along-front average
    umean = np.mean(uswap,axis=2)
    wmean = np.mean(wswap,axis=2)
    Tmean = np.mean(Tswap,axis=2)
    Tumean = np.mean(Tu,axis=2)
    Twmean = np.mean(Tw,axis=2)

    uprime = uswap - np.expand_dims(umean,axis=2)
    wprime = wswap - np.expand_dims(wmean,axis=2)
    Tprime = Tswap - np.expand_dims(Tmean,axis=2)
    Tuprime = Tu - np.expand_dims(Tumean,axis=2)
    Twprime = Tw - np.expand_dims(Twmean,axis=2)

    #calculate u'u' and w'w' with along-front average
    upup = np.mean(uprime**2,axis=2)
    wpwp = np.mean(wprime**2,axis=2)

    #calculate stress divergences (all at cell centers)
    if n==0:
        ududx = 0.5*(umean[:,1:,1] + umean[:,:-1,1]) * (umean[:,1:,1] - umean[:,:-1,1]) / dx
        wdwdz = 0.5*(wmean[:,:,1:] + wmean[:,:,:-1]) * (wmean[:,:,1:] - wmean[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1])
        ddxupup = (upup[:,1:,1] - upup[:,:-1,1]) / dx
        ddzwpwp = (wpwp[:,:,1:] - wpwp[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1])
    else:
        ududx = np.concatenate((ududx,0.5*(umean[:,1:,1] + umean[:,:-1,1]) * (umean[:,1:,1] - umean[:,:-1,1]) / dx))
        wdwdz = np.concatenate((wdwdz,0.5*(wmean[:,:,1:] + wmean[:,:,:-1]) * (wmean[:,:,1:] - wmean[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1])))
        ddxupup = np.concatenate((ddxupup,(upup[:,1:,1] - upup[:,:-1,1]) / dx))
        ddzwpwp = np.concatenate((ddzwpwp,(wpwp[:,:,1:] - wpwp[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1])))

    #calculate u'T' and w'T' with along-front average
    upTp = np.mean(uprime[:,1:-1,:,:]*Tuprime, axis=2)
    wpTp = np.mean(wprime*Twprime, axis=2)

    #calculate flux divergences (all at cell centers)
    if n==0:
        udTdx = 0.5*(umean[:,2:-1,1] + umean[:,1:-2,1]) * (Tumean[:,1:,1] - Tumean[:,:-1,1]) / dx
        wdTdz = 0.5*(wmean[:,:,1:] + wmean[:,:,:-1]) * (Twmean[:,:,1:] - Twmean[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1])
        ddxupTp = (upTp[:,1:,1] - upTp[:,:-1,1]) / dx
        ddzwpTp = (wpTp[:,:,1:] - wpTp[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1])
    else:
        udTdx = np.concatenate((udTdx,0.5*(umean[:,2:-1,1] + umean[:,1:-2,1]) * (Tumean[:,1:,1] - Tumean[:,:-1,1]) / dx))
        wdTdz = np.concatenate((wdTdz,0.5*(wmean[:,:,1:] + wmean[:,:,:-1]) * (Twmean[:,:,1:] - Twmean[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1])))
        ddxupTp = np.concatenate((ddxupTp,(upTp[:,1:,1] - upTp[:,:-1,1]) / dx))
        ddzwpTp = np.concatenate((ddzwpTp,(wpTp[:,:,1:] - wpTp[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1])))

    ######################################################################################
    #calculate mean and turbulent stress divergences for cross terms (u,w)

    #project u and w to e points
    ue = fnm[:,1:] * u[:,1:,:,1:-1] + fnp[:,1:] * u[:,:-1,:,1:-1] #dont need i edge values
    we = 0.5 * (w[:,:,:,1:] + w[:,:,:,:-1])

    #swap axes so that broadcasting works
    #index is now n,i,j,k instead of n,k,j,i
    ue = np.swapaxes(ue,1,3)
    we = np.swapaxes(we,1,3)

    #calculate primes as departure from along-front average
    uemean = np.mean(ue,axis=2)
    wemean = np.mean(we,axis=2)
    ueprime = ue - np.expand_dims(uemean,axis=2)
    weprime = we - np.expand_dims(wemean,axis=2)

    #calculate u'w' with along-front average
    upwpe = np.mean(ueprime*weprime,axis=2)

    #calculate stress divergences
    if n==0:
        wdudz = 0.5*(wemean[:,:,1:] + wemean[:,:,:-1]) * (uemean[:,:,1:] - uemean[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1]) #at u points
        tmp = 0.5*(uemean[:,1:,:] + uemean[:,:-1,:]) * (wemean[:,1:,:] - wemean[:,:-1,:]) / dx #at w points
        udwdx = 0.5*(tmp[:,:,1:] + tmp[:,:,:-1]) #at cell centers
        ddzupwp = (upwpe[:,:,1:] - upwpe[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1]) #at u points
        tmp = (upwpe[:,1:,:] - upwpe[:,:-1,:]) / dx #at w points
        ddxupwp = 0.5*(tmp[:,:,1:] + tmp[:,:,:-1]) #at cell centers
    else:
        wdudz = np.concatenate((wdudz,0.5*(wemean[:,:,1:] + wemean[:,:,:-1]) * (uemean[:,:,1:] - uemean[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1]))) #at u points
        tmp = 0.5*(uemean[:,1:,:] + uemean[:,:-1,:]) * (wemean[:,1:,:] - wemean[:,:-1,:]) / dx #at w points
        udwdx = np.concatenate((udwdx,0.5*(tmp[:,:,1:] + tmp[:,:,:-1]))) #at cell centers
        ddzupwp = np.concatenate((ddzupwp,(upwpe[:,:,1:] - upwpe[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1]))) #at u points
        tmp = (upwpe[:,1:,:] - upwpe[:,:-1,:]) / dx #at w points
        ddxupwp = np.concatenate((ddxupwp,0.5*(tmp[:,:,1:] + tmp[:,:,:-1]))) #at cell centers

######################################################################################
#along-front average tendency terms
ax = 1

adv_u_u_avg = np.mean(adv_u_u,axis=ax)
adv_u_w_avg = np.mean(adv_u_w,axis=ax)
pres_u_avg = np.mean(pres_u,axis=ax)
cor_u_avg = np.mean(cor_u,axis=ax)
hdiff_u_avg = np.mean(hdiff_u,axis=ax)
vdiff_u_avg = np.mean(vdiff_u,axis=ax)

adv_w_u_avg = np.mean(adv_w_u,axis=ax)
adv_w_w_avg = np.mean(adv_w_w,axis=ax)
pres_w_avg = np.mean(pres_w,axis=ax)
cor_w_avg = np.mean(cor_w,axis=ax)
hdiff_w_avg = np.mean(hdiff_w,axis=ax)
vdiff_w_avg = np.mean(vdiff_w,axis=ax)

adv_t_u_avg = np.mean(adv_t_u,axis=ax)
adv_t_w_avg = np.mean(adv_t_w,axis=ax)
hdiff_t_avg = np.mean(hdiff_t,axis=ax)
vdiff_t_avg = np.mean(vdiff_t,axis=ax)

dudt_avg = np.mean((u_allt[2:,:,:]-u_allt[:-2,:,:])/(2*10*60),axis=ax)
# dudt_avg = np.mean(dudt,axis=ax)
# dwdt_avg = np.mean(dwdt,axis=ax)
# dTdt_avg = np.mean(dTdt,axis=ax)

#%%
######################################################################################
#plot
tStart = dt.datetime(2000,1,1,00,00,00)
t = np.array([tStart + dt.timedelta(minutes=n*10) for n in range(nt)])
L = 10.0

#TEST
plt.figure()
plt.plot(t[1:-1],-dudt_avg[:,125],'b',label='dudt')
plt.plot(t,cor_u_avg[:,125],'k',label='fv')
plt.legend()
plt.show()

#plot hovmoller for u tendency terms
plt.figure(figsize=(25,18))

xplot = [xs,x,xs[1:-1],x,xs[1:-1],xs,xs,xs,xs]
tplot = [t[1:-1],t,t,t,t,t,t,t,t]
tendplot = [dudt_avg,-ududx,-wdudz,-ddxupup,-ddzupwp,pres_u_avg,cor_u_avg,hdiff_u_avg,vdiff_u_avg]
tendmax = max(np.abs(np.amax(dudt_avg)), \
              np.abs(np.amax(-ududx)), \
              np.abs(np.amax(-wdudz)), \
              np.abs(np.amax(-ddxupup)), \
              np.abs(np.amax(-ddzupwp)), \
              np.abs(np.amax(pres_u_avg)), \
              np.abs(np.amax(cor_u_avg)), \
              np.abs(np.amax(hdiff_u_avg)), \
              np.abs(np.amax(vdiff_u_avg)))
titles = ['dudt','ududx','wdudz','ddxupup','ddzupwp','PRES_U','COR_U','HDIFF_U','VDIFF_U']
for n in range(9):
    plt.subplot(3,3,n+1)
    plt.pcolormesh(xplot[n],tplot[n],np.squeeze(tendplot[n]),cmap='RdBu_r')
    plt.colorbar()
    plt.clim(-0.1*tendmax,0.1*tendmax)
    plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if (n+1)>=7:
        plt.xlabel('x [km]')
    if n==0 or n==2 or n==4 or n==6:
        plt.ylabel('t')
    plt.title(titles[n])

if savefig:
    plt.savefig(outfig_u,dpi=100)

#plot hovmoller for w tendency terms
plt.figure(figsize=(18,18))

xplot = [x[1:-1],x,x[1:-1],x,x,x,x,x]
tendplot = [-udwdx,-wdwdz,-ddxupwp,-ddzwpwp,pres_w_avg,cor_w_avg,hdiff_w_avg,vdiff_w_avg]
tendmax = max(np.abs(np.amax(-udwdx)), \
              np.abs(np.amax(-wdwdz)), \
              np.abs(np.amax(-ddxupwp)), \
              np.abs(np.amax(-ddzwpwp)), \
              np.abs(np.amax(pres_w_avg)), \
              np.abs(np.amax(cor_w_avg)), \
              np.abs(np.amax(hdiff_w_avg)), \
              np.abs(np.amax(vdiff_w_avg)))
titles = ['udwdx','wdwdz','ddxupwp','ddzwpwp','PRES_W','COR_W','HDIFF_W','VDIFF_W']
for n in range(8):
    plt.subplot(4,2,n+1)
    plt.pcolormesh(xplot[n],t,np.squeeze(tendplot[n]),cmap='RdBu_r')
    plt.colorbar()
    plt.clim(-tendmax,tendmax)
    plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if (n+1)>=7:
        plt.xlabel('x [km]')
    if n==0 or n==2 or n==4 or n==6:
        plt.ylabel('t')
    plt.title(titles[n])

if savefig:
    plt.savefig(outfig_w,dpi=100)

#plot hovmoller for T tendency terms
plt.figure(figsize=(18,18))

xplot = [x[1:-1],x,x[1:-1],x,x,x]
tendplot = [-udTdx,-wdTdz,-ddxupTp,-ddzwpTp,hdiff_t_avg,vdiff_t_avg]
tendmax = max(np.abs(np.amax(-udTdx)), \
              np.abs(np.amax(-wdTdz)), \
              np.abs(np.amax(-ddxupTp)), \
              np.abs(np.amax(-ddzwpTp)), \
              np.abs(np.amax(hdiff_t_avg)), \
              np.abs(np.amax(vdiff_t_avg)))
titles = ['udTdx','wdTdz','ddxupTp','ddzwpTp','HDIFF_T','VDIFF_T']
for n in range(6):
    plt.subplot(3,2,n+1)
    plt.pcolormesh(xplot[n],t,np.squeeze(tendplot[n]),cmap='RdBu_r')
    plt.colorbar()
    plt.clim(-tendmax,tendmax)
    plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if (n+1)>=5:
        plt.xlabel('x [km]')
    if n==0 or n==2 or n==4:
        plt.ylabel('t')
    plt.title(titles[n])

if savefig:
    plt.savefig(outfig_T,dpi=100)
else:
    plt.show()

#close data
data.close()
