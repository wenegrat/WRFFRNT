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
nk = 100
nk += 1
# Preallocate
adv_u_u_avg = np.zeros((106, nk, 501))
adv_u_w_avg = np.zeros((106, nk, 501))

ududx_avg = np.zeros((106, nk, 500))
ddxupup_avg = np.zeros((106, nk, 500))
wdudz_avg = np.zeros((106, nk, 499))
ddzupwp_avg = np.zeros((106, nk, 499))

pres_u_avg= np.zeros((106, nk, 501))
cor_u_avg = np.zeros((106, nk, 501))
hdiff_u_avg = np.zeros((106, nk, 501))
vdiff_u_avg = np.zeros((106, nk, 501))

adv_w_u_avg = np.zeros((106, nk, 500))
adv_w_w_avg = np.zeros((106, nk, 500))
pres_w_avg = np.zeros((106, nk, 500))
cor_w_avg = np.zeros((106, nk, 500))
hdiff_w_avg = np.zeros((106, nk, 500))
vdiff_w_avg = np.zeros((106, nk, 500))

adv_t_u_avg = np.zeros((106, nk, 500))
adv_t_w_avg = np.zeros((106, nk, 500))
hdiff_t_avg = np.zeros((106, nk, 500))
vdiff_t_avg = np.zeros((106, nk, 500))

dudt_avg = np.zeros((104, nk, 501))

nk -= 1
for k in range(1, nk+1):
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
            zf = (ph[0,0:-1] + ph[0,1:] + phb[0,0:-1] + phb[0,1:]) / 2 / 9.81
            print(k,zf[k])
    
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
    
#    adv_u_u_avg[:,k,:] = np.mean(adv_u_u,axis=ax)
#    adv_u_w_avg[:,k,:] = np.mean(adv_u_w,axis=ax)
    ududx_avg[:,k,:] = np.squeeze(ududx)
    ddxupup_avg[:,k,:] = np.squeeze(ddxupup)
    wdudz_avg[:,k,:] = np.squeeze(wdudz)
    ddzupwp_avg[:,k,:] = np.squeeze(ddzupwp)
    pres_u_avg[:,k,:] = np.mean(pres_u,axis=ax)
    cor_u_avg[:,k,:] = np.mean(cor_u,axis=ax)
    hdiff_u_avg[:,k,:] = np.mean(hdiff_u,axis=ax)
    vdiff_u_avg[:,k,:] = np.mean(vdiff_u,axis=ax)
    
    adv_w_u_avg[:,k,:] = np.mean(adv_w_u,axis=ax)
    adv_w_w_avg[:,k,:] = np.mean(adv_w_w,axis=ax)
    pres_w_avg[:,k,:] = np.mean(pres_w,axis=ax)
    cor_w_avg[:,k,:] = np.mean(cor_w,axis=ax)
    hdiff_w_avg[:,k,:] = np.mean(hdiff_w,axis=ax)
    vdiff_w_avg[:,k,:] = np.mean(vdiff_w,axis=ax)
    
    adv_t_u_avg[:,k,:] = np.mean(adv_t_u,axis=ax)
    adv_t_w_avg[:,k,:] = np.mean(adv_t_w,axis=ax)
    hdiff_t_avg[:,k,:] = np.mean(hdiff_t,axis=ax)
    vdiff_t_avg[:,k,:] = np.mean(vdiff_t,axis=ax)
    
    dudt_avg[:,k,:] = np.mean((u_allt[2:,:,:]-u_allt[:-2,:,:])/(2*10*60),axis=ax)
# dudt_avg = np.mean(dudt,axis=ax)
# dwdt_avg = np.mean(dwdt,axis=ax)
# dTdt_avg = np.mean(dTdt,axis=ax)

#%% ZONAL
#Set time range

nh = 0.25
zl = 300
xl = [2.5, 7.5]
xlim = 1.5e-4

plt.figure()
for i in range(0, 2):
    #Set x range
    tind1 = np.int((i+nh)*60/10)
    tind2 = np.int((i+1+nh)*60/10)

    warm = True

    xlind1 = np.where(xs>xl[0])[0][0]
    xlind2 = np.where(xs>xl[1])[0][0]
    xr = range(xlind1, xlind2)
    # WARM PATCH
    if warm:
        xr = np.r_[range(0, xlind1), range(xlind2, 498)]  
    
    plt.subplot(2,2,2*i + 1)
    plt.plot(np.mean(np.mean(dudt_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]),linestyle='--', label='TEND')

    plt.plot(np.mean(np.mean(-ududx_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]), label='UADV')
    plt.plot(np.mean(np.mean(-wdudz_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]), label='WADV')
    plt.plot(np.mean(np.mean(pres_u_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]), label='PRES')
    plt.plot(np.mean(np.mean(cor_u_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]), label = 'COR')
    plt.plot(np.mean(np.mean(-ddzupwp_avg[tind1:tind2,:,xr]+vdiff_u_avg[tind1:tind2,:,xr]+0*cor_u_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]), label='VDIFF')
    plt.plot(np.mean(np.mean(-ddxupup_avg[tind1:tind2,:,xr]+hdiff_u_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]), label='HDIFF')
    plt.legend()
    plt.ylim((0, zl))
    plt.xlim((-xlim, xlim))
    plt.grid()
    plt.title('Warm SST   Hour: %i' % int(tind1/6))
    #Set x range
    warm = False
    xlind1 = np.where(xs>xl[0])[0][0]
    xlind2 = np.where(xs>xl[1])[0][0]
    xr = range(xlind1, xlind2)
    # WARM PATCH
    if warm:
        xr = np.r_[range(0, xlind1), range(xlind2, 498)]  
    
    plt.subplot(2,2,2*(i+1))
    plt.plot(np.mean(np.mean(dudt_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]),linestyle='--', label='TEND')

    plt.plot(np.mean(np.mean(-ududx_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]), label='UADV')
    plt.plot(np.mean(np.mean(-wdudz_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]), label='WADV')
    plt.plot(np.mean(np.mean(pres_u_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]), label='PRES')
    plt.plot(np.mean(np.mean(cor_u_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]), label = 'COR')
    plt.plot(np.mean(np.mean(-ddzupwp_avg[tind1:tind2,:,xr]+vdiff_u_avg[tind1:tind2,:,xr]+0*cor_u_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]), label='VDIFF')
    plt.plot(np.mean(np.mean(-ddxupup_avg[tind1:tind2,:,xr]+hdiff_u_avg[tind1:tind2,:,xr], axis=0), axis=-1), np.array(zf[0:nk+1]), label='HDIFF')
    plt.legend()
    plt.ylim((0, zl))
    plt.xlim((-xlim, xlim))
    plt.title('Cold SST   Hour: %i' % int(tind1/6))

    plt.grid()
#%% HOV
#Set x range
nt = 106
tStart = dt.datetime(2000,1,1,00,00,00)
t = np.array([tStart + dt.timedelta(minutes=n*10) for n in range(nt)])

warm = False
xl = [2.5, 7.5]
xlind1 = np.where(xs>xl[0])[0][0]
xlind2 = np.where(xs>xl[1])[0][0]
xr = range(xlind1, xlind2)
# WARM PATCH
if warm:
    xr = np.r_[range(0, xlind1), range(xlind2, 498)]  
plt.figure()


tendplot = [np.transpose(np.mean(dudt_avg[:,:,xr], axis=-1)),np.transpose(np.mean(-ududx_avg[:,:,xr]-wdudz_avg[:,:,xr], axis=-1))
    ,np.transpose(np.mean(pres_u_avg[:,:,xr], axis=-1)),np.transpose(np.mean(cor_u_avg[:,:,xr], axis=-1)),
   np.transpose(np.mean(-ddzupwp_avg[:,:,xr]+vdiff_u_avg[:,:,xr], axis=-1)),np.transpose(np.mean(-ddxupup_avg[:,:,xr]+hdiff_u_avg[:,:,xr], axis=-1))]
tendmax = 1e-4
titles = ['TEND','ADV','PRES','CORR','VDIFF','HDIFF']
tplot = [t[1:-1],t,t,t,t,t,t,t,t]

for n in range(6):
    plt.subplot(3,2,n+1)
    plt.pcolormesh(tplot[n],zf[0:nk+1], np.squeeze(tendplot[n]),cmap='RdBu_r')
    plt.colorbar()
    plt.clim(-tendmax,tendmax)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.ylabel('t')
    plt.title(titles[n])
#%%
######################################################################################
#plot
tStart = dt.datetime(2000,1,1,00,00,00)
t = np.array([tStart + dt.timedelta(minutes=n*10) for n in range(nt)])
L = 10.0

#Set x range
warm = False
xl = [2.5, 7.5]
xlind1 = np.where(xs>xl[0])[0][0]
xlind2 = np.where(xs>xl[1])[0][0]
xr = range(xlind1, xlind2)
# WARM PATCH
if warm:
    xr = np.r_[range(0, xlind1), range(xlind2, 498)]


dudt_savg = np.mean(dudt_avg[:,xr], axis=-1)
ududx_savg = np.mean(ududx[:,xr], axis=-1)
ddxupup_savg = np.mean(ddxupup[:,xr], axis=-1)
ddxupwp_savg = np.mean(ddzupwp[:,xr,0], axis=-1)

wdudz_savg = np.mean(wdudz[:,xr,0], axis=-1)
vdiff_savg = np.mean(-ddzupwp[:,xr,0]+vdiff_u_avg[:,xr], axis=-1)
hdiff_savg = np.mean(-ddxupup[:,xr]+hdiff_u_avg[:,xr], axis=-1)
pressu_savg = np.mean(pres_u_avg[:,xr], axis=-1)
coru_savg = np.mean(cor_u_avg[:,xr], axis=-1)
advu_savg = np.mean(np.mean(adv_u_u[:,:,xr], axis=-1), axis=-1)
advw_savg = np.mean(np.mean(adv_u_w[:,:,xr], axis=-1), axis=-1)
#plot hovmoller for u tendency terms
plt.figure()

xplot = [xs,x,xs[1:-1],x,xs[1:-1],xs,xs,xs,xs]
tplot = [t[1:-1],t,t,t,t,t,t,t,t]
tendplot = [dudt_savg, -ududx_savg, -wdudz_savg,  vdiff_savg, hdiff_savg, pressu_savg, coru_savg]
leg = ['dudt', 'ududx', 'wdudz', 'VDIFF', 'HDIFF', 'PRESS', 'COR']
for n in range(7):
    plt.plot(tplot[n], np.squeeze(tendplot[n]), label=leg[n])
    
total = -ududx_savg -wdudz_savg+ vdiff_savg+ hdiff_savg+ pressu_savg+ coru_savg 
#total = advw_savg + advu_savg+ vdiff_savg+ hdiff_savg+ pressu_savg+ coru_savg 

plt.plot(tplot[-1], total, linestyle=':', label='total')    
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


#%% TEST PRIME DECOMP
plt.figure()
plt.plot(t, advu_savg)
plt.plot(t, -ududx_savg-ddxupup_savg)

plt.figure()
plt.plot(t, advw_savg)
plt.plot(t, -wdudz_savg-ddxupwp_savg)

#n = 2
#plt.plot(tplot[n], np.squeeze(tendplot[n]))
#%%
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
