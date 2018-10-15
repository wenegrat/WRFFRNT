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
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import datetime as dt

#colorbar formatter
def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

#inputs
savefig = True
outfig = 'u_mix_hov.png'

#dz=5
zplot = 10

#FORCED#####################################################################################
dirall = '/p/lscratchh/arthur7/front/forced_d188.7_dz5_2Tispinup_isfflx_2'
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
    fpn = dirall + wrfout[n] + hr + ':' + mins[n] + ':00'
    print fpn
    data = Dataset(fpn,mode='r')

    #get variables from netcdf
    print 'getting data'
    nt = nt + data.dimensions['Time'].size

    if n==0:
        ph = data.variables['PH'][:,:,0,0]
        phb = data.variables['PHB'][:,:,0,0]
        z = (ph[0,0:-1] + ph[0,1:] + phb[0,0:-1] + phb[0,1:]) / 2 / 9.81
        k = np.argwhere(z>=zplot)[0][0]
        print k,z[k]

    #budget variables
    # if n==0:
    #     adv_u_u = data.variables['ADV_U_U'][:,k,:,:]
    #     adv_u_w = data.variables['ADV_U_W'][:,k,:,:]
    #     pres_u = data.variables['PRES_U'][:,k,:,:]
    #     cor_u = data.variables['COR_U'][:,k,:,:]
    #     hdiff_u = data.variables['HDIFF_U'][:,k,:,:]
    #     vdiff_u = data.variables['VDIFF_U'][:,k,:,:]
    # else:
    #     adv_u_u = np.concatenate((adv_u_u,data.variables['ADV_U_U'][:,k,:,:]))
    #     adv_u_w = np.concatenate((adv_u_w,data.variables['ADV_U_W'][:,k,:,:]))
    #     pres_u = np.concatenate((pres_u,data.variables['PRES_U'][:,k,:,:]))
    #     cor_u = np.concatenate((cor_u,data.variables['COR_U'][:,k,:,:]))
    #     hdiff_u = np.concatenate((hdiff_u,data.variables['HDIFF_U'][:,k,:,:]))
    #     vdiff_u = np.concatenate((vdiff_u,data.variables['VDIFF_U'][:,k,:,:]))

    # if n==0:
    #     tmp = data.variables['ADV_W_U'][:,k:k+2,:,:]
    #     adv_w_u = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
    #     tmp = data.variables['ADV_W_W'][:,k:k+2,:,:]
    #     adv_w_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
    #     tmp = data.variables['PRES_W'][:,k:k+2,:,:]
    #     pres_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
    #     tmp = data.variables['COR_W'][:,k:k+2,:,:]
    #     cor_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
    #     tmp = data.variables['HDIFF_W'][:,k:k+2,:,:]
    #     hdiff_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
    #     tmp = data.variables['VDIFF_W'][:,k:k+2,:,:]
    #     vdiff_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
    # else:
    #     tmp = data.variables['ADV_W_U'][:,k:k+2,:,:]
    #     adv_w_u = np.concatenate((adv_w_u,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
    #     tmp = data.variables['ADV_W_W'][:,k:k+2,:,:]
    #     adv_w_w = np.concatenate((adv_w_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
    #     tmp = data.variables['PRES_W'][:,k:k+2,:,:]
    #     pres_w = np.concatenate((pres_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
    #     tmp = data.variables['COR_W'][:,k:k+2,:,:]
    #     cor_w = np.concatenate((cor_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
    #     tmp = data.variables['HDIFF_W'][:,k:k+2,:,:]
    #     hdiff_w = np.concatenate((hdiff_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
    #     tmp = data.variables['VDIFF_W'][:,k:k+2,:,:]
    #     vdiff_w = np.concatenate((vdiff_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))

    # if n==0:
    #     adv_t_u = data.variables['ADV_T_U'][:,k,:,:]
    #     adv_t_w = data.variables['ADV_T_W'][:,k,:,:]
    #     hdiff_t = data.variables['HDIFF_T'][:,k,:,:]
    #     vdiff_t = data.variables['VDIFF_T'][:,k,:,:]
    # else:
    #     adv_t_u = np.concatenate((adv_t_u,data.variables['ADV_T_U'][:,k,:,:]))
    #     adv_t_w = np.concatenate((adv_t_w,data.variables['ADV_T_W'][:,k,:,:]))
    #     hdiff_t = np.concatenate((hdiff_t,data.variables['HDIFF_T'][:,k,:,:]))
    #     vdiff_t = np.concatenate((vdiff_t,data.variables['VDIFF_T'][:,k,:,:]))

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
    # #calculate mean and turbulent stress divergences for cross terms (u,w)

    # #project u and w to e points
    # ue = fnm[:,1:] * u[:,1:,:,1:-1] + fnp[:,1:] * u[:,:-1,:,1:-1] #dont need i edge values
    # we = 0.5 * (w[:,:,:,1:] + w[:,:,:,:-1])

    # #swap axes so that broadcasting works
    # #index is now n,i,j,k instead of n,k,j,i
    # ue = np.swapaxes(ue,1,3)
    # we = np.swapaxes(we,1,3)

    # #calculate primes as departure from along-front average
    # uemean = np.mean(ue,axis=2)
    # wemean = np.mean(we,axis=2)
    # ueprime = ue - np.expand_dims(uemean,axis=2)
    # weprime = we - np.expand_dims(wemean,axis=2)

    # #calculate u'w' with along-front average
    # upwpe = np.mean(ueprime*weprime,axis=2)

    # #calculate stress divergences
    # if n==0:
    #     wdudz = 0.5*(wemean[:,:,1:] + wemean[:,:,:-1]) * (uemean[:,:,1:] - uemean[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1]) #at u points
    #     tmp = 0.5*(uemean[:,1:,:] + uemean[:,:-1,:]) * (wemean[:,1:,:] - wemean[:,:-1,:]) / dx #at w points
    #     udwdx = 0.5*(tmp[:,:,1:] + tmp[:,:,:-1]) #at cell centers
    #     ddzupwp = (upwpe[:,:,1:] - upwpe[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1]) #at u points
    #     tmp = (upwpe[:,1:,:] - upwpe[:,:-1,:]) / dx #at w points
    #     ddxupwp = 0.5*(tmp[:,:,1:] + tmp[:,:,:-1]) #at cell centers
    # else:
    #     wdudz = np.concatenate((wdudz,0.5*(wemean[:,:,1:] + wemean[:,:,:-1]) * (uemean[:,:,1:] - uemean[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1]))) #at u points
    #     tmp = 0.5*(uemean[:,1:,:] + uemean[:,:-1,:]) * (wemean[:,1:,:] - wemean[:,:-1,:]) / dx #at w points
    #     udwdx = np.concatenate((udwdx,0.5*(tmp[:,:,1:] + tmp[:,:,:-1]))) #at cell centers
    #     ddzupwp = np.concatenate((ddzupwp,(upwpe[:,:,1:] - upwpe[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1]))) #at u points
    #     tmp = (upwpe[:,1:,:] - upwpe[:,:-1,:]) / dx #at w points
    #     ddxupwp = np.concatenate((ddxupwp,0.5*(tmp[:,:,1:] + tmp[:,:,:-1]))) #at cell centers

######################################################################################
#along-front average tendency terms
ax = 1

# adv_u_u_avg = np.mean(adv_u_u,axis=ax)
# adv_u_w_avg = np.mean(adv_u_w,axis=ax)
# pres_u_avg = np.mean(pres_u,axis=ax)
# cor_u_avg = np.mean(cor_u,axis=ax)
# hdiff_u_avg = np.mean(hdiff_u,axis=ax)
# vdiff_u_avg = np.mean(vdiff_u,axis=ax)

# adv_w_u_avg = np.mean(adv_w_u,axis=ax)
# adv_w_w_avg = np.mean(adv_w_w,axis=ax)
# pres_w_avg = np.mean(pres_w,axis=ax)
# cor_w_avg = np.mean(cor_w,axis=ax)
# hdiff_w_avg = np.mean(hdiff_w,axis=ax)
# vdiff_w_avg = np.mean(vdiff_w,axis=ax)

# adv_t_u_avg = np.mean(adv_t_u,axis=ax)
# adv_t_w_avg = np.mean(adv_t_w,axis=ax)
# hdiff_t_avg = np.mean(hdiff_t,axis=ax)
# vdiff_t_avg = np.mean(vdiff_t,axis=ax)

u_allt_c = 0.5 * (u_allt[:,:,1:] + u_allt[:,:,:-1])
u_allt_c_avg = np.mean(u_allt_c,axis=ax)
# dudt_avg = np.mean((u_allt[2:,:,:]-u_allt[:-2,:,:])/(2*10*60),axis=ax)
# dudt_avg = np.mean(dudt,axis=ax)
# dwdt_avg = np.mean(dwdt,axis=ax)
# dTdt_avg = np.mean(dTdt,axis=ax)

######################################################################################
#plot
tStart = dt.datetime(2000,01,01,00,00,00)
t = np.array([tStart + dt.timedelta(minutes=n*10) for n in range(nt)])
# tout = 10
# t = np.arange(0,nt*tout,tout)/60.0 #hours
# t = t / 17.5

L = 10.0
plt.figure(figsize=(18,8))

plt.subplot(1,2,1)
plt.pcolormesh(x,t,u_allt_c_avg,cmap='RdBu_r')
plt.colorbar()
plt.clim(-0.6,0.6)
plt.plot([L/4,L/4],[t[0],t[-1]],'k--',linewidth=2)
plt.plot([3*L/4,3*L/4],[t[0],t[-1]],'k--',linewidth=2)
plt.title('$U_{10}\ \mathrm{[m\ s^{-1}]}$',fontsize=24)
plt.xlabel('$x\ \mathrm{[km]}$',fontsize=24)
# plt.ylabel('$t/T$',fontsize=24)
plt.ylabel(r'$\mathrm{Time}$',fontsize=24)
plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tick_params(labelsize=16)

plt.subplot(1,2,2)
plt.pcolormesh(xs[1:-1],t,ddxupTp,cmap='RdBu_r')
plt.clim(-6e-5,6e-5)
plt.colorbar(format=ticker.FuncFormatter(fmt))
plt.plot([L/4,L/4],[t[0],t[-1]],'k--',linewidth=2)
plt.plot([3*L/4,3*L/4],[t[0],t[-1]],'k--',linewidth=2)
plt.title(r"$\partial / \partial x (\overline{u'\theta'})\ \mathrm{[K\ s^{-1}]}$",fontsize=24)
plt.xlabel('$x\ \mathrm{[km]}$',fontsize=24)
plt.gca().set_yticklabels('')
plt.tick_params(labelsize=16)


if savefig:
    plt.savefig(outfig,dpi=100)
else:
    plt.show()

#close data
data.close()
