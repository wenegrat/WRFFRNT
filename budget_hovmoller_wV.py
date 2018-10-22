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
from scipy import ndimage
#%%

#inputs
savefig = True
outfig_u = 'budget_hov_u_10m.png'
outfig_v = 'budget_hov_v_10m.png'
outfig_w = 'budget_hov_w_10m.png'
outfig_T = 'budget_hov_T_10m.png'

#dz=5
zplot = 500

#FORCED#####################################################################################
#dirall = '/media/ExtDriveFolder/WRFRUNS/WeakRun2'
#wrfout = ['/wrfout_d01_0001-01-02_', \
#          '/wrfout_d01_0001-01-02_', \
#          '/wrfout_d01_0001-01-02_', \
#          '/wrfout_d01_0001-01-02_', \
#          '/wrfout_d01_0001-01-03_']#, \
#          # '/wrfout_d01_0001-01-03_']
#hrs = ['11','14','18','21','01']#,'04']
#mins = ['00','30','00','30','00']#,'30']

dirall = '/media/ExtDriveFolder/WRFRUNS/StrongRun1'
wrfout = ['/wrfout_d01_0001-01-02_', \
          '/wrfout_d01_0001-01-02_', \
          '/wrfout_d01_0001-01-02_', \
          '/wrfout_d01_0001-01-02_']

hrs = ['11','14','18','21']#,'04']
mins = ['00','30','00','30']#,'30']

nt = 0
for n,hr in enumerate(hrs):
    fpn = dirall  + wrfout[n] + hr + '_' + mins[n] + '_00'
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
        # adv_u_u = data.variables['ADV_U_U'][:,k,:,:]
        # adv_u_v = data.variables['ADV_U_V'][:,k,:,:]
        # adv_u_w = data.variables['ADV_U_W'][:,k,:,:]
        pres_u = data.variables['PRES_U'][:,k,:,:]
        cor_u = data.variables['COR_U'][:,k,:,:]
        hdiff_u = data.variables['HDIFF_U'][:,k,:,:]
        vdiff_u = data.variables['VDIFF_U'][:,k,:,:]
    else:
        # adv_u_u = np.concatenate((adv_u_u,data.variables['ADV_U_U'][:,k,:,:]))
        # adv_u_v = np.concatenate((adv_u_v,data.variables['ADV_U_V'][:,k,:,:]))
        # adv_u_w = np.concatenate((adv_u_w,data.variables['ADV_U_W'][:,k,:,:]))
        pres_u = np.concatenate((pres_u,data.variables['PRES_U'][:,k,:,:]))
        cor_u = np.concatenate((cor_u,data.variables['COR_U'][:,k,:,:]))
        hdiff_u = np.concatenate((hdiff_u,data.variables['HDIFF_U'][:,k,:,:]))
        vdiff_u = np.concatenate((vdiff_u,data.variables['VDIFF_U'][:,k,:,:]))

    if n==0:
        # adv_v_u = data.variables['ADV_V_U'][:,k,:,:]
        # adv_v_v = data.variables['ADV_V_V'][:,k,:,:]
        # adv_v_w = data.variables['ADV_V_W'][:,k,:,:]
        pres_v = data.variables['PRES_V'][:,k,:,:]
        cor_v = data.variables['COR_V'][:,k,:,:]
        hdiff_v = data.variables['HDIFF_V'][:,k,:,:]
        vdiff_v = data.variables['VDIFF_V'][:,k,:,:]
    else:
        # adv_v_u = np.concatenate((adv_v_u,data.variables['ADV_V_U'][:,k,:,:]))
        # adv_v_v = np.concatenate((adv_v_v,data.variables['ADV_V_U'][:,k,:,:]))
        # adv_v_w = np.concatenate((adv_v_w,data.variables['ADV_V_W'][:,k,:,:]))
        pres_v = np.concatenate((pres_v,data.variables['PRES_V'][:,k,:,:]))
        cor_v = np.concatenate((cor_v,data.variables['COR_V'][:,k,:,:]))
        hdiff_v = np.concatenate((hdiff_v,data.variables['HDIFF_V'][:,k,:,:]))
        vdiff_v = np.concatenate((vdiff_v,data.variables['VDIFF_V'][:,k,:,:]))

    if n==0:
        # tmp = data.variables['ADV_W_U'][:,k:k+2,:,:]
        # adv_w_u = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
        # tmp = data.variables['ADV_W_V'][:,k:k+2,:,:]
        # adv_w_v = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
        # tmp = data.variables['ADV_W_W'][:,k:k+2,:,:]
        # adv_w_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
        tmp = data.variables['PRES_W'][:,k:k+2,:,:]
        pres_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
        tmp = data.variables['COR_W'][:,k:k+2,:,:]
        cor_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
        tmp = data.variables['HDIFF_W'][:,k:k+2,:,:]
        hdiff_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
        tmp = data.variables['VDIFF_W'][:,k:k+2,:,:]
        vdiff_w = 0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])
    else:
        # tmp = data.variables['ADV_W_U'][:,k:k+2,:,:]
        # adv_w_u = np.concatenate((adv_w_u,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
        # tmp = data.variables['ADV_W_V'][:,k:k+2,:,:]
        # adv_w_v = np.concatenate((adv_w_v,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
        # tmp = data.variables['ADV_W_W'][:,k:k+2,:,:]
        # adv_w_w = np.concatenate((adv_w_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
        tmp = data.variables['PRES_W'][:,k:k+2,:,:]
        pres_w = np.concatenate((pres_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
        tmp = data.variables['COR_W'][:,k:k+2,:,:]
        cor_w = np.concatenate((cor_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
        tmp = data.variables['HDIFF_W'][:,k:k+2,:,:]
        hdiff_w = np.concatenate((hdiff_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))
        tmp = data.variables['VDIFF_W'][:,k:k+2,:,:]
        vdiff_w = np.concatenate((vdiff_w,0.5*(tmp[:,0,:,:] + tmp[:,1,:,:])))

    if n==0:
        # adv_t_u = data.variables['ADV_T_U'][:,k,:,:]
        # adv_t_v = data.variables['ADV_T_V'][:,k,:,:]
        # adv_t_w = data.variables['ADV_T_W'][:,k,:,:]
        hdiff_t = data.variables['HDIFF_T'][:,k,:,:]
        vdiff_t = data.variables['VDIFF_T'][:,k,:,:]
    else:
        # adv_t_u = np.concatenate((adv_t_u,data.variables['ADV_T_U'][:,k,:,:]))
        # adv_t_v = np.concatenate((adv_t_v,data.variables['ADV_T_V'][:,k,:,:]))
        # adv_t_w = np.concatenate((adv_t_w,data.variables['ADV_T_W'][:,k,:,:]))
        hdiff_t = np.concatenate((hdiff_t,data.variables['HDIFF_T'][:,k,:,:]))
        vdiff_t = np.concatenate((vdiff_t,data.variables['VDIFF_T'][:,k,:,:]))

    #raw variables for turbulent calculations
    ph = data.variables['PH'][:,k:k+2,:,:]
    phb = data.variables['PHB'][:,k:k+2,:,:]
    u = data.variables['U'][:,k-1:k+2,:,:]
    v = data.variables['V'][:,k-1:k+2,:,:]
    w = data.variables['W'][:,k:k+2,:,:]
    T = data.variables['T'][:,k-1:k+2,:,:]
    fnm = data.variables['FNM'][:,k:k+2]
    fnm = np.expand_dims(np.expand_dims(fnm,axis=2),axis=3) #for broadcasting
    fnp = data.variables['FNP'][:,k:k+2]
    fnp = np.expand_dims(np.expand_dims(fnp,axis=2),axis=3) #for broadcasting

    #time tendency terms
    if n==0:
        u_allt = u[:,1,:,:]
        v_allt = v[:,1,:,:]
        w_allt = 0.5 * (w[:,0,:,:] + w[:,1,:,:])
        T_allt = T[:,1,:,:]
    else:
        u_allt = np.concatenate((u_allt,u[:,1,:,:]))
        v_allt = np.concatenate((v_allt,v[:,1,:,:]))
        w_allt = np.concatenate((w_allt,0.5*(w[:,0,:,:]+w[:,1,:,:])))
        T_allt = np.concatenate((T_allt,T[:,1,:,:]))

    dx = getattr(data,'DX')
    dy = getattr(data,'DY')
    nx = data.dimensions['west_east'].size
    ny = data.dimensions['south_north'].size
    nxs = data.dimensions['west_east_stag'].size
    nys = data.dimensions['south_north_stag'].size
    x = np.linspace(dx/2,nx*dx-dx/2,nx)/1000
    y = np.linspace(dy/2,ny*dy-dy/2,ny)/1000
    xs = np.linspace(0,nx*dx,nxs)/1000
    ys = np.linspace(0,ny*dy,nys)/1000

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

    #calculate stress divergences (all at cell centers).
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

    #calculate u'T', v'T', and w'T' with along-front average
    upTp = np.mean(uprime[:,1:-1,:,:]*Tuprime, axis=2)
    wpTp = np.mean(wprime*Twprime, axis=2)

    #calculate flux divergences (all at cell centers).
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
    #calculate mean and turbulent stress divergences for u and w terms

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
    #calculate mean and turbulent stress divergences for v terms

    #project u and v to d points, v and w to f points
    uc = 0.5 * (u[:,:,:,1:] + u[:,:,:,:-1])
    uv = 0.5 * (uc[:,:,1:,:] + uc[:,:,:-1,:])
    ud = 0.5 * (u[:,:,1:,1:-1] + u[:,:,:-1,1:-1]) #dont need i edge values
    vd = 0.5 * (v[:,:,1:-1,1:] + v[:,:,1:-1,:-1]) #dont need j edge values
    vf = fnm[:,1:] * v[:,1:,1:-1,:] + fnp[:,1:] * v[:,:-1,1:-1,:] #dont need j edge values
    wf = 0.5 * (w[:,:,1:,:] + w[:,:,:-1,:])

    #swap axes so that broadcasting works
    #index is now n,i,j,k instead of n,k,j,i
    uv = np.swapaxes(uv,1,3)
    ud = np.swapaxes(ud,1,3)
    vd = np.swapaxes(vd,1,3)
    vf = np.swapaxes(vf,1,3)
    wf = np.swapaxes(wf,1,3)

    #calculate primes as departure from along-front average
    uvmean = np.mean(uv,axis=2)
    udmean = np.mean(ud,axis=2)
    vdmean = np.mean(vd,axis=2)
    vfmean = np.mean(vf,axis=2)
    wfmean = np.mean(wf,axis=2)
    udprime = ud - np.expand_dims(udmean,axis=2)
    vdprime = vd - np.expand_dims(vdmean,axis=2)
    vfprime = vf - np.expand_dims(vfmean,axis=2)
    wfprime = wf - np.expand_dims(wfmean,axis=2)

    #calculate u'v' and v'w' with along-front average
    upvpd = np.mean(udprime*vdprime,axis=2)
    vpwpf = np.mean(vfprime*wfprime,axis=2)

    #calculate stress divergences
    if n==0:
        udvdx = uvmean[:,1:-1,1] * (vdmean[:,1:,1] - vdmean[:,:-1,1]) / dx #at v points
        wdvdz = 0.5 * (wfmean[:,:,1:] + wfmean[:,:,:-1]) * (vfmean[:,:,1:] - vfmean[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1]) #at v points
        ddxupvp = (upvpd[:,1:,1] - upvpd[:,:-1,1]) / dx #at v points
        ddzvpwp = (vpwpf[:,:,1:] - vpwpf[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1]) #at v points
    else:
        udvdx = np.concatenate((udvdx,uvmean[:,1:-1,1] * (vdmean[:,1:,1] - vdmean[:,:-1,1]) / dx)) #at v points
        wdvdz = np.concatenate((wdvdz,0.5 * (wfmean[:,:,1:] + wfmean[:,:,:-1]) * (vfmean[:,:,1:] - vfmean[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1]))) #at v points
        ddxupvp = np.concatenate((ddxupvp,(upvpd[:,1:,1] - upvpd[:,:-1,1]) / dx)) #at v points
        ddzvpwp = np.concatenate((ddzvpwp,(vpwpf[:,:,1:] - vpwpf[:,:,:-1]) / (zs[:,:,1:] - zs[:,:,:-1]))) #at v points

######################################################################################
#along-front average tendency terms
ax = 1

# adv_u_u_avg = np.mean(adv_u_u,axis=ax)
# adv_u_v_avg = np.mean(adv_u_u,axis=ax)
# adv_u_w_avg = np.mean(adv_u_w,axis=ax)
pres_u_avg = np.mean(pres_u,axis=ax)
cor_u_avg = np.mean(cor_u,axis=ax)
hdiff_u_avg = np.mean(hdiff_u,axis=ax)
vdiff_u_avg = np.mean(vdiff_u,axis=ax)

# adv_v_u_avg = np.mean(adv_v_u,axis=ax)
# adv_v_v_avg = np.mean(adv_v_v,axis=ax)
# adv_v_w_avg = np.mean(adv_v_w,axis=ax)
pres_v_avg = np.mean(pres_v,axis=ax)
cor_v_avg = np.mean(cor_v,axis=ax)
hdiff_v_avg = np.mean(hdiff_v,axis=ax)
vdiff_v_avg = np.mean(vdiff_v,axis=ax)

# adv_w_u_avg = np.mean(adv_w_u,axis=ax)
# adv_w_v_avg = np.mean(adv_w_v,axis=ax)
# adv_w_w_avg = np.mean(adv_w_w,axis=ax)
pres_w_avg = np.mean(pres_w,axis=ax)
cor_w_avg = np.mean(cor_w,axis=ax)
hdiff_w_avg = np.mean(hdiff_w,axis=ax)
vdiff_w_avg = np.mean(vdiff_w,axis=ax)

# adv_t_u_avg = np.mean(adv_t_u,axis=ax)
# adv_t_v_avg = np.mean(adv_t_v,axis=ax)
# adv_t_w_avg = np.mean(adv_t_w,axis=ax)
hdiff_t_avg = np.mean(hdiff_t,axis=ax)
vdiff_t_avg = np.mean(vdiff_t,axis=ax)

dudt_avg = np.mean((u_allt[2:,:,:]-u_allt[:-2,:,:])/(2*10*60),axis=ax)
dvdt_avg = np.mean((v_allt[2:,:,:]-v_allt[:-2,:,:])/(2*10*60),axis=ax)
dwdt_avg = np.mean((w_allt[2:,:,:]-w_allt[:-2,:,:])/(2*10*60),axis=ax)
dTdt_avg = np.mean((T_allt[2:,:,:]-T_allt[:-2,:,:])/(2*10*60),axis=ax)

######################################################################################
#plot
tStart = dt.datetime(2000,1,1,00,00,00)
t = np.array([tStart + dt.timedelta(minutes=n*10) for n in range(nt)])
L = 10.0

#plot hovmoller for u tendency terms
# plt.figure(figsize=(25,18))

# xplot = [xs,x,xs[1:-1],x,xs[1:-1],xs,xs,xs,xs]
# tplot = [t[1:-1],t,t,t,t,t,t,t,t]
# tendplot = [dudt_avg,-ududx,-wdudz,-ddxupup,-ddzupwp,pres_u_avg,cor_u_avg,hdiff_u_avg,vdiff_u_avg]
# tendmax = max(np.abs(np.amax(dudt_avg)), \
#               np.abs(np.amax(-ududx)), \
#               np.abs(np.amax(-wdudz)), \
#               np.abs(np.amax(-ddxupup)), \
#               np.abs(np.amax(-ddzupwp)), \
#               np.abs(np.amax(pres_u_avg)), \
#               np.abs(np.amax(cor_u_avg)), \
#               np.abs(np.amax(hdiff_u_avg)), \
#               np.abs(np.amax(vdiff_u_avg)))
# titles = ['dudt','ududx','wdudz','ddxupup','ddzupwp','PRES_U','COR_U','HDIFF_U','VDIFF_U']
# for n in range(9):
#     plt.subplot(3,3,n+1)
#     plt.pcolormesh(xplot[n],tplot[n],np.squeeze(tendplot[n]),cmap='RdBu_r')
#     plt.colorbar()
#     plt.clim(-0.1*tendmax,0.1*tendmax)
#     plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#     if (n+1)>=7:
#         plt.xlabel('x [km]')
#     if n==0 or n==2 or n==4 or n==6:
#         plt.ylabel('t')
#     plt.title(titles[n])

# if savefig:
#     plt.savefig(outfig_u,dpi=100)

#plot hovmoller for v tendency terms
#%% TEMPERATURE
plt.figure(figsize=(20,12))

xplot = [x,x[1:-1],x,x[1:-1],x,x,x,x,x]
tplot = [t[1:-1],t,t,t,t,t,t,t,t]
tendplot = [dTdt_avg,-udTdx,-wdTdz,-ddxupTp,-ddzwpTp[:,:,0] +vdiff_t_avg,hdiff_t_avg,vdiff_t_avg]
tendmax = 5e-5
titles = ['dTdt','udTdx','wdTdz','ddxupTp','ddzwptp','$HDIFF_T$','$VDIFF_T$']
for n in range(7):
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
    plt.savefig(outfig_v,dpi=100)
else:
    plt.show()
    
#%% VERT VEL HOV
fsizex = 25 #250
fsizey = 25 #150
w_filt = ndimage.uniform_filter(w_allt,mode='wrap',size=(1,fsizey,fsizex))
plt.figure()
plt.pcolor(x, t, np.mean(w_filt, axis=1))
plt.colorbar()
plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#%% MERIDIONAL
plt.figure(figsize=(20,12))

xplot = [x,x[1:-1],x,x[1:-1],x,x,x,x,x]
tplot = [t[1:-1],t,t,t,t,t,t,t,t]
tendplot = [dvdt_avg,-udvdx,-wdvdz,-ddxupvp,-ddzvpwp[:,:,0] +vdiff_v_avg,pres_v_avg,cor_v_avg,hdiff_v_avg,vdiff_v_avg]
tendmax = max(np.abs(np.amax(dvdt_avg)), \
              np.abs(np.amax(-udvdx)), \
              np.abs(np.amax(-wdvdz)), \
              np.abs(np.amax(-ddxupvp)), \
              np.abs(np.amax(-ddzvpwp)), \
              np.abs(np.amax(pres_v_avg)), \
              np.abs(np.amax(cor_v_avg)), \
              np.abs(np.amax(hdiff_v_avg)), \
              np.abs(np.amax(vdiff_v_avg)))
titles = ['dvdt','udvdx','wdvdz','ddxupvp','ddzvpwp','$PRES_V$','$COR_V$','$HDIFF_V$','$VDIFF_V$']
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
    plt.savefig(outfig_v,dpi=100)
else:
    plt.show()
        
    
#%%
    # SPATIAL AVERAGE PLOT TEMPERATURE
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


dTdt_savg = np.mean(dTdt_avg[:,xr], axis=-1)
udTdx_savg = np.mean(udTdx[:,xr], axis=-1)
ddxupTp_savg = np.mean(ddxupTp[:,xr], axis=-1)
ddxwpTp_savg = np.mean(ddzwpTp[:,xr,0], axis=-1)

wdTdz_savg = np.mean(wdTdz[:,xr,0], axis=-1)
vdiff_savg = np.mean(-ddzwpTp[:,xr,0]+vdiff_t_avg[:,xr], axis=-1)
hdiff_savg = np.mean(-ddxupTp[:,xr]+hdiff_t_avg[:,xr], axis=-1)

#plot hovmoller for u tendency terms
plt.figure()

xplot = [xs,x,xs[1:-1],x,xs[1:-1],xs,xs,xs,xs]
tplot = [t[1:-1],t,t,t,t,t,t,t,t]
tendplot = [dTdt_savg, -udTdx_savg, -wdTdz_savg,  vdiff_savg, hdiff_savg]
leg = ['dTdt', 'udTdx', 'wdTdz', 'VDIFF', 'HDIFF']
for n in range(5):
    plt.plot(tplot[n], np.squeeze(tendplot[n]), label=leg[n])
    
#total = -ududx_savg -wdudz_savg+ vdiff_savg+ hdiff_savg+ pressu_savg+ coru_savg 
##total = advw_savg + advu_savg+ vdiff_savg+ hdiff_savg+ pressu_savg+ coru_savg 
#
#plt.plot(tplot[-1], total, linestyle=':', label='total')    
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#%%
    # SPATIAL AVERAGE PLOT
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


dvdt_savg = np.mean(dvdt_avg[:,xr], axis=-1)
udvdx_savg = np.mean(udvdx[:,xr], axis=-1)
ddxupvp_savg = np.mean(ddxupvp[:,xr], axis=-1)
ddxvpwp_savg = np.mean(ddzvpwp[:,xr,0], axis=-1)

wdvdz_savg = np.mean(wdvdz[:,xr,0], axis=-1)
vdiff_savg = np.mean(-ddzvpwp[:,xr,0]+vdiff_v_avg[:,xr], axis=-1)
hdiff_savg = np.mean(-ddxupvp[:,xr]+hdiff_v_avg[:,xr], axis=-1)
pressv_savg = np.mean(pres_v_avg[:,xr], axis=-1)
corv_savg = np.mean(cor_v_avg[:,xr], axis=-1)

#plot hovmoller for u tendency terms
plt.figure()

xplot = [xs,x,xs[1:-1],x,xs[1:-1],xs,xs,xs,xs]
tplot = [t[1:-1],t,t,t,t,t,t,t,t]
tendplot = [dvdt_savg, -udvdx_savg, -wdvdz_savg,  vdiff_savg, hdiff_savg, pressv_savg, corv_savg]
leg = ['dvdt', 'udvdx', 'wdvdz', 'VDIFF', 'HDIFF', 'PRESS', 'COR']
for n in range(7):
    plt.plot(tplot[n], np.squeeze(tendplot[n]), label=leg[n])
    
#total = -ududx_savg -wdudz_savg+ vdiff_savg+ hdiff_savg+ pressu_savg+ coru_savg 
##total = advw_savg + advu_savg+ vdiff_savg+ hdiff_savg+ pressu_savg+ coru_savg 
#
#plt.plot(tplot[-1], total, linestyle=':', label='total')    
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


#%%
cl = 0.0005
fsizex = 20
fsizey = 1
fudvdx= ndimage.uniform_filter(udvdx,mode='wrap',size=(fsizey,fsizex))
plt.figure()
plt.pcolormesh(-fudvdx)
plt.clim((-cl, cl))
plt.colorbar()
#plot hovmoller for w tendency terms
# plt.figure(figsize=(18,18))

