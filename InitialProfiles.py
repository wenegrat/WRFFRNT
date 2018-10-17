#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:10:24 2018

@author: jacob
"""

######################################################
#
# Script to plot initial t, u, v profiles, domain averaged
# JW 10-18-18
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
dirall = '/media/ExtDriveFolder/WRFRUNS/WeakRun1'
wrfout = ['/wrfout_d01_0001-01-02_']
hrs = ['11']
mins = ['00']

nt = 0





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
    
    #budget variables
    if n==0:
        u = data.variables['U'][:,:,:,:]
        v = data.variables['V'][:,:,:,:]
        t = data.variables['T'][:,:,:,:]
        
#%%
uavg = np.mean(np.mean(u, axis=-1), axis=-1)
vavg = np.mean(np.mean(v, axis=-1), axis=-1)
tavg = np.mean(np.mean(t, axis=-1), axis=-1)

#%%
ti = 0

fig, ax = plt.subplots(1,2,sharey=True, figsize=(6, 6))

ax[0].plot(tavg[ti,:]+300, z[0,:,0,0])
ax[0].grid()
ax[0].set_ylabel('z [m]')
ax[0].set_ylim((0, 2000))
ax[0].set_xlim((298, 304))
ax[0].set_xticks([298, 300, 302, 304])
ax[0].set_xlabel('$^\circ$ K')

ax[1].plot(uavg[ti,:], z[0,:,0,0], label='$U_o$')
ax[1].plot(vavg[ti,:], z[0,:,0,0], label='$V_o$')
ax[1].grid()
ax[1].legend()
ax[1].set_xlim((0, 6))
ax[1].set_xticks([0, 2, 4, 6])
ax[1].set_xlabel('m s$^{-1}$')
plt.tight_layout()