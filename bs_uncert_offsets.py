#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:14:16 2022

@author: amartinez
"""

import numpy as np
import scipy as sp
import pylab
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import glob
import os
import sys
from scipy.stats import binned_statistic_2d
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.table import vstack
from astropy.table import unique
from matplotlib import colors
# %%plotting parametres
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 20})
rcParams.update({'figure.figsize':(10,5)})
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})# a warniing for matplot lib pop up because so many plots, this turining it of
# %%

# field_one = 'B1'
# field_two = 20
field_one = 16
field_two = 7
survey = 1#TODO ONLY for Gaia. Can be 1 or 2


# pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2off/pruebas/'
bs1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1absolute_SUPER/bootstrapping/'
bs2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2absolute_SUPER/bootstrapping/'
# if surev
# uncer_folder= bs1
# %%
# ref_fr = 'gns'#TODO
color_map ='Greys'
dmax = 2
max_sig= 0.5
degree = 1
resam = 100
for degree in range(degree,degree+1):
                   # /Users/amartinez/Desktop/PhD/HAWK/GNS_1relative/lists/7/uncert_lists/chip4/BS_lists_degre1/
    boot_folder = f'/Users/amartinez/Desktop/PhD/HAWK/GNS_{survey}absolute_SUPER/bootstrapping/'
   
    it=len(glob.glob(boot_folder+f'BS*_gns{survey}_Super_F1_{field_one}_F2_{field_two}.txt')) 
    print(it)
   
    line =[]
    # skip ={113,232,283,284,293,299}
    # skip ={44}
    skip ={}
    # for i in range(1,it+1):
    for i in range(1,100):   
        # print(i)
        
        # if i in skip:
        #     continue
        # xi, yi,RaH1,DecH1, ID
        
        f = Table.read(boot_folder +f'BS{i}_gns{survey}_Super_F1_{field_one}_F2_{field_two}.txt', format = 'ascii')
        # if len(f) != 4773 or len(f) != len(unique(f)):
        #     print('Chunga', i, len(f))
        #     continue
        
        line.append(f)
    # %
    ar = vstack(line)
   
    # ar = line[0]
    # for l in range(1,len(line)):
    #     ar = np.r_[ar,line[l]]
    # %
    sr = list(set(ar['ID']))
    print(sr[1],len(sr))  
    
    # %
# =============================================================================
#     uncer_pos =np.empty((len(sr),6))
#     for ind in range(len(sr)):
#         
#         data = np.where(ar['ID']==sr[ind])
#         uncer_pos[ind][0],uncer_pos[ind][1] = np.mean(ar[data]['xp']),np.mean(ar[data]['yp'])
#         # uncer_pos[ind][2],uncer_pos[ind][3] = np.std(ar[data][:,0])*np.sqrt(len(data[0])-1), np.std(ar[data][:,1]*np.sqrt(len(data[0])-1))
#         uncer_pos[ind][2],uncer_pos[ind][3] = np.std(ar[data]['xp']), np.std(ar[data]['yp'])
#         uncer_pos[ind][4],uncer_pos[ind][5] = np.mean(ar[data]['l']),np.mean(ar[data]['b'])
#     
# =============================================================================
# =============================================================================
#     uncer_pos = []
# 
#     for ind in range(len(sr)):
#         data = np.where(ar['ID'] == sr[ind])
#         
#         xm, ym = np.mean(ar[data]['xp']), np.mean(ar[data]['yp'])
#         xs, ys = np.std(ar[data]['xp']), np.std(ar[data]['yp'])
#         lm, bm = np.mean(ar[data]['l']), np.mean(ar[data]['b'])
#         
#         # only keep if ALL values are <= 20
#         if all(val <=  100 for val in [ xs, ys]):
#             uncer_pos.append([xm, ym, xs, ys, lm, bm])
#     
#     uncer_pos = np.array(uncer_pos)
# 
#    
# =============================================================================
    uncer_pos = []

    for ind in range(len(sr)):
        data = np.where(ar['ID'] == sr[ind])
        
        xm, ym = np.mean(ar[data]['xp']), np.mean(ar[data]['yp'])
        xs = np.std(sigma_clip(ar[data]['xp'], sigma=5))
        ys = np.std(sigma_clip(ar[data]['yp'], sigma=5))
        lm, bm = np.mean(ar[data]['l']), np.mean(ar[data]['b'])
        #         
        
        # keep only if all values ≤ 20
        if all(val <=1 for val in [xs, ys]):
            uncer_pos.append([xm, ym, xs, ys, lm, bm])
    
    uncer_pos = np.array(uncer_pos)

    uncer_pos[:,2] = uncer_pos[:,2]*1000
    uncer_pos[:,3] = uncer_pos[:,3]*1000
    # colors =uncer_pos[:,2]
# %%
    # fig, ax = plt.subplots(1,1,figsize = (10,10))
    
    # # ax.set_title('Degree = %s. Ref. frame : %s'%(degree,ref_fr))
    # # ax.set_title('GNS1f%sc%s, Degree = %s. Survey = %s. loops = %s'%(field_one,chip_one,degree, survey,i))
    # # im = ax.scatter(uncer_pos[:,4],uncer_pos[:,5], c=np.sqrt((uncer_pos[:,2])**2+(uncer_pos[:,3])**2),
    # #                 norm=colors.LogNorm(vmin =0.0001*1000, vmax = 0.008*1000),s=100, cmap = 'Greys')
    # im = ax.scatter(uncer_pos[:,4],uncer_pos[:,5], c=np.sqrt((uncer_pos[:,2])**2+(uncer_pos[:,3])**2),
    #               s=100,norm=colors.LogNorm(), cmap = 'Greys')
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # fig.colorbar(im, label = 'mas')
    # plt.show()
# %%
    # color_map ='Set1'
    # fig, ax = plt.subplots(1,1,figsize = (10,10))
    
    # # ax.set_title('Degree = %s. Resample = %s %%. Dmax = %s'%(degree,resam,dmax))
    # # ax.set_title('GNS1f%sc%s, Degree = %s. Survey = %s. loops = %s'%(field_one,chip_one,degree, survey,i))
    # # im = ax.scatter(uncer_pos[:,4],uncer_pos[:,5], c=np.sqrt((uncer_pos[:,2])**2+(uncer_pos[:,3])**2),
    # #                 norm=colors.LogNorm(vmin = 4, vmax = 10),s=100, cmap = color_map)
    # im = ax.scatter(uncer_pos[:,4],uncer_pos[:,5], c=np.sqrt((uncer_pos[:,2])**2+(uncer_pos[:,3])**2),
    #                 s=100, cmap = color_map)
    
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # fig.colorbar(im, label='arcsec?')
    # ax.set_xlabel('RA')
    # ax.set_ylabel('DEC')
# %%
    
   
    fig, ax = plt.subplots(1,1,figsize = (10,10))
    num_bins = (10,20)
    unce_value = (uncer_pos[:,2] + uncer_pos[:,3])/2
    unce_value = (uncer_pos[:,2]**2 + uncer_pos[:,3]**2)**0.5
    
    statistic, x_edges, y_edges, binnumber = binned_statistic_2d(uncer_pos[:,4],uncer_pos[:,5],
                                                                 unce_value , 
                                                                 statistic='mean', bins=(num_bins))
    X, Y = np.meshgrid(x_edges, y_edges)
    # im = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',norm=colors.LogNorm())
    im = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r')
    
    # ax.set_title('alignment uncertainty')
    # im = ax.hist2d(uncer_pos[:,4],uncer_pos[:,5], c=np.sqrt((uncer_pos[:,2])**2+(uncer_pos[:,3])**2),
    #               s=100,norm=colors.LogNorm(), cmap = 'Greys')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('l')
    ax.set_ylabel('b')
    ax.invert_xaxis()

    
    cbar = fig.colorbar(im, label = '$\overline{\sigma}_{(l,b)}$ [mas]', fraction = 0.0218 )
    # cbar.ax.set_yticklabels([])
    # cbar.ax.set_yticklabels([0.5,'',1, 2])
    cbar.ax.tick_params(size=8,width=1,direction='out')
    
    # cbar.ax.set_yticklabels([0.5,1, 2])
    ax.axis('scaled')
    fig.tight_layout()
    
    meta = {'scrip':'/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_absolute_SUPER/bs_uncert_offsets.py'}
    article = '/Users/amartinez/Desktop/PhD/My_papers/GNS_pm_catalog/images/'
    # plt.savefig(article + 'alig_error.png', bbox_inches='tight', transparent = 'True', dpi = 150, metadata = meta)
    plt.show()
    
    
# statistic, x_edges, y_edges, binnumber = binned_statistic_2d(x, y, vx*-1, statistic='median', bins=(num_bins))
# # statistic, x_edges, y_edges, binnumber = binned_statistic_2d(x_, y_, vx_, statistic='median', bins=(num_bins))
# %%

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, (ax,ax2) = plt.subplots(1,2,figsize = (20,10))
    num_bins = 49
    statistic, x_edges, y_edges, binnumber = binned_statistic_2d(uncer_pos[:,4],uncer_pos[:,5],
                                                                 (uncer_pos[:,2]) , 
                                                                 statistic='mean', bins=(num_bins))
    
    new_ax = make_axes_locatable(ax)# make a new axes ling with ax2 axis
    cax = new_ax.append_axes("right", size = '5%', pad = 0.05)
    X, Y = np.meshgrid(x_edges, y_edges)
    # im = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',norm=colors.LogNorm())
    im = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r')
    
    ax.set_title('$\delta l$')
    # im = ax.hist2d(uncer_pos[:,4],uncer_pos[:,5], c=np.sqrt((uncer_pos[:,2])**2+(uncer_pos[:,3])**2),
    #               s=100,norm=colors.LogNorm(), cmap = 'Greys')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('l')
    ax.set_ylabel('b')
    ax.invert_xaxis()
    # fig.colorbar(im, label = 'Alignment uncertainty [arcsec]', ax = cax)
    plt.colorbar(im, cax=cax )
    ax.axis('scaled')
    
    statistic, x_edges, y_edges, binnumber = binned_statistic_2d(uncer_pos[:,4],uncer_pos[:,5],
                                                                 (uncer_pos[:,3]) , 
                                                                 statistic='mean', bins=(num_bins))
    
    new_ax = make_axes_locatable(ax2)# make a new axes ling with ax2 axis
    cax2 = new_ax.append_axes("right", size = '5%', pad = 0.05)
    X, Y = np.meshgrid(x_edges, y_edges)
    # im = ax2.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',norm=colors.LogNorm())
    im = ax2.pcolormesh(X, Y, statistic.T, cmap='Spectral_r')
    
    ax2.set_title('$\delta b$')
    # im = ax.hist2d(uncer_pos[:,4],uncer_pos[:,5], c=np.sqrt((uncer_pos[:,2])**2+(uncer_pos[:,3])**2),
    #               s=100,norm=colors.LogNorm(), cmap = 'Greys')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_xlabel('l')
    # ax2.set_ylabel('b')
    ax2.invert_xaxis()
    # fig.colorbar(im, label = 'Alignment uncertainty [arcsec]', ax = cax)
    plt.colorbar(im, cax=cax2, label = 'Alignment uncertainty [mas]',)

    ax2.axis('scaled')
    fig.tight_layout()
# %%

    fig, ax = plt.subplots(1,1,figsize = (7,7))
    num_bins = (17,24)
    unce_value = (uncer_pos[:,2] + uncer_pos[:,3])/2
    # unce_value = (uncer_pos[:,2]**2 + uncer_pos[:,3]**2)**0.5
    
    statistic, x_edges, y_edges, binnumber = binned_statistic_2d(uncer_pos[:,4],uncer_pos[:,5],
                                                                 unce_value , 
                                                                 statistic='mean', bins=(num_bins))
   
    X, Y = np.meshgrid(x_edges, y_edges)
    # im = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',norm=colors.LogNorm())
    # im = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',vmin = 0.5, vmax = 3.505 )
    im = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r')
    # im = ax.pcolormesh(X, Y, statistic.T, cmap='Set1_r', vmin = 0.5, vmax = 3.5)
    # im = ax.pcolormesh(X, Y, statistic.T, cmap='Set1_r')
    
    # ax.set_title('alignment uncertainty')
    # im = ax.hist2d(uncer_pos[:,4],uncer_pos[:,5], c=np.sqrt((uncer_pos[:,2])**2+(uncer_pos[:,3])**2),
    #               s=100,norm=colors.LogNorm(), cmap = 'Greys')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('l')
    ax.set_ylabel('b')
    ax.invert_xaxis()

    
    # cbar = fig.colorbar(im, label = '$\overline{\sigma}_{(l,b)}$ [mas]',fraction = 0.0218)
    cbar = fig.colorbar(im, label = '$\overline{\sigma}_{(l,b)}$ [mas]',aspect = 30, pad = -0.1)
    # cbar.ax.set_yticklabels([])
    # cbar.ax.set_yticklabels([0.5,'',1, 2])
    cbar.ax.tick_params(size=8,width=1,direction='out')
    
    # cbar.ax.set_yticklabels([0.5,1, 2])
    ax.axis('scaled')
    pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1absolute_SUPER/pruebas/'
    gaia_ref = Table.read(pruebas1  + f'gaia_refstars_F{field_one}_F{field_two}.txt', format = 'ascii')
    ax.scatter(gaia_ref['l'], gaia_ref['b'], marker = '*', facecolor = 'none', color = 'k', s =1000,lw = 2, label = 'Gaia')
    # ax.legend(loc =1, markerscale = 1)
    
    # ax.tick_params(axis='both', which='major', labelsize=30)  # major ticks
    # ax.tick_params(axis='both', which='minor', labelsize=30)  # minor ticks
    # 
    fig.tight_layout()
    
    meta = {'scrip':'/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_absolute_SUPER/bs_uncert_offsets.py'}
    article = '/Users/amartinez/Desktop/PhD/My_papers/GNS_pm_catalog/images/'
    # plt.savefig(article + 'ABS_alig_error.png', bbox_inches='tight', transparent = 'True', dpi = 150, metadata = meta)
    plt.savefig(article + 'ABS_arches_alig_error.png', bbox_inches='tight', transparent = 'True', dpi = 150, metadata = meta)
    plt.show()
    
    sys.exit(294)
# %%
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(1,1,figsize = (10,10))
    num_bins = 50,25
    statistic, x_edges, y_edges, binnumber = binned_statistic_2d(uncer_pos[:,4],uncer_pos[:,5],
                                                                 (uncer_pos[:,2]) , 
                                                                 statistic='mean', bins=(num_bins))
    
    new_ax = make_axes_locatable(ax)# make a new axes ling with ax2 axis
    cax = new_ax.append_axes("right", size = '5%', pad = 0.05)
    X, Y = np.meshgrid(x_edges, y_edges)
    im = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r')
    # im = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',norm=colors.LogNorm())
    # im = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',norm=colors.SymLogNorm(linthresh= 1, linscale=1, vmin=0, vmax=2))

    # im = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r')
    # im = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r', vmin = 0, vmax = 3)
    
    ax.set_title('$\delta l$')
    # im = ax.hist2d(uncer_pos[:,4],uncer_pos[:,5], c=np.sqrt((uncer_pos[:,2])**2+(uncer_pos[:,3])**2),
    #               s=100,norm=colors.LogNorm(), cmap = 'Greys')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('l')
    ax.set_ylabel('b')
    ax.invert_xaxis()
    # fig.colorbar(im, label = 'Alignment uncertainty [arcsec]', ax = cax)
    plt.colorbar(im, cax=cax )
    ax.axis('scaled')

 
































