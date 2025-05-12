#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:49:42 2024

@author: amartinez
"""

import numpy as np
from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from scipy.stats import sigmaclip
from astropy.stats import sigma_clip
from matplotlib import rc
from matplotlib import rcParams
import IPython
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
# Enable automatic plotting mode
# IPython.get_ipython().run_line_magic('matplotlib', 'auto')
IPython.get_ipython().run_line_magic('matplotlib', 'inline')

# field_one, chip_one, field_two, chip_two,t1,t2,max_sig = np.loadtxt('/Users/amartinez/Desktop/PhD/HAWK/GNS_1absolute_python/lists/fields_and_chips.txt', 
#                                                        unpack=True)
# field_one = field_one.astype(int)
# chip_one = chip_one.astype(int)
# field_two = field_two.astype(int)
# chip_two = chip_two.astype(int)

def diff_hist(survey,l1_x,l2_x,l1_y,l2_y,sig_cl, gaia_all, gaia_ind,
              align_by = None,clipping = None, variable = None,
              field_one = None , chip_one = None, field_two = None, chip_two = None):
    diff_x = l1_x - l2_x
    diff_y = l1_y - l2_y
    
    leg_x = '$\overline{\Delta x}$'
    leg_y = '$\overline{\Delta y}$'
    lab_x = '$\Delta x$'
    lab_y = '$\Delta y$'

    if variable is not None:
        if variable == 'coordinates':
            leg_x = '$\overline{\Delta RA}$'
            leg_y = '$\overline{\Delta Dec}$'
            lab_x = '$\Delta RA$ [mas/yr]'
            lab_y = '$\Delta Dec$ [mas/yr]'
        if variable == 'pm':
            leg_x = '$\overline{\Delta \mu_{RA}}$'
            leg_y = '$\overline{\Delta \mu_{Dec}}$'
            lab_x = '$\Delta \mu_{RA} [mas/yr]$'
            lab_y = '$\Delta \mu_{Dec} [mas/yr]$'
            if align_by is not None:
                lab_alig = align_by

    # sig_cl = 3#!!!
    mask_x, lx_lim,hx_lim = sigma_clip(diff_x, sigma=sig_cl, masked = True, return_bounds= True)
    mask_y, ly_lim,hy_lim = sigma_clip(diff_y, sigma=sig_cl, masked = True, return_bounds= True)
 
   # mask_xy = mask_x & mask_y # mask_xy = np.logical(mx, my)
    mask_xy = np.logical_and(np.logical_not(mask_x.mask), np.logical_not(mask_y.mask))


    
    
    mal_xy = np.logical_not(mask_xy)
    # mal_ind = gaia_all['gaia1_id'][gaia_ind][mal_xy]
    fig, ax = plt.subplots(1,1)
    plt.suptitle('GAIA & \nGNS%s [F%sC%s], GNS2 [F%sC%s]'%( survey, field_one, chip_one,field_two,chip_two))

    
    try:
        mal_ind = gaia_all['gaia1_id'][gaia_ind][mal_xy]
    except:
        mal_ind = gaia_all['gaia2_id'][gaia_ind][mal_xy]
        
    ax.scatter(diff_x, diff_y)
    for i, (x, y) in enumerate(zip(diff_x[mal_xy], diff_y[mal_xy])):
        ax.annotate(mal_ind[i], (x, y), textcoords="offset points", xytext=(5, 5), ha='center')
    ax.axvline(lx_lim, ls = 'dashed', color = 'r')
    ax.axvline(hx_lim, ls = 'dashed', color = 'r')
    ax.axhline(ly_lim, ls = 'dashed', color = 'r')
    ax.axhline(hy_lim, ls = 'dashed', color = 'r')
    ax.set_xlabel(lab_x)
    ax.set_ylabel(lab_y)
    ax.grid()
    ax.axis('equal')
    
    fig, (ax,ax1) = plt.subplots(1,2)
    plt.suptitle('GAIA & GNS%s [F%sC%s], GNS2 [F%sC%s]'%( survey, field_one, chip_one,field_two,chip_two))
    ax.set_title(f'Matching stars = {len(diff_x)}')
    ax1.set_title(f'Aligend by = {align_by}')
    ax.hist(diff_x, histtype = 'step', color ='#1f77b4', 
            label = '%s = %.2f\n$\sigma = %.2f$'%(leg_x, np.mean(diff_x),np.std(diff_x)) )
    ax.axvline(lx_lim, ls = 'dashed', color = 'r', lw = 1)
    ax.axvline(hx_lim, ls = 'dashed', color = 'r', lw = 1)
    ax1.hist(diff_y, histtype = 'step', color ='#ff7f0e', 
            label = '%s = %.2f\n$\sigma = %.2f$'%(leg_y, np.mean(diff_y),np.std(diff_y)) )
    ax1.axvline(ly_lim, ls = 'dashed', color = 'r', lw = 1)
    ax1.axvline(hy_lim, ls = 'dashed', color = 'r', lw = 1)
    
    
    if np.all(mask_xy) == False:
        diff_xm = diff_x[mask_xy]    
        diff_ym = diff_y[mask_xy]    
        ax.hist(diff_xm, color ='k', alpha = 0.5,
                label = '%s = %.2f\n$\sigma = %.2f$'%(leg_x,np.mean(diff_xm),np.std(diff_xm)) )
        ax1.hist(diff_ym, color ='k', alpha = 0.5,
                label = '%s = %.2f\n$\sigma = %.2f$'%(leg_y, np.mean(diff_ym),np.std(diff_ym)) )

    ax.legend()
    ax1.legend()
    ax.set_xlabel(lab_x)
    ax1.set_xlabel(lab_y)
    print(mal_ind)
    
    return [mal_ind.value]