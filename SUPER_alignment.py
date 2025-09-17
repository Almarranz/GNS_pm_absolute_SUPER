#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:19:45 2022

@author: amartinez
"""

# Generates the GNS1 second reduction with the Ks and H magnitudes
import sys
sys.path.append("/Users/amartinez/Desktop/pythons_imports/")
import numpy as np
import matplotlib.pyplot as plt
import astroalign as aa
from astropy.io.fits import getheader
from astropy.io import fits
from scipy.spatial import distance
import pandas as pd
import sys
import time
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astroquery.gaia import Gaia
from astropy.stats import sigma_clip
from filters import filter_gaia_data
import skimage as ski
from astropy.table import unique
from astropy.coordinates import search_around_sky
from collections import Counter
from matplotlib.colors import LogNorm
from scipy import stats
from astropy.table import Table, vstack

# %% 
# %%plotting parametres
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
# Enable automatic plotting mode
import IPython
# IPython.get_ipython().run_line_magic('matplotlib', 'auto')
IPython.get_ipython().run_line_magic('matplotlib', 'inline')
# %%
def filter_gns_by_percentile(gns_table, mag_col='H', err_col='dH', sl_col='sl', sb_col='sb', bin_width=0.5, percentile=85, mag_lim = None, pos_lim = None):
    """
    Filter stars in magnitude bins based on the 85th percentile of photometric and positional uncertainties.

    Parameters
    ----------
    gns_table : astropy.table.Table
        The input table with columns for magnitude, uncertainty, and position errors.
    mag_col : str
        Name of the magnitude column (e.g., 'H').
    err_col : str
        Name of the magnitude uncertainty column (e.g., 'dH').
    sl_col, sb_col : str
        Names of the positional uncertainty columns.
    bin_width : float
        Width of the magnitude bins.
    percentile : float
        Percentile cutoff (default is 85).

    Returns
    -------
    filtered_table : astropy.table.Table
        Table with rows that pass the percentile thresholding.
    """
    # Assuming gns1 is your input Table
    H = gns_table['H']
    pos_err = np.sqrt(gns_table['sl']**2 + gns_table['sb']**2)

    # Add temporary column for position uncertainty
    gns_table['pos_err'] = pos_err

    # Define bins
    H_min, H_max = np.nanmin(H), np.nanmax(H)
    bins = np.arange(H_min, H_max + bin_width, bin_width)

    # Container for filtered results
    filtered_tables = []

    # Iterate over H-magnitude bins
    for i in range(len(bins)-1):
        bin_mask = (H >= bins[i]) & (H < bins[i+1])
        bin_data = gns_table[bin_mask]

        if len(bin_data) == 0:
            continue

        # Compute 85th percentiles
        dH_thresh = np.percentile(bin_data['dH'], percentile)
        pos_thresh = np.percentile(bin_data['pos_err'], percentile)

        # Apply selection criteria
        good_mask = (bin_data['dH'] <= dH_thresh) & (bin_data['pos_err'] <= pos_thresh)
        filtered_tables.append(bin_data[good_mask])
    
    
    filtered_table = vstack(filtered_tables)
    
    if mag_lim is not None:
        mH = filtered_table['H'] < mag_lim
        filtered_table = filtered_table[mH]
    if pos_lim is not None:
        
        unc_cut = np.where((filtered_table['sl']<pos_lim) & (filtered_table['sb']<pos_lim))
        filtered_table  = filtered_table[unc_cut]
        
    
    return filtered_table
    

#%%
field_one = 10
chip_one = 0
field_two = 4
chip_two = 0
# field_one = 'B1'
# chip_one = 0
# field_two = 20
# chip_two = 0




if field_one == 7 or field_one == 12 or field_one == 10 or field_one == 16:
    t1_gns = Time(['2015-06-07T00:00:00'],scale='utc')
elif field_one == 'B6':
    t1_gns = Time(['2016-06-13T00:00:00'],scale='utc')
elif field_one ==  'B1':
    t1_gns = Time(['2016-05-20T00:00:00'],scale='utc')
else:
    print(f'NO time detected for this field_one = {field_one}')
    sys.exit()
if field_two == 7 or field_two == 5:
    t2_gns = Time(['2022-05-27T00:00:00'],scale='utc')
elif field_two == 4:
    t2_gns = Time(['2022-04-05T00:00:00'],scale='utc')
elif field_two == 20:
    t2_gns = Time(['2022-07-25T00:00:00'],scale='utc')
else:
    print(f'NO time detected for this field_two = {field_two}')
    sys.exit()

dt_gns = t2_gns - t1_gns


# transf = 'affine'#!!!
# transf = 'similarity'#!!!1|
transf = 'polynomial'#!!!
# transf = 'shift'#!!!
order_trans = 1
# mlim = 22
gns_mags = [12,20]

max_sig = 0.05
# max_sig = 0.
perc = 100
bin_width = 1

# Arches and Quintuplet coordinates for plotting and check if it will be covered.
# Choose Arches or Quituplet central coordinates #!!!
# arch = SkyCoord(ra = '17h45m50.65020s', dec = '-28d49m19.51468s', equinox = 'J2000').galactic
# arch_ecu = SkyCoord(ra = '17h45m50.65020s', dec = '-28d49m19.51468s', equinox = 'J2000')
arch =  SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs',obstime ='J2016.0').galactic#Quintuplet
arch_ecu =  SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs',obstime ='J2016.0')#Quintuplet


# GNS_1='/Users/amartinez/Desktop/PhD/HAWK/GNS_1/lists/%s/chip%s/'%(field_one, chip_one)
GNS_1='/Users/amartinez/Desktop/PhD/HAWK/GNS_1/lists/%s/chip%s/'%(field_one, chip_one)
GNS_2='/Users/amartinez/Desktop/PhD/HAWK/GNS_2/lists/%s/chip%s/'%(field_two, chip_two)

pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1absolute_SUPER/pruebas/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2absolute_SUPER/pruebas/'


# gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_one}/{field_one}_H_chips_opti.ecsv',  format = 'ascii.ecsv')
gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_one}_old/{field_one}_H_chips_opti.ecsv',  format = 'ascii.ecsv')

fig, (ax, ax2) = plt.subplots(1,2)
ax.hist2d(gns1['H'],gns1['sl'], bins = 100,norm = LogNorm())
his = ax2.hist2d(gns1['H'],gns1['sb'], bins = 100,norm = LogNorm())
fig.colorbar(his[3], ax =ax2)
ax.set_title('GNS1')
ax.set_ylabel('$\delta l$ [arcsec]')
ax2.set_ylabel('$\delta b$ [arcsec]')
ax.set_xlabel('[H]')
ax2.set_xlabel('[H]')
fig.tight_layout()
ax.axhline(max_sig,ls = 'dashed', color = 'r')
ax.axvline(gns_mags[1],ls = 'dashed', color = 'r')

# %%
gns1_fil = filter_gns_by_percentile(gns1, mag_col='H', err_col='dH', sl_col='sl', sb_col='sb', bin_width=bin_width, percentile=perc, mag_lim = gns_mags[1])

if perc <99:
    ax.scatter(gns1_fil['H'],gns1_fil['sl'],s =0.1,color = 'k', alpha = 0.01)
    gns1 = filter_gns_by_percentile(gns1, mag_col='H', err_col='dH', sl_col='sl', sb_col='sb', bin_width=bin_width, percentile=perc, mag_lim = gns_mags[1], pos_lim = max_sig)

m1_mask = (gns1['H']>gns_mags[0]) & (gns1['H']<gns_mags[1])
gns1 = gns1[m1_mask]
unc_cut1 = (gns1['sl']<max_sig) & (gns1['sb']<max_sig)
gns1 = gns1[unc_cut1]
# %%
# m_mask = gns1['H']<mlim

# gns1 = gns1[m_mask]
# unc_cut = np.where((gns1['sl']<max_sig) & (gns1['sb']<max_sig))
# gns1 = gns1[unc_cut]

gns1_gal = SkyCoord(l = gns1['l'], b = gns1['b'], 
                    unit = 'degree', frame = 'galactic')

num_bins = 100
statistic, x_edges, y_edges, binnumber = stats.binned_statistic_2d(gns1['l'], gns1['b'], np.sqrt(gns1['sl']**2 + gns1['sb']**2), statistic='median', bins=(num_bins,int(num_bins/2)))
# Create a meshgrid for plotting
X, Y = np.meshgrid(x_edges, y_edges)
# Plot the result
fig, ax = plt.subplots()
# c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r', norm = LogNorm()) 
c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',vmax = 0.1) 
fig.colorbar(c, ax=ax, label='$\sqrt {\delta l^{2} + \delta b^{2}}$ [arcsec]', shrink = 1)
ax.set_title(f'GNS1 Max $\delta$ posit = {max_sig}. Max mag = {gns_mags[1]}')
ax.set_xlabel('l')
ax.set_ylabel('b')
ax.axis('equal')
# sys.exit(213)

# %%

# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_two}/{field_two}_H_chips_opti.ecsv', format = 'ascii.ecsv')
gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_two}_old/{field_two}_H_chips_opti.ecsv', format = 'ascii.ecsv')


fig, (ax, ax2) = plt.subplots(1,2)
ax.set_title('GNS2')
ax.hist2d(gns2['H'],gns2['sl'],cmap = 'inferno', bins = 100,norm = LogNorm())
his = ax2.hist2d(gns2['H'],gns2['sb'],cmap = 'inferno', bins = 100,norm = LogNorm())
fig.colorbar(his[3], ax =ax2)
ax.set_ylabel('$\delta l$ [arcsec]')
ax2.set_ylabel('$\delta b$ [arcsec]')
ax.set_xlabel('[H]')
ax2.set_xlabel('[H]')
fig.tight_layout()
ax.axhline(max_sig,ls = 'dashed', color = 'r')
ax.axvline(gns_mags[1],ls = 'dashed', color = 'r')

m2_mask = (gns2['H']>gns_mags[0]) & (gns2['H']<gns_mags[1])
gns2 = gns2[m2_mask]
unc_cut2 = (gns2['sl']<max_sig) & (gns2['sb']<max_sig)
gns2 = gns2[unc_cut2]

# %%
gns2_fil = filter_gns_by_percentile(gns2, mag_col='H', err_col='dH', sl_col='sl', sb_col='sb', bin_width= bin_width, percentile=perc, mag_lim =gns_mags[1], pos_lim=max_sig)

if perc <99:
    ax.scatter(gns2_fil['H'],gns2_fil['sl'],s =0.1,color = 'k', alpha = 0.01)
    gns2 = filter_gns_by_percentile(gns2, mag_col='H', err_col='dH', sl_col='sl', sb_col='sb', bin_width=0.5, percentile=perc, mag_lim = gns_mags[1], pos_lim=max_sig)




# unc_cut2 = np.where((gns2['sl']<max_sig) & (gns2['sb']<max_sig))
# gns2 = gns2[unc_cut2]
# m_mask = gns2['H']<mlim
# gns2 = gns2[m_mask]

gns2_gal = SkyCoord(l = gns2['l'], b = gns2['b'], 
                    unit = 'degree', frame = 'galactic')


num_bins = 100
statistic, x_edges, y_edges, binnumber = stats.binned_statistic_2d(gns2['l'], gns2['b'], np.sqrt(gns2['sl']**2 + gns2['sb']**2), statistic='median', bins=(num_bins,int(num_bins/2)))
# Create a meshgrid for plotting
X, Y = np.meshgrid(x_edges, y_edges)
# Plot the result
fig, ax = plt.subplots()
# c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',norm = LogNorm())
c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r', vmax= 0.06)
fig.colorbar(c, ax=ax, label='$\sqrt {\delta l^{2} + \delta l^{2}}$', shrink = 1)
ax.set_title(f'GNS2 Max $\delta$ posit = {max_sig}. Max mag = {gns_mags[1]}')
ax.set_xlabel('l')
ax.set_ylabel('b')
ax.axis('equal')


l2 = gns2_gal.l.wrap_at('360d')
l1 = gns1_gal.l.wrap_at('360d')
fig, ax = plt.subplots(1,1,figsize =(5,5))
ax.scatter(l1[::10], gns1['b'][::10],label = 'GNS_1 Fied %s, chip %s'%(field_one,chip_one),zorder=3)
ax.scatter(l2[::10], gns2['b'][::10],label = 'GNS_2 Fied %s, chip %s'%(field_two,chip_two))

ax.invert_xaxis()
ax.legend()
ax.set_xlabel('l[deg]', fontsize = 10)
ax.set_ylabel('b [deg]', fontsize = 10)
ax.axis('scaled')



# f_work = '/Users/amartinez/Desktop/PhD/Thesis/document/mi_tesis/tesis/Future_work/'
# plt.savefig(f_work+ 'gsn1_gns2_fields.png', bbox_inches='tight')




# Select only overlapping areas
# =============================================================================
# buenos1 = np.where((gns1_gal.l>min(gns2_gal.l)) & (gns1_gal.l<max(gns2_gal.l)) &
#                    (gns1_gal.b>min(gns2_gal.b)) & (gns1_gal.b<max(gns2_gal.b)))
# 
# gns1 = gns1[buenos1]
# gns1['ID'] = np.arange(len(gns1))
# 
# buenos2 = np.where((gns2_gal.l>min(gns1_gal.l)) & (gns2_gal.l<max(gns1_gal.l)) &
#                    (gns2_gal.b>min(gns1_gal.b)) & (gns2_gal.b<max(gns1_gal.b)))
# 
# gns2 = gns2[buenos2]
# gns2['ID'] = np.arange(len(gns2))
# 
# l1 = l1[buenos1]
# l2 = l2[buenos2]
# =============================================================================


gns1_c = SkyCoord(l = gns1['l'], b = gns1['b'], 
                    unit = 'degree', frame = 'galactic')
gns2_c = SkyCoord(l = gns2['l'], b = gns2['b'], 
                    unit = 'degree', frame = 'galactic')

# Driect alignemnet
max_sep = 100*u.mas#!!!

matches = 0
for i in range(0):
    # idx,d2d,d3d = gns2_c.match_to_catalog_sky(gns1_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
    # sep_constraint = d2d < max_sep
    # gns2_m = gns2[sep_constraint]
    # gns1_m = gns1[idx[sep_constraint]]
    
    # print(f'MATCHES i = {i-1} = {len(gns2_m)}')
    max_sep_fine = 30*u.mas
    idx1, idx2, sep2d, _ = search_around_sky(gns1_c, gns2_c, max_sep_fine)

    count1 = Counter(idx1)
    count2 = Counter(idx2)

    # Step 3: Create mask for one-to-one matches only
    mask_unique = np.array([
        count1[i1] == 1 and count2[i2] == 1
        for i1, i2 in zip(idx1, idx2)
    ])

    # Step 4: Apply the mask
    idx1_clean = idx1[mask_unique]
    idx2_clean = idx2[mask_unique]

    gns1_m= gns1[idx1_clean]
    gns2_m = gns2[idx2_clean]
    
    
    diff_H = gns2_m['H']-gns1_m['H']
    off_s = np.mean(diff_H)
    gns1_m['H'] = gns1_m['H'] + off_s
    diff_H = gns2_m['H'] - gns1_m['H']
    
    diff_H = gns2_m['H']-gns1_m['H']
    sig_cl = 3
    mask_H, l_lim,h_lim = sigma_clip(diff_H, sigma=sig_cl, masked = True, return_bounds= True)
    
    gns2_m = gns2_m[mask_H.mask]
    gns1_m = gns1_m[mask_H.mask]
    
    fig,ax = plt.subplots(1,1)
    ax.hist(diff_H, bins = 'auto',histtype = 'step')
    ax.axvline(np.mean(diff_H), color = 'k', ls = 'dashed', label = 'H offset = %.2f\n$\sigma$ = %.2f'%(off_s,np.std(diff_H)))
    ax.axvline(l_lim, ls = 'dashed', color ='r', label = 'f{sig_cl}$\sigma$')
    ax.axvline(h_lim, ls = 'dashed', color ='r')
    ax.legend() 
    
    
    
    
    
    # xy_gn2 = np.array([gns2_m['x'],gns2_m['y']]).T
    # xy_gn1 = np.array([gns1_m['x'],gns1_m['y']]).T
    lb_gn2 = np.array([gns2_m['l'],gns2_m['b']]).T
    lb_gn1 = np.array([gns1_m['l'],gns1_m['b']]).T
    
    N = -1
    if transf == 'polynomial':
        p = ski.transform.estimate_transform(transf,
                                            lb_gn2[::N], 
                                            lb_gn1[::N], order = order_trans)
    else:    
        p = ski.transform.estimate_transform(transf,
                                        lb_gn2[::N], 
                                        lb_gn1[::N])
        
    print(p)
    
    lb_gn2 = np.array([gns2['l'],gns2['b']]).T
    lb_gn2_t = p(lb_gn2)
    
    gns2['l'] = lb_gn2_t[:,0]*u.deg
    gns2['b'] = lb_gn2_t[:,1]*u.deg
    
    gns1_c = SkyCoord(l = gns1['l'], b = gns1['b'], 
                        unit = 'degree', frame = 'galactic')
    gns2_c = SkyCoord(l = gns2['l'], b = gns2['b'], 
                        unit = 'degree', frame = 'galactic')
    
    
    
    idx,d2d,d3d = gns2_c.match_to_catalog_sky(gns1_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
    sep_constraint = d2d < max_sep
    gns2_m = gns2[sep_constraint]
    gns1_m_re = gns1[idx[sep_constraint]]

    gns1_m = unique(gns1_m_re, keep = 'none')

    unique_indices = []
    for row in gns1_m:
        # Find the index of the row in the original table
        index = np.where((gns1_m_re['x'] == row['x']) & (gns1_m_re['y'] == row['y']))[0][0]
        unique_indices.append(index)
    
    if matches > len(gns2_m):
        break
    else:
        matches = len(gns2_m)
    print(f'MATCHES i = {i}= {len(gns2_m)}')
    
    sys.exit(458)

idx,d2d,d3d = gns2_c.match_to_catalog_sky(gns1_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
sep_constraint = d2d < max_sep
gns2_m = gns2[sep_constraint]
gns1_m_re = gns1[idx[sep_constraint]]

# gns1_m = unique(gns1_m_re, keep = 'none')
gns1_m = unique(gns1_m_re, keep = 'first')

unique_indices = []
for row in gns1_m:
    # Find the index of the row in the original table
    index = np.where((gns1_m_re['x'] == row['x']) & (gns1_m_re['y'] == row['y']))[0][0]
    unique_indices.append(index)

gns2_m = gns2_m[unique_indices]

# sys.exit(475)
# =============================================================================
# idx1, idx2, sep2d, _ = search_around_sky(gns1_c, gns2_c, max_sep)
# 
# count1 = Counter(idx1)
# count2 = Counter(idx2)
# 
# # Step 3: Create mask for one-to-one matches only
# mask_unique = np.array([
#     count1[i1] == 1 and count2[i2] == 1
#     for i1, i2 in zip(idx1, idx2)
# ])
# 
# # Step 4: Apply the mask
# idx1_clean = idx1[mask_unique]
# idx2_clean = idx2[mask_unique]
# 
# gns1_m= gns1[idx1_clean]
# gns2_m = gns2[idx2_clean]
# =============================================================================

# %%
sig_H = 3
dm = gns1_m['H'] - gns2_m['H']

mH, lm, hm = sigma_clip(dm, sigma = sig_H, masked = True, return_bounds= True)

dm_m = dm[np.logical_not(mH.mask)]

fig, ax = plt.subplots(1,1)
ax.hist(dm, bins = 'auto', color = 'k', alpha = 0.1)
ax.hist(dm_m, histtype = 'step',bins = 'auto', label = '$\Delta H$ = %.2f $\pm$ %.2f'%(np.mean(dm_m), np.std(dm_m), ))
ax.axvline(lm, color = 'r', ls ='dashed')
ax.axvline(hm, color = 'r', ls ='dashed')
ax.legend()


gns1_m= gns1_m[np.logical_not(mH.mask)]
gns2_m= gns2_m[np.logical_not(mH.mask)]

# %%






# sys.exit(285)

# mean_b  = np.cos((gns1_m['b'].to(u.rad) + gns2_m['b'].to(u.rad)) / 2.0)

dl = (gns2_m['l']- gns1_m['l'])*np.cos(gns1_m['b'].to(u.rad))

db = (gns2_m['b']- gns1_m['b'])



pm_l = (dl.to(u.mas))/dt_gns.to(u.year)
pm_b = (db.to(u.mas))/dt_gns.to(u.year)

sig_pm = 3
m_pml, l_pml, h_pml = sigma_clip(pm_l, sigma = sig_pm, masked = True, return_bounds= True)
m_pmb, l_pmb, h_pmb = sigma_clip(pm_b, sigma = sig_pm, masked = True, return_bounds= True)


m_pm = np.logical_and(np.logical_not(m_pml.mask),np.logical_not(m_pmb.mask))
pm_lm = pm_l[m_pm]
pm_bm = pm_b[m_pm]

gns1_m = gns1_m[m_pm]
gns1_m['pm_l']  = pm_lm
gns1_m['pm_b']  = pm_bm
#%%
bins = 'auto'

fig, (ax,ax2) = plt.subplots(1,2)
ax.hist(pm_l, bins = bins, color = 'k', alpha = 0.2)
ax2.hist(pm_b, bins = bins,color = 'k', alpha = 0.2)
ax.hist(pm_lm, bins = bins, histtype = 'step', label = '$\overline{\mu}_{l}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(pm_lm.value),np.std(pm_lm.value)))
ax2.hist(pm_bm, bins = bins, histtype = 'step',label = '$\overline{\mu}_{b}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(pm_bm.value),np.std(pm_bm.value)))
ax.set_xlabel('$\Delta \mu_{l}$ [mas]')
ax2.set_xlabel('$\Delta\mu_{b}$ [mas]')
ax.axvline(l_pml.value, ls = 'dashed', color = 'r')
ax.axvline(h_pml.value, ls = 'dashed', color = 'r')
ax2.axvline(l_pmb.value, ls = 'dashed', color = 'r')
ax2.axvline(h_pmb.value, ls = 'dashed', color = 'r')
ax.legend()
ax2.legend()


sys.exit(545)
# %%

comparison_gns = 2#!!! # GNS1 or GN2 for Gaia comparison
radius = abs(np.min(gns1['l'])-np.max(gns1['l']))*0.6*u.degree
center_g = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic')
center_eq = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic').icrs
try:
    
    gaia = Table.read(pruebas1  + 'gaia_f1%s_f2%s_r%.0f.ecsv'%(field_one,field_two,radius.to(u.arcsec).value))
    print('Gaia from table')
except:
    print('Gaia from web')
    center = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic').icrs

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" # Select early Data Release 3
    Gaia.ROW_LIMIT = -1  # it not especifty, Default rows are limited to 50. 
    j = Gaia.cone_search_async(center, radius = abs(radius))
    gaia = j.get_results()
    gaia.write(pruebas1  + 'gaia_f1%s_f2%s_r%.0f.ecsv'%(field_one,field_two,radius.to(u.arcsec).value))


fig_ga, (ax_ga,ax2_ga) = plt.subplots(1,2)
ax_ga.set_title(f'GNS vs Gaia (radius = {radius.to(u.arcsec):.0f})')
ax_ga.scatter(gns1['l'][::20], gns1['b'][::20],s =1, color = 'k', alpha = 0.2)
# ax.scatter(gaia['ra'], gaia['dec'], label = 'All Gaia')
ax2_ga.scatter(gaia['phot_g_mean_mag'],gaia['pmra_error'], s= 2, color = 'k', label = 'All Gaia')
ax2_ga.scatter(gaia['phot_g_mean_mag'],gaia['pmdec_error'], s= 2, color = 'grey')
ax_ga.invert_xaxis()
ax2_ga.grid()
e_pm = 0.5#!!!
mlim_ga = 20#!!!
gaia = filter_gaia_data(
    gaia_table=gaia,
    astrometric_params_solved=31,
    duplicated_source= False,
    parallax_over_error_min=-10,
    astrometric_excess_noise_sig_max=2,
    phot_g_mean_mag_min= mlim_ga,
    phot_g_mean_mag_max=1,
    pm_min=0,
    pmra_error_max=e_pm,
    pmdec_error_max=e_pm
    )


ax2_ga.axvline(mlim_ga, color = 'r', ls = 'dashed')

t1 = Time(['2016-01-01T00:00:00'],scale='utc')




# g_gpm = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], pm_ra_cosdec = gaia['pmra'].value*u.mas/u.yr, pm_dec = ['pmdec'].value*u.mas/u.yr, obstime = 'J2016', equinox = 'J2000', frame = 'fk5')
ga_gpm = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], pm_ra_cosdec = gaia['pmra'],
                 pm_dec = gaia['pmdec'], obstime = 'J2016', 
                 equinox = 'J2000', frame = 'icrs').galactic

if comparison_gns == 1:

    dt = t1_gns-t1

elif comparison_gns == 2:

    dt = t2_gns-t1
    



# %
gaia_c = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], 
                  pm_ra_cosdec = gaia['pmra'], pm_dec = gaia['pmdec'],
                  frame = 'icrs', obstime = 'J2016.0').galactic


gaia['pm_l'] = gaia_c.pm_l_cosb
gaia['pm_b'] = gaia_c.pm_b
# %

ga_l = gaia_c.l.wrap_at('360.02d')
ga_b = gaia_c.b.wrap_at('180d')

ax2_ga.axhline(e_pm, color = 'red', ls = 'dashed')
ax_ga.scatter(ga_l, gaia['b'], color = '#ff7f0e')


# %
if comparison_gns == 1:
    gns_c = SkyCoord(l = gns1_m['l'], b = gns1_m['b'], 
                     unit = 'degree', frame = 'galactic')
if comparison_gns == 2:
    gns_c = SkyCoord(l = gns2_m['l'], b = gns2_m['b'], 
                     unit = 'degree', frame = 'galactic')


# %
# % Calculating the position of Gaia stasr
gaia_c = SkyCoord(ra = gaia['ra'], dec = gaia['dec'],pm_ra_cosdec = gaia['pmra'],pm_dec = gaia['pmdec'], frame = 'icrs', obstime = 'J2016.0')

gaia['ra'] = gaia['ra'] + gaia_c.pm_ra_cosdec*dt.to(u.yr)
gaia['dec'] = gaia['dec'] + gaia_c.pm_dec*dt.to(u.yr)

gaia_cg = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], unit = 'degree', frame = 'icrs', obstime = 'J2016.0').galactic

# %
# =============================================================================
# gaia_ceq = SkyCoord(ra = gaia['ra'], dec = gaia['dec'],pm_ra_cosdec = gaia['pmra'],pm_dec = gaia['pmdec'], frame = 'icrs', obstime = 'J2016.0')
# 
# ra_off,dec_off = center_eq.spherical_offsets_to(gaia_ceq.frame)
# ra_offt = ra_off.to(u.mas) + (gaia_ceq.pm_ra_cosdec)*dt.to(u.yr)
# dec_offt = dec_off.to(u.mas) + (gaia_ceq.pm_dec)*dt.to(u.yr)
# 
# ga_gtc = center_eq.spherical_offsets_by(ra_offt, dec_offt)
# 
# 
# gaia['ra'] = ga_gtc.ra
# gaia['dec'] = ga_gtc.dec
# 
# gaia_cg = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], frame = 'icrs', obstime = 'J2016.0').galactic
# =============================================================================


# %
# Gaia comparison
max_sep_ga =45*u.mas#!!!
idx,d2d,d3d = gaia_c.match_to_catalog_sky(gns_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
sep_constraint = d2d < max_sep_ga
gaia_m = gaia[sep_constraint]
gg_m = gns1_m[idx[sep_constraint]]

# idx1, idx2, sep2d, _ = search_around_sky(gaia_cg, gns_c, max_sep_ga)

# count1 = Counter(idx1)
# count2 = Counter(idx2)

# # Step 3: Create mask for one-to-one matches only
# mask_unique = np.array([
#     count1[i1] == 1 and count2[i2] == 1
#     for i1, i2 in zip(idx1, idx2)
# ])

# # Step 4: Apply the mask
# idx1_clean = idx1[mask_unique]
# idx2_clean = idx2[mask_unique]

# # Optional: Extract matched tables
# gaia_m= gaia[idx1_clean]
# gg_m = gns1_m[idx2_clean]

ax2_ga.scatter(gaia_m['phot_g_mean_mag'],gaia_m['pmra_error'], s= 2, color = 'blue', label = 'Gaia matches')
ax2_ga.scatter(gaia_m['phot_g_mean_mag'],gaia_m['pmdec_error'], s= 2, color = 'cyan')

ax_ga.scatter(gaia_m['l'], gaia_m['b'], color = 'cyan', marker = 'x')

ax2_ga.legend()
ax_ga.legend()

diff_l = gaia_m['pm_l'] - gg_m['pm_l']
diff_b = gaia_m['pm_b'] - gg_m['pm_b']



sig_gaia = 3
m_dl, l_dl, h_dl = sigma_clip(diff_l, sigma = sig_gaia, masked = True, return_bounds= True)
m_db, l_db, h_db = sigma_clip(diff_b, sigma = sig_gaia, masked = True, return_bounds= True)
m_dpm = np.logical_and(np.logical_not(m_dl.mask),np.logical_not(m_db.mask))

diff_lm = diff_l[m_dpm]
diff_bm = diff_b[m_dpm]

# diff_lm = diff_l
# diff_bm = diff_b
# %
bins = 'auto'
fig, (axh,axh2) = plt.subplots(1,2)
axh.set_title(f'Gaia Matches = {len(diff_lm)}')

#
axh.hist(diff_lm , bins = bins, histtype = 'step', label = '$\Delta{\mu}_{l}$ = %.2f\n$\sigma$ = %.2f'%(np.mean(diff_lm.value),np.std(diff_lm.value)))
axh2.hist(diff_bm, bins = bins, histtype = 'step',label = '$\Delta{\mu}_{b}$ = %.2f\n$\sigma$ = %.2f'%(np.mean(diff_bm.value),np.std(diff_bm.value)))
# ax.hist(diff_l, color = 'k', alpha = 0.2)
# ax2.hist(diff_b, color = 'k', alpha = 0.2)
axh.set_xlabel('$\Delta \mu_{l}$ [mas/yr]')
axh2.set_xlabel('$\Delta\mu_{b}$ [mas/yr]')
axh.axvline(l_dl, ls = 'dashed', color = 'r')
axh.axvline(h_dl, ls = 'dashed', color = 'r')
axh2.axvline(l_db, ls = 'dashed', color = 'r',label = f'{sig_gaia}$\sigma$')
axh2.axvline(h_db, ls = 'dashed', color = 'r')
axh.legend()
axh2.legend()

# plt.savefig('/Users/amartinez/Desktop/for_people/for_Rainer/hist.png', transparent=True,bbox_inches = 'tight')
# %
N = 100
# %
fig, (ax,ax2) = plt.subplots(1,2)
ax.scatter(gns1['l'][::N], gns1['b'][::N], color = 'k', alpha = 0.1)
ax.scatter(gaia_m['l'], gaia_m['b'], label = 'All Gaia matches')
ax.scatter(gaia_m['l'][~m_dpm], gaia_m['b'][np.logical_not(m_dpm)],label = f'Gaia {sig_gaia}$\sigma clipped$', marker = 'x' )
ax.axis('equal')
ax.legend()
ax2.scatter(gns2['l'][::N], gns2['b'][::N], color = 'k', alpha = 0.1)
ax2.scatter(gaia_m['l'], gaia_m['b'], label = 'All Gaia matches')
ax2.scatter(gaia_m['l'][~m_dpm], gaia_m['b'][~m_dpm],label = f'Gaia {sig_gaia}$\sigma clipped$', marker = 'x' )
ax2.axis('equal')
ax2.legend()
fig.tight_layout()

# %
# %


# Create the figure and gridspec layout
fig_g = plt.figure(figsize=(4, 4))
gs = fig_g.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0, wspace=0)

# Main scatter plot
axg = fig_g.add_subplot(gs[1, 0])
axg.scatter(diff_lm, diff_bm, s=200, edgecolor='k', zorder=3)
axg.set_xlabel(r'$\Delta \mu_{l}$ [mas/yr]', fontsize=16)
axg.set_ylabel(r'$\Delta \mu_{b}$ [mas/yr]', fontsize=16)
axg.set_title(f'Gaia matches {len(diff_bm)} Max sep = {max_sep_ga}', fontsize=12)
axg.grid()

props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
axg.text(0.05, 0.95,
         '$\sigma_x$ = %.2f mas/yr\n$\sigma_y$ = %.2f mas/yr' % (np.std(diff_lm), np.std(diff_bm)),
         transform=axg.transAxes, fontsize=12, verticalalignment='top', bbox=props)
axg.tick_params(axis='both', labelsize=12)

# # Histogram on the top
# ax_histx = fig_g.add_subplot(gs[0, 0], sharex=axg)
# ax_histx.hist(diff_lm, bins='auto', histtype='step', linewidth=1.5, color='k')
# ax_histx.tick_params(axis='x', labelbottom=False)
# ax_histx.set_yticks([])
# ax_histx.axis('off')

# # Histogram on the right
# ax_histy = fig_g.add_subplot(gs[1, 1], sharey=axg)
# ax_histy.hist(diff_bm, bins='auto', orientation='horizontal', histtype='step', linewidth=1.5, color='k')
# ax_histy.tick_params(axis='y', labelleft=False)
# ax_histy.set_xticks([])
# ax_histy.axis('off')

# plt.show()

# %
# 
# mta = {'Title': 'v-p.png','script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_absolute_SUPER/SUPER_alignment.py'}
# plt.savefig('/Users/amartinez/Desktop/for_people/for_Rainer/v-p.png', transparent=True, bbox_inches = 'tight')
sys.exit(730)
# %%

# %

mask_gns = (gaia['l']<max(gns2_m['l'])) & (gaia['l']>min(gns2_m['l'])) & (gaia['b']<max(gns2_m['b'])) & (gaia['b']>min(gns2_m['b']))
gaia = gaia[mask_gns]

gaia_c = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], 
                  pm_ra_cosdec = gaia['pmra'], pm_dec = gaia['pmdec'],
                  frame = 'icrs', obstime = 'J2016.0').galactic

ga_l = gaia_c.l.wrap_at('360.02d')

n_bins = 6
statistic, x_edges, y_edges, binnumber = stats.binned_statistic_2d(ga_l, gaia['b'], np.sqrt((gaia['ra_error']**2) + (gaia['dec_error']**2)), statistic='median', bins=(n_bins,int(n_bins/2)))
# Create a meshgrid for plotting
X, Y = np.meshgrid(x_edges, y_edges)


N=10
fig, ax = plt.subplots(1,1)
ax.set_title('GAIA error distribution')
ax.scatter(gns2_m['l'][::N], gns2_m['b'][::N], color = 'k', alpha = 0.1)
ge = ax.pcolormesh(X, Y, statistic.T, cmap = 'Spectral_r')
c = fig.colorbar(ge, ax = ax, label='$\sqrt {\sigma RA^{2} + \sigma Dec^{2}}$')
ax.scatter(ga_l, gaia['b'], c = np.sqrt((gaia['ra_error']**2) + (gaia['dec_error']**2)), cmap = 'Spectral_r', edgecolor = 'k')
ax.axis('equal')


# %%


fig, (ax, ax2) = plt.subplots(1,2)

ax.scatter(gg_m['H'], gg_m['sl'], label = '$\delta l$' ) 
ax.scatter(gg_m['H'], gg_m['sb'],label = '$\delta b$',marker='x')
ax2.hist(gg_m['H'], bins = 'auto',histtype = 'step')
ax2.set_xlabel('[H] matches')
ax.set_xlabel('[H] matches')
ax.set_ylabel('Uncertainty position')
ax.legend()




# %%
fig, ax = plt.subplots(1,1)
ax.scatter(gns2['l'][::N], gns2['b'][::N], color = 'k', alpha = 0.1)
ax.scatter(gns1['l'][::N], gns1['b'][::N], color = 'white', alpha = 0.01)
ax.scatter(gaia_m['l'], gaia_m['b'], label = 'All Gaia matches')
ax.axis('equal')
ax.legend()
# %%

# Comparison with VVV
vvv = Table.read('/Users/amartinez/Desktop/PhD/Catalogs/VVV/b333/PMS/b333.dat', format = 'ascii')


ok_mask = vvv['ok'] > 0
vvv = vvv[ok_mask]
vvv_c = SkyCoord(ra = vvv['ra'], dec = vvv['dec'], unit = 'degree', equinox = 'J2000')
 

max_sep = 50*u.mas
idx,d2d,d3d = vvv_c.match_to_catalog_sky(gns_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 match
sep_constraint = d2d < max_sep
vvv_m= vvv[sep_constraint]
cat_m = gns_c[idx[sep_constraint]]


fig, ax = plt.subplots(1,1)

ax.scatter(gns_c.ra,gns_c.dec, label = 'GNS')
ax.scatter(vvv_m['ra'], vvv_m['dec'], label = 'VVV')
ax.legend()


# mux_d = (cat_m['ALPHA_J2000'] - vvv_m['ra'])*3600*1000
# muy_d = (cat_m['DELTA_J2000'] - vvv_m['dec'])*3600*1000
mux_d = cat_m['PMALPHA_J2000'] - vvv_m['pmRA']
muy_d = cat_m['PMDELTA_J2000'] - vvv_m['pmDEC']

sig_cl = 3
mask_x, lx_lim,hx_lim = sigma_clip(mux_d, sigma=sig_cl, masked = True, return_bounds= True)
mask_y, ly_lim,hy_lim = sigma_clip(muy_d, sigma=sig_cl, masked = True, return_bounds= True)

# mask_xy = mask_x & mask_y # mask_xy = np.logical(mx, my)
mask_xy = np.logical_and(np.logical_not(mask_x.mask), np.logical_not(mask_y.mask))

mux_dc = mux_d[mask_xy]
muy_dc = muy_d[mask_xy]

fig, (ax,ax2) = plt.subplots(1,2)
ax.set_title('GNS vs VVVV')

ax.hist(mux_dc, histtype  = 'step', label = r'$\mu_{x}$ = %.2f,$\sigma$ = %.2f'%(np.mean(mux_dc),np.std(mux_dc)))
ax2.hist(muy_dc, histtype  = 'step',label = r'$\mu_{y}$ = %.2f,$\sigma$ = %.2f'%(np.mean(muy_dc),np.std(muy_dc)))
# ax.hist(mux_d,color = 'k', alpha = 0.2)
# ax2.hist(muy_d,color = 'k', alpha = 0.2)

# ax.axvline(lx_lim, ls = 'dashed', color ='r')
# ax.axvline(hx_lim, ls = 'dashed', color ='r')
# ax2.axvline(ly_lim, ls = 'dashed', color ='r')
# ax2.axvline(hy_lim, ls = 'dashed', color ='r')
ax.legend()
ax2.legend()









