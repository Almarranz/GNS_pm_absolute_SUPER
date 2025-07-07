#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:19:45 2022

@author: amartinez
"""

# Generates the GNS1 second reduction with the Ks and H magnitudes

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

#%%
field_one = 60#
chip_one = 0
field_two = 20
chip_two = 0


# max_sig = 0.1
max_sig = 0.1

if field_one == 7 or field_one == 12 or field_one == 10 or field_one == 16:
    t1_gns = Time(['2015-06-07T00:00:00'],scale='utc')
elif field_one == 60:
    t1_gns = Time(['2016-06-13T00:00:00'],scale='utc')
elif field_one ==  100:
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
transf = 'similarity'#!!!1|
# transf = 'polynomial'#!!!
# transf = 'shift'#!!!
order_trans = 1
mlim = 19

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


# gns1 = Table.read(GNS_1 + 'stars_calibrated_H_chip%s.ecsv'%(chip_one),  format = 'ascii.ecsv')
gns1 = Table.read('/Users/amartinez/Desktop/Projects/GNS_gd/scamp/lxp/GNS1/H/F06/B6_H_opti.ecsv',  format = 'ascii.ecsv')

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

num_bins = 100
statistic, x_edges, y_edges, binnumber = stats.binned_statistic_2d(gns1['l'], gns1['b'], np.sqrt(gns1['sl']**2 + gns1['sb']**2), statistic='median', bins=(num_bins,int(num_bins/2)))
# Create a meshgrid for plotting
X, Y = np.meshgrid(x_edges, y_edges)
# Plot the result
fig, ax = plt.subplots()
# c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r', norm = LogNorm()) 
c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',vmax = 0.10) 
fig.colorbar(c, ax=ax, label='$\sqrt {\sigma l^{2} + \sigma l^{2}}$', shrink = 1)
ax.set_title('GNS1')
ax.set_xlabel('l')
ax.set_ylabel('b')
ax.axis('equal')

m_mask = gns1['H']<mlim

gns1 = gns1[m_mask]
unc_cut = np.where((gns1['sl']<max_sig) & (gns1['sb']<max_sig))
gns1 = gns1[unc_cut]

gns1_gal = SkyCoord(l = gns1['l'], b = gns1['b'], 
                    unit = 'degree', frame = 'galactic')


# %%
# gns2_all = np.loadtxt(GNS_2 + 'stars_calibrated_H_chip%s.txt'%(chip_two))

# gns2 = Table.read(GNS_2 + 'stars_calibrated_H_chip%s.ecsv'%(chip_two), format = 'ascii')
gns2 = Table.read('/Users/amartinez/Desktop/Projects/GNS_gd/scamp/lxp/GNS2/H/F20/20_H_opti.ecsv', format = 'ascii.ecsv')
# gns2 = Table.read('/Users/amartinez/Desktop/Projects/GNS_gd/scamp/lxp/GNS2/H/F20/20_H_opti_fcalib.ecsv', format = 'ascii.ecsv')

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

num_bins = 100
statistic, x_edges, y_edges, binnumber = stats.binned_statistic_2d(gns2['l'], gns2['b'], np.sqrt(gns2['sl']**2 + gns2['sb']**2), statistic='median', bins=(num_bins,int(num_bins/2)))
# Create a meshgrid for plotting
X, Y = np.meshgrid(x_edges, y_edges)
# Plot the result
fig, ax = plt.subplots()
# c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',norm = LogNorm())
c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r', vmax= 0.10)
fig.colorbar(c, ax=ax, label='$\sqrt {\sigma l^{2} + \sigma l^{2}}$', shrink = 1)
ax.set_title('GNS2')
ax.set_xlabel('l')
ax.set_ylabel('b')
ax.axis('equal')

unc_cut2 = np.where((gns2['sl']<max_sig) & (gns2['sb']<max_sig))
gns2 = gns2[unc_cut2]
m_mask = gns2['H']<mlim
gns2 = gns2[m_mask]

gns2_gal = SkyCoord(l = gns2['l'], b = gns2['b'], 
                    unit = 'degree', frame = 'galactic')



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


radius = abs(np.min(gns1['l'])-np.max(gns1['l']))*0.6*u.degree
center_g = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic')

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

e_pm = 0.1
gaia = filter_gaia_data(
    gaia_table=gaia,
    astrometric_params_solved=31,
    duplicated_source= False,
    parallax_over_error_min=-10,
    astrometric_excess_noise_sig_max=2,
    phot_g_mean_mag_min= None,
    phot_g_mean_mag_max=1,
    pm_min=0,
    pmra_error_max=e_pm,
    pmdec_error_max=e_pm
    )




t1 = Time(['2016-01-01T00:00:00'],scale='utc')


dt = t1_gns-t1

# g_gpm = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], pm_ra_cosdec = gaia['pmra'].value*u.mas/u.yr, pm_dec = ['pmdec'].value*u.mas/u.yr, obstime = 'J2016', equinox = 'J2000', frame = 'fk5')
ga_gpm = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], pm_ra_cosdec = gaia['pmra'],
                 pm_dec = gaia['pmdec'], obstime = 'J2016', 
                 equinox = 'J2000', frame = 'icrs').galactic


l_off,b_off = center_g.spherical_offsets_to(ga_gpm.frame)
l_offt = l_off.to(u.mas) + (ga_gpm.pm_l_cosb)*dt.to(u.yr)
b_offt = b_off.to(u.mas) + (ga_gpm.pm_b)*dt.to(u.yr)

ga_gtc = center_g.spherical_offsets_by(l_offt, b_offt)


gaia['l'] = ga_gtc.l
gaia['b'] = ga_gtc.b


# %%
gaia_c = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], 
                  pm_ra_cosdec = gaia['pmra'], pm_dec = gaia['pmdec'],
                  frame = 'icrs', obstime = 'J2016.0').galactic

gaia['pm_l'] = gaia_c.pm_l_cosb
gaia['pm_b'] = gaia_c.pm_b
# %%

ga_l = gaia_c.l.wrap_at('360.02d')
ga_b = gaia_c.b.wrap_at('180d')

fig, ax = plt.subplots(1,1,figsize =(10,10))
ax.scatter(l1[::10], gns1['b'][::10],label = 'GNS_1 Fied %s, chip %s'%(field_one,chip_one))
ax.scatter(l2[::10],  gns2['b'][::10],s = 1, label = 'GNS_2 Fied %s, chip %s'%(field_two,chip_two))
ax.scatter(ga_l,gaia['b'], label = f'Gaia stars = {len(gaia)}')
ax.invert_xaxis()
ax.legend()
ax.set_xlabel('l[ deg]', fontsize = 10)
ax.set_ylabel('b [deg]', fontsize = 10)
# ax.axis('scaled')
ax.set_ylim(min(gaia['b']),max(gaia['b']))



gns1_c = SkyCoord(l = gns1['l'], b = gns1['b'], 
                    unit = 'degree', frame = 'galactic')
gns2_c = SkyCoord(l = gns2['l'], b = gns2['b'], 
                    unit = 'degree', frame = 'galactic')


# %%
# Driect alignemnet
max_sep = 150*u.mas#!!!


for i in range(0):
    # idx,d2d,d3d = gns2_c.match_to_catalog_sky(gns1_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
    # sep_constraint = d2d < max_sep
    # gns2_m = gns2[sep_constraint]
    # gns1_m = gns1[idx[sep_constraint]]
    
    # print(f'MATCHES i = {i-1} = {len(gns2_m)}')
    max_sep_fine = 5*u.mas
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
    sig_cl = 2
    mask_H, l_lim,h_lim = sigma_clip(diff_H, sigma=sig_cl, masked = True, return_bounds= True)
    
    gns2_m = gns2_m[mask_H.mask]
    gns1_m = gns1_m[mask_H.mask]
    
    fig,ax = plt.subplots(1,1)
    ax.hist(diff_H, bins = 'auto',histtype = 'step')
    ax.axvline(np.mean(diff_H), color = 'k', ls = 'dashed', label = 'H offset = %.2f\n$\sigma$ = %.2f'%(off_s,np.std(diff_H)))
    ax.axvline(l_lim, ls = 'dashed', color ='r')
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

    
    print(f'MATCHES i = {i}= {len(gns2_m)}')


# idx,d2d,d3d = gns2_c.match_to_catalog_sky(gns1_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
# sep_constraint = d2d < max_sep
# gns2_m = gns2[sep_constraint]
# gns1_m_re = gns1[idx[sep_constraint]]

# gns1_m = unique(gns1_m_re, keep = 'none')

# unique_indices = []
# for row in gns1_m:
#     # Find the index of the row in the original table
#     index = np.where((gns1_m_re['x'] == row['x']) & (gns1_m_re['y'] == row['y']))[0][0]
#     unique_indices.append(index)

# gns2_m = gns2_m[unique_indices]

idx1, idx2, sep2d, _ = search_around_sky(gns1_c, gns2_c, max_sep)

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


# %%

gns1_c = SkyCoord(l = gns1_m['l'], b = gns1_m['b'], 
                    unit = 'degree', frame = 'galactic')


# Gaia comparison
max_sep = 12*u.mas
# idx,d2d,d3d = gaia_c.match_to_catalog_sky(gns1_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
# sep_constraint = d2d < max_sep
# gaia_m = gaia[sep_constraint]
# gg_m = gns1_m[idx[sep_constraint]]

idx1, idx2, sep2d, _ = search_around_sky(gaia_c, gns1_c, max_sep)

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

# Optional: Extract matched tables
gaia_m= gaia[idx1_clean]
gg_m = gns1_m[idx2_clean]


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
fig, (ax,ax2) = plt.subplots(1,2)
ax.set_title(f'Gaia Matches = {len(diff_lm)}')
# ax2.set_title(f'Matching dis. = {max_sep.value} mas')

#
ax.hist(diff_lm , bins = bins, histtype = 'step', label = '$\Delta{\mu}_{l}$ = %.2f\n$\sigma$ = %.2f'%(np.mean(diff_lm.value),np.std(diff_lm.value)))
ax2.hist(diff_bm, bins = bins, histtype = 'step',label = '$\Delta{\mu}_{b}$ = %.2f\n$\sigma$ = %.2f'%(np.mean(diff_bm.value),np.std(diff_bm.value)))
# ax.hist(diff_l, color = 'k', alpha = 0.2)
# ax2.hist(diff_b, color = 'k', alpha = 0.2)
ax.set_xlabel('$\Delta \mu_{l}$ [mas/yr]')
ax2.set_xlabel('$\Delta\mu_{b}$ [mas/yr]')
ax.axvline(l_dl, ls = 'dashed', color = 'r')
ax.axvline(h_dl, ls = 'dashed', color = 'r')
ax2.axvline(l_db, ls = 'dashed', color = 'r',label = f'{sig_gaia}$\sigma$')
ax2.axvline(h_db, ls = 'dashed', color = 'r')
ax.legend()
ax2.legend()

# plt.savefig('/Users/amartinez/Desktop/for_people/for_Rainer/hist.png', transparent=True,bbox_inches = 'tight')


# %
fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0, wspace=0)


# ax.axis('scaled')

# Histogram on the top
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histx.hist(diff_lm - np.mean(diff_lm.value) , bins='auto',histtype = 'step',linewidth=1, alpha=0.7, color ='k')
ax_histx.set_yticks([])
ax_histx.set_xticks([])
ax_histx.axis('off')

# Histogram on the right
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
ax_histy.hist(diff_bm - np.mean(diff_bm.value),  bins='auto',
              histtype = 'step',linewidth=1, 
              alpha=0.7, orientation='horizontal', color = 'k') 
ax_histy.set_yticks([])
ax_histy.set_xticks([])
ax_histy.axis('off')



# Main scatter plot
ax = fig.add_subplot(gs[1, 0])
ax.set_title(f'Gaia matches {len(diff_bm)}', fontsize = 30)
ax.grid()
ax.scatter(diff_lm - np.mean(diff_lm.value), diff_bm -  np.mean(diff_bm.value), s = 300,edgecolor='k',zorder=3,label ='$\sigma_{x}$ = %.2f'%(np.std(diff_lm)))
# ax.axvline(lx_lim, ls='dashed', color='r', alpha=1)
# ax.axvline(hx_lim, ls='dashed', color='r', alpha=1)
# ax.axhline(ly_lim, ls='dashed', color='r', alpha=1)
# ax.axhline(hy_lim, ls='dashed', color='r', alpha=1)
ax.set_xlabel(r'$\Delta$ $\mu_{l}$ [mas/yr]', fontsize = 20)
ax.set_ylabel(r'$\Delta$ $\mu_{b}$ [mas/yr]',fontsize = 20)
props = dict(boxstyle='round',
             facecolor='lightblue', alpha=1)
# '#1f77b4'
ax.text(0.06, 0.95, '$\sigma_{x}$ = %.2f mas/yr\n$\sigma_{y}$ = %.2f mas/yr'%(np.std(diff_lm),np.std(diff_bm)), transform=ax.transAxes, fontsize=20,
           verticalalignment='top', bbox=props)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)

# 
# plt.savefig('/Users/amartinez/Desktop/for_people/for_Rainer/v-p.png', transparent=True, bbox_inches = 'tight')
# %
N = 100
fig, (ax,ax2) = plt.subplots(1,2)
ax.scatter(gns1['l'][::N], gns1['b'][::N], color = 'k', alpha = 0.1)
ax.scatter(gaia_m['l'][m_dpm], gaia_m['b'][m_dpm],label = f'Gaia {sig_gaia}$\sigma clipped$' )
ax.scatter(gaia_m['l'], gaia_m['b'], label = 'All Gaia matches')
ax.axis('equal')
ax.legend()
ax2.scatter(gns2['l'][::N], gns2['b'][::N], color = 'k', alpha = 0.1)
ax2.scatter(gaia_m['l'][m_dpm], gaia_m['b'][m_dpm],label = f'Gaia {sig_gaia}$\sigma clipped$' )
ax2.scatter(gaia_m['l'], gaia_m['b'], label = 'All Gaia matches')
ax2.axis('equal')
ax2.legend()
fig.tight_layout()
# %%


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

ax.scatter(gg_m['H'], gg_m['sl'])
ax.scatter(gg_m['H'], gg_m['sb'],marker='x')
ax2.hist(gg_m['H'], bins = 'auto')













