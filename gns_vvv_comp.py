#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:15:54 2025

@author: amartinez
"""
import sys
sys.path.append("/Users/amartinez/Desktop/pythons_imports/")


import numpy as np

import matplotlib.pyplot as plt
from compare_lists import compare_lists 
from astropy.table import Table
from astropy.table import unique
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import matplotlib.colors as colors_plt
from skimage import data
from skimage import transform
import astroalign as aa
from astroquery.gaia import Gaia
import skimage as ski
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy.coordinates import SkyCoord
import Polywarp as pw
from astroquery.gaia import Gaia
from astropy import units as u
import cluster_finder
import pandas as pd
import copy
import cluster_finder
from filters import filter_gaia_data
from filters import filter_hosek_data
from filters import filter_gns_data
from filters import filter_vvv_data
from astropy.time import Time
from astropy.coordinates import search_around_sky
from collections import Counter
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
# %%
# field_one = 10
# chip_one = 0
# field_two = 4
# chip_two = 0

# field_one = 'B1'
# chip_one = 0
# field_two = 20
# chip_two = 0

field_one = 16
chip_one = 0
field_two = 7
chip_two = 0

# =============================================================================
# GNS data
# =============================================================================
e_pm_gns = 1



# =============================================================================
# VVVComparison
# =============================================================================
max_sep_vvv = 50*u.mas

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

vvv_t = Time(['2012-04-01'], scale='utc')

dt1 = t1_gns - vvv_t


GNS_1='/Users/amartinez/Desktop/PhD/HAWK/GNS_1/lists/%s/chip%s/'%(field_one, chip_one)
GNS_2='/Users/amartinez/Desktop/PhD/HAWK/GNS_2/lists/%s/chip%s/'%(field_two, chip_two)

pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1absolute_SUPER/pruebas/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2absolute_SUPER/pruebas/'


 # gns1.write(pruebas1 + f'gns1_pmSuper_F1_{field_one}_F2_{field_two}.ecvs',format = 'ascii.ecsv', overwrite = True)
 # gns2.write(pruebas2 + f'gns2_pmSuper_F1_{field_one}_F2_{field_two}.ecvs',format = 'ascii.ecsv', overwrite = True)

gns1 = Table.read(pruebas1 + f'gns1_pmSuper_F1_{field_one}_F2_{field_two}.ecvs',format = 'ascii.ecsv')

gns1 = filter_gns_data(gns1, max_e_pm = e_pm_gns, min_mag = 18, max_mag = 12)
# sys.exit(116)
# #Gaia comparison

# search_r = 50*u.arcsec

# gns1_radec = SkyCoord(l = gns1['l'], b = gns1['b'],
# ra_c = np.mean(gns1_pm['ra1'])
# dec_c = np.mean(gns1_pm['Dec1'])


# Comparison with VVV 
VVV = Table.read('/Users/amartinez/Desktop/PhD/Catalogs/VVV/b333/PMS/b333.dat', format = 'ascii')

Ks_max = 12.5
Ks_min = 14
vvv = filter_vvv_data(VVV,
                    pmRA = 'good',
                    pmDE = None,
                    epm = 0.8,
                    ok = 'yes',
                    max_Ks = Ks_max,
                    min_Ks = Ks_min,
                    center = 'yes'
                    )


vvv_g = SkyCoord(ra = vvv['ra'], dec = vvv['dec'], unit = 'degree', 
                 pm_ra_cosdec = vvv['pmRA']*u.mas/u.yr,
                 pm_dec = vvv['pmDEC']*u.mas/u.yr,
                 frame = 'icrs', obstime = 'J2012.29578304').galactic

vvv['l'] = vvv_g.l + vvv_g.pm_l_cosb*dt1
vvv['b'] = vvv_g.b + vvv_g.pm_b*dt1
vvv['pml'] = vvv_g.pm_l_cosb
vvv['pmb'] = vvv_g.pm_b

lc  = np.mean(gns1['l'])
bc  = np.mean(gns1['b'])




buenos1 = (gns1['l']>min(vvv['l'])) & (gns1['l']<max(vvv['l'])) & (gns1['b']>min(vvv['b'])) & (gns1['b']<max(vvv['b']))
gns1 = gns1[buenos1]

buenos2 = (vvv['l']>min(gns1['l'])) & (vvv['l']<max(gns1['l'])) & (vvv['b']>min(gns1['b'])) & (vvv['b']<max(gns1['b']))
vvv = vvv[buenos2]
# %%


fig, ax = plt.subplots(1,1)
ax.scatter(gns1['l'], gns1['b'], label = 'GNS')
ax.scatter(vvv['l'], vvv['b'], label = 'VVV')
# %%

vvvc = SkyCoord(l = vvv['l'], b = vvv['b'], frame = 'galactic')
gns1c = SkyCoord(l = gns1['l'], b = gns1['b'], frame = 'galactic')
idx1, idx2, sep2d, _ = search_around_sky(vvvc, gns1c, max_sep_vvv)

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

vvv_m= vvv[idx1_clean]
gns1_m = gns1[idx2_clean]


print(40*'+')
unicos = unique(gns1_m, keep = 'first')
print(len(gns1_m),len(unicos))
print(40*'+')
  
dpm_x = vvv_m['pml'] - gns1_m['pm_xp']
dpm_y = vvv_m['pmb'] - gns1_m['pm_yp']

def sig_cl(x, y,s):
    mx, lx, hx = sigma_clip(x , sigma = s, masked = True, return_bounds= True)
    my, ly, hy = sigma_clip(y , sigma = s, masked = True, return_bounds= True)
    m_xy = np.logical_and(np.logical_not(mx.mask),np.logical_not(my.mask))
    
    return m_xy, [lx,hx,ly,hy]


mpm , lim = sig_cl(dpm_x, dpm_y,s=3)

dpm_xm = dpm_x[mpm]
dpm_ym = dpm_y[mpm]

rcParams.update({
    "figure.figsize": (10, 5),
    "font.size": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax.hist(dpm_xm, histtype='step', bins='auto', lw=2,
        label='$\overline{\Delta \mu_{l}}$ = %.2f'
              '\n$\sigma_b$ = %.2f' % 
              (np.mean(dpm_xm), np.std(dpm_xm)))

ax2.hist(dpm_ym, histtype='step', bins='auto', lw=2,
         label='$\overline{\Delta \mu_{b}}$ = %.2f'
               '\n$\sigma_l$ = %.2f' % 
               (np.mean(dpm_ym), np.std(dpm_ym)))

ax.set_xlabel(r'$\Delta \mu_{l}$ [mas/yr]')
ax2.set_xlabel(r'$\Delta \mu_{b}$ [mas/yr]')
ax.set_ylabel('# stars')
ax.set_xlim(-3,3)
ax2.set_xlim(-3,3)
ax.legend(loc=1)
ax2.legend(loc=1)

fig.tight_layout()


# gns1 = filter_gns_data(gns1, max_e_pos = max_sig, max_mag = gns_mags[0], min_mag = gns_mags[1] )

# =============================================================================
# center = 
# 
# delta_ra = (VVV['ra'] - ra_c) * np.cos(np.radians(dec_c))
# delta_dec = VVV['dec'] - dec_c
# 
# # Calculate the angular distance
# angular_distance = np.sqrt(delta_ra**2 + delta_dec**2)
# 
# # Select rows where the distance is within the radius
# within_radius = angular_distance <= rad
# 
# vvv_c = VVV[within_radius]
# 
# Ks_max = None
# Ks_min = None
# vvv_c = filter_vvv_data(vvv_c,
#                     pmRA = 'good',
#                     pmDE = None,
#                     epm = 0.95,
#                     ok = 'yes',
#                     max_Ks = Ks_max,
#                     min_Ks = Ks_min,
#                     center = 'yes'
#                     )
# 
# 
# 
# fig, ax = plt.subplots(1,1)
# ax.scatter(vvv_c['ra'],vvv_c['dec'])
# ax.scatter(gns1_pm['ra1'], gns1_pm['Dec1'], alpha = 0.1)
# 
# # Moves coordinates to GNS1 obstime???
# tvvv = 2012.29578304
# # vvv_c['ra'] = vvv_c['ra'] + (t1-tvvv) * (vvv_c['pmRA']/1000.0)/3600.0 * np.cos(vvv_c['dec']*np.pi/180.)
# # vvv_c['ra'] = vvv_c['ra'] + (t1-tvvv) * (vvv_c['pmRA']/1000.0)/3600.0 
# # vvv_c['dec'] = vvv_c['dec'] + (t1-tvvv) * (vvv_c['pmDEC']/1000.0)/3600.0
# vvv_coord = SkyCoord(ra = vvv_c['ra'], dec = vvv_c['dec'], unit = 'degree',
#                       frame = 'icrs', obstime = 'J2012.29578304')
# # vvv_coord = SkyCoord(ra = vvv_c['ra'], dec = vvv_c['dec'], unit = 'degree',
# #                      frame = 'icrs', obstime = 'J2015.4301')
# 
# 
# max_sep = 0.035*u.arcsec#!!!
# 
# idx,d2d,d3d = vvv_coord.match_to_catalog_sky(gns1_coor,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
# sep_constraint = d2d < max_sep
# gns_vvv = gns1_pm[idx[sep_constraint]]
# vvv_match = vvv_c[sep_constraint]
# 
# 
# 
# sig_cl = 3
# dKs = gns_vvv['Ks1']-vvv_match['Ks']
# mask_m, l_lim,h_lim = sigma_clip(dKs, sigma=sig_cl, masked = True, return_bounds= True)
# 
# 
# fig, (ax, ax1) = plt.subplots(1,2)
# ax.scatter(gns_vvv['ra1'],gns_vvv['Dec1'])
# ax.scatter(vvv_match['ra'],vvv_match['dec'],s =1)
# ax.set_title('Macthes = %s'%(len(vvv_match)))
# ax1.hist(dKs,bins = 'auto', label = '$\Delta$Ks = %.2f\n$\sigma$ = %.2f'%(np.mean(dKs),np.std(dKs)))
# ax1.axvline(l_lim, ls = 'dashed', color = 'r', label = '$\pm$ %s$\sigma$'%(sig_cl))
# ax1.axvline(h_lim, ls = 'dashed', color = 'r')
# ax1.legend(loc = 4, fontsize = 12)
# 
# vvv_match = vvv_match[np.logical_not(mask_m.mask)]
# gns_vvv = gns_vvv[np.logical_not(mask_m.mask)]
# ax1.set_title(f'{sig_cl}$\sigma$ Macthes= %s'%(len(vvv_match)))
# 
# 
# 
# 
# diff_pmx = gns_vvv['pm_RA'] - vvv_match['pmRA']
# diff_pmy = gns_vvv['pm_Dec'] - vvv_match['pmDEC']
# 
# 
# 
# 
# mask_pmx, l_lim,h_lim = sigma_clip(diff_pmx, sigma=3, masked = True, return_bounds= True)
# mask_pmy, l_lim,h_lim = sigma_clip(diff_pmy, sigma=3, masked = True, return_bounds= True)
# 
# mask_pm = np.logical_and(np.logical_not(mask_pmx.mask),np.logical_not(mask_pmy.mask))
# diff_pmx_clip = diff_pmx[mask_pm]
# diff_pmy_clip = diff_pmy[mask_pm]
# 
# # %
# fig, (ax,ax1) = plt.subplots(1,2)
# ax.set_title('Matching = %s'%(len(gns_vvv['pm_RA'])))
# ax1.set_title('VVV comparison')
# ax.hist(diff_pmx_clip, histtype = 'step', label = '$\overline{\Delta x}$ =%.2f\n$\sigma$ = %.2f '%(np.nanmean(diff_pmx_clip),np.nanstd(diff_pmx_clip)))
# ax.hist(diff_pmx, histtype = 'step', color ='k', alpha = 0.3)
# 
# ax1.hist(diff_pmy_clip, histtype = 'step',color = 'orange', label = '$\overline{\Delta x}$ =%.2f\n$\sigma$ = %.2f '%(np.nanmean(diff_pmy_clip),np.nanstd(diff_pmy_clip)))
# ax1.hist(diff_pmy, histtype = 'step',color = 'k', alpha = 0.3, ls = 'dashed')
# ax.legend()
# ax1.legend()
# ax.set_xlabel('$\Delta \mu_{\parallel}$')
# ax1.set_xlabel('$\Delta \mu_{\perp}$')
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# =============================================================================
