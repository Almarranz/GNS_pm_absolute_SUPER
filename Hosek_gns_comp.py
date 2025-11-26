#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 10:13:05 2025

@author: amartinez
"""


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
def sig_cl(x, y,s):
    mx, lx, hx = sigma_clip(x , sigma = s, masked = True, return_bounds= True)
    my, ly, hy = sigma_clip(y , sigma = s, masked = True, return_bounds= True)
    m_xy = np.logical_and(np.logical_not(mx.mask),np.logical_not(my.mask))
    
    return m_xy, [lx,hx,ly,hy]

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
e_pm_gns = 0.5
min_mag_gns = 18, 
max_mag_gns = 12
max_epos_gns = 0.05

# =============================================================================
# Hosek data
# =============================================================================
max_sep_vvv = 50*u.mas
e_pm_hos = e_pm_gns/2
max_e_pos = 1000e-3

mag_h = 'F153M'
min_mag_hos = 20,
max_mag_hos = 15,

# mag_h = 'F127M'
# min_mag_hos = 23,
# max_mag_hos = 15,

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


pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1absolute_SUPER/pruebas/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2absolute_SUPER/pruebas/'


 # gns1.write(pruebas1 + f'gns1_pmSuper_F1_{field_one}_F2_{field_two}.ecvs',format = 'ascii.ecsv', overwrite = True)
 # gns2.write(pruebas2 + f'gns2_pmSuper_F1_{field_one}_F2_{field_two}.ecvs',format = 'ascii.ecsv', overwrite = True)

gns1 = Table.read(pruebas1 + f'gns1_pmSuper_F1_{field_one}_F2_{field_two}.ecvs',format = 'ascii.ecsv')

gns1 = filter_gns_data(gns1, max_e_pm = e_pm_gns, min_mag = min_mag_gns, max_mag = max_mag_gns,max_e_pos = max_epos_gns)

# =============================================================================
# # Comparison with Hosek
# =============================================================================

catal='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/'

choosen_cluster = 'Arches'
# choosen_cluster = 'Quintuplet'
center_arc = SkyCoord(ra = '17h45m50.65020s', dec = '-28d49m19.51468s', equinox = 'J2000') if choosen_cluster =='Arches' else SkyCoord('17h46m 14.68579s', '-28d49m38.99169s', frame='icrs',obstime ='J2016.0')#Quintuplet

if choosen_cluster == 'Arches':
    arches = Table.read(catal + 'Arches_from_Article.txt', format = 'ascii')
if choosen_cluster == 'Quintuplet':
    arches = ascii.read(catal + 'Quintuplet_from_Article.txt')
arches = arches[arches['F127M']>10]
RA_DEC = center_arc.spherical_offsets_by(arches['dRA'], arches['dDE'])
RA = RA_DEC.ra
DEC = RA_DEC.dec

arches.add_column(RA,name = 'RA',index = 0)
arches.add_column(DEC,name = 'DEC',index = 1)



# %%

fig, (ax, ax2) = plt.subplots(1,2)
ax.set_title('HOSEK')
ax.hist2d(arches[mag_h], (arches['e_pmRA'] + arches['e_pmDE'])/2, bins = 100, norm = LogNorm())
ax2.hist2d(arches[mag_h], (arches['e_dRA']*1000 + arches['e_dDE']*1000)/2,  bins = 100, norm = LogNorm())
ax.set_ylabel('$\overline{\sigma\mu}_{l,b}$ [mas/yr]')
ax2.set_ylabel('$\overline{\sigma}_{l,b}$ [mas]')
ax.set_xlabel(f'[{mag_h}]')
ax2.set_xlabel(f'[{mag_h}]')
ax.axhline(e_pm_hos, ls = 'dashed', color = 'red')
ax2.axhline(max_e_pos*1000, ls = 'dashed', color = 'red')
ax.axvline(min_mag_hos,ls = 'dashed', color = 'red')
ax.axvline(max_mag_hos,ls = 'dashed', color = 'red')
ax2.axvline(min_mag_hos,ls = 'dashed', color = 'red')
ax2.axvline(max_mag_hos,ls = 'dashed', color = 'red')
ax.set_xticks(np.arange(14,27))
fig.tight_layout()
# ax.scatter(arches['F127M'], arches['e_pmRA'], label = 'Hosek',s=15)
# ax2.scatter(arches['F127M'], arches['e_dRA'], label = 'Hosek',s=15)

# %%


arches = filter_hosek_data(arches,
                      max_e_pos = max_e_pos,
                      max_e_pm = e_pm_hos,
                       min_mag = min_mag_hos,
                       max_mag = max_mag_hos,
                      max_Pclust = 1,
                      # center = 'yes'
                      ) 

obstimes = Time(arches['t0'], format='jyear')  
arches_gal = SkyCoord(ra=arches['RA'], dec=arches['DEC'],
                    pm_ra_cosdec =arches['pmRA'], pm_dec = arches['pmDE'],
                    unit = 'degree',frame = 'icrs', obstime=obstimes).galactic   

dt1 = t1_gns - obstimes

arches['l'] = arches_gal.l + arches_gal.pm_l_cosb*dt1
arches['b'] = arches_gal.b + arches_gal.pm_b*dt1
arches['pml'] = arches_gal.pm_l_cosb
arches['pmb'] = arches_gal.pm_b
# %%


fig, ax = plt.subplots(1,1)
# ax.scatter(gns1['l'], gns1['b'], label = 'GNS', facecolor = 'none', edgecolor = 'k',zorder = 2)
ax.scatter(gns1['l'], gns1['b'], label = 'GNS',s=1)
ax.scatter(arches['l'], arches['b'], label = 'Hosek',s=1)
ax.axis('scaled')
# sys.exit(219)
# %%


buenos1 = (gns1['l']>min(arches['l'])) & (gns1['l']<max(arches['l'])) & (gns1['b']>min(arches['b'])) & (gns1['b']<max(arches['b']))
gns1 = gns1[buenos1]

buenos2 = (arches['l']>min(gns1['l'])) & (arches['l']<max(gns1['l'])) & (arches['b']>min(gns1['b'])) & (arches['b']<max(gns1['b']))
arches = arches[buenos2]

# fig, ax = plt.subplots(1,1)
# ax.scatter(gns1['l'], gns1['b'], label = 'GNS', facecolor = 'none', edgecolor = 'k',zorder = 2)
# ax.scatter(arches['l'], arches['b'], label = 'Hosek',s=15)
# ax.axis('scaled')

#


arcc = SkyCoord(l = arches['l'], b = arches['b'], frame = 'galactic')
gns1c = SkyCoord(l = gns1['l'], b = gns1['b'], frame = 'galactic')
idx1, idx2, sep2d, _ = search_around_sky(arcc, gns1c, max_sep_vvv)

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

arch_m= arches[idx1_clean]
gns1_m = gns1[idx2_clean]

ax.scatter(arch_m['l'], arch_m['b'], facecolor = 'none', edgecolor = 'green')

print(40*'+')
unicos = unique(gns1_m, keep = 'first')
print(len(gns1_m),len(unicos))
print(40*'+')
  
dpm_x = arch_m['pml'] - gns1_m['pm_xp']
dpm_y = arch_m['pmb'] - gns1_m['pm_yp']



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
# ax.set_title(f'Matches {len(dpm_xm)}')
# ax2.set_title('ABSOLUTE')
ax.hist(dpm_x, color = 'k', alpha = 0.2, bins = 'auto')
ax.axvline(lim[0],color = 'r', ls = 'dashed' )
ax.axvline(lim[1],color = 'r', ls = 'dashed' )
ax.hist(dpm_xm, histtype='step', bins='auto', lw=2,
        label='$\overline{\Delta \mu_{l}}$ = %.2f'
              '\n$\sigma_l$ = %.2f' % 
              (np.mean(dpm_xm), np.std(dpm_xm)))
ax2.axvline(lim[2],color = 'r', ls = 'dashed' )
ax2.axvline(lim[3],color = 'r', ls = 'dashed' )

ax2.hist(dpm_y, color = 'k', alpha = 0.2, bins = 'auto')
ax2.hist(dpm_ym, histtype='step', bins='auto', lw=2,
         label='$\overline{\Delta \mu_{b}}$ = %.2f'
               '\n$\sigma_b$ = %.2f' % 
               (np.mean(dpm_ym), np.std(dpm_ym)))

ax.set_xlabel(r'$\Delta \mu_{l}$ [mas/yr]')
ax2.set_xlabel(r'$\Delta \mu_{b}$ [mas/yr]')
ax.set_ylabel('# stars')
ax.set_xlim(-3,3)
ax2.set_xlim(-3,3)
ax.legend(loc=1)
ax2.legend(loc=1)

fig.tight_layout()
meta = {'Script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_relative_SUPER/SUPER_alignment.py'}
plt.savefig('/Users/amartinez/Desktop/PhD/My_papers/GNS_pm_catalog/images/ABS_Hosek_resi_pm.png', dpi = 150, transparent = True, metadata = meta)














