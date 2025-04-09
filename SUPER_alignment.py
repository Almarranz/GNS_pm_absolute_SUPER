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


max_sig = 0.05
# max_sig = 2

if field_one == 7 or field_one == 12 or field_one == 10 or field_one == 16:
    t1 = Time(['2015-06-07T00:00:00'],scale='utc')
elif field_one == 60:
    t1 = Time(['2016-06-13T00:00:00'],scale='utc')
elif field_one ==  100:
    t1 = Time(['2016-05-20T00:00:00'],scale='utc')
else:
    print(f'NO time detected for this field_one = {field_one}')
    sys.exit()
if field_two == 7 or field_two == 5:
    t2 = Time(['2022-05-27T00:00:00'],scale='utc')
elif field_two == 4:
    t2 = Time(['2022-04-05T00:00:00'],scale='utc')
elif field_two == 20:
    t2 = Time(['2022-07-25T00:00:00'],scale='utc')
else:
    print(f'NO time detected for this field_two = {field_two}')
    sys.exit()

dt = t2 - t1


# transf = 'affine'#!!!
# transf = 'similarity'#!!!1
transf = 'polynomial'#!!!
# transf = 'shift'#!!!
order_trans = 2


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


gns1 = Table.read(GNS_1 + 'stars_calibrated_H_chip%s.ecsv'%(chip_one),  format = 'ascii.ecsv')

unc_cut = np.where((gns1['sl']<max_sig) & (gns1['sb']<max_sig))
gns1 = gns1[unc_cut]

gns1_gal = SkyCoord(l = gns1['l'], b = gns1['b'], 
                    unit = 'degree', frame = 'galactic')


# %%
# gns2_all = np.loadtxt(GNS_2 + 'stars_calibrated_H_chip%s.txt'%(chip_two))

gns2 = Table.read(GNS_2 + 'stars_calibrated_H_chip%s.ecsv'%(chip_two), format = 'ascii')
unc_cut2 = np.where((gns2['sl']<max_sig) & (gns2['sb']<max_sig))
gns2 = gns2[unc_cut2]


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





buenos1 = np.where((gns1_gal.l>min(gns2_gal.l)) & (gns1_gal.l<max(gns2_gal.l)) &
                   (gns1_gal.b>min(gns2_gal.b)) & (gns1_gal.b<max(gns2_gal.b)))

gns1 = gns1[buenos1]
gns1['ID'] = np.arange(len(gns1))

buenos2 = np.where((gns2_gal.l>min(gns1_gal.l)) & (gns2_gal.l<max(gns1_gal.l)) &
                   (gns2_gal.b>min(gns1_gal.b)) & (gns2_gal.b<max(gns1_gal.b)))

gns2 = gns2[buenos2]
gns2['ID'] = np.arange(len(gns2))

l1 = l1[buenos1]
l2 = l2[buenos2]


radius = abs(np.min(gns1['l'])-np.max(gns1['l']))*0.6*u.degree
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
max_sep = 100*u.mas#!!!



idx,d2d,d3d = gns2_c.match_to_catalog_sky(gns1_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
sep_constraint = d2d < max_sep
gns2_m = gns2[sep_constraint]
gns1_m = gns1[idx[sep_constraint]]


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





xy_gn2 = np.array([gns2_m['x'],gns2_m['y']]).T
xy_gn1 = np.array([gns1_m['x'],gns1_m['y']]).T
# xy_gn2 = np.array([gns2_m['l'],gns2_m['b']]).T
# xy_gn1 = np.array([gns1_m['l'],gns1_m['b']]).T

N = 5
if transf == 'polynomial':
    p = ski.transform.estimate_transform(transf,
                                        xy_gn2[::N], 
                                        xy_gn1[::N], order = order_trans)
else:    
    p = ski.transform.estimate_transform(transf,
                                    xy_gn2[::N], 
                                    xy_gn1[::N])
    
print(p)

xy_gn2_t = p(xy_gn2)

gns2_m['l'] = xy_gn2_t[:,0]*u.deg
gns2_m['b'] = xy_gn2_t[:,1]*u.deg


mean_b  = np.cos((gns1_m['b'].to(u.rad) + gns2_m['b'].to(u.rad)) / 2.0)

dl = (gns2_m['l']- gns1_m['l']).to(u.mas)
db = (gns2_m['b']- gns1_m['b']).to(u.mas)

pm_l = (dl*mean_b)/dt.to(u.year)
pm_b = (db)/dt.to(u.year)

fig, (ax,ax2) = plt.subplots(1,2)
ax.hist(pm_l, bins = 'auto', histtype = 'step')
ax2.hist(pm_b, bins = 'auto', histtype = 'step')




# e_pm = 0.3
# gaia_good = filter_gaia_data(
#     gaia_table=gaia,
#     astrometric_params_solved=31,
#     duplicated_source= False,
#     parallax_over_error_min=-10,
#     astrometric_excess_noise_sig_max=2,
#     phot_g_mean_mag_min= None,
#     phot_g_mean_mag_max=13 ,
#     pm_min=0,
#     pmra_error_max=e_pm,
#     pmdec_error_max=e_pm
#     )














