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


# max_sig = 0.0005
max_sig = 2

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
# transf = 'similarity'#!!!1|
transf = 'polynomial'#!!!
# transf = 'shift'#!!!
order_trans = 1


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


# m_mask = gns1['H']<17

# gns1 = gns1[m_mask]
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
    phot_g_mean_mag_max=13 ,
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


for i in range(1, 4):
    idx,d2d,d3d = gns2_c.match_to_catalog_sky(gns1_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
    sep_constraint = d2d < max_sep
    gns2_m = gns2[sep_constraint]
    gns1_m = gns1[idx[sep_constraint]]
    
    print(f'MATCHES i = {i-1} = {len(gns2_m)}')
    
    
    # diff_H = gns2_m['H']-gns1_m['H']
    # off_s = np.mean(diff_H)
    # gns1_m['H'] = gns1_m['H'] + off_s
    # diff_H = gns2_m['H'] - gns1_m['H']
    
    # diff_H = gns2_m['H']-gns1_m['H']
    # sig_cl = 2
    # mask_H, l_lim,h_lim = sigma_clip(diff_H, sigma=sig_cl, masked = True, return_bounds= True)
    
    # gns2_m = gns2_m[mask_H.mask]
    # gns1_m = gns1_m[mask_H.mask]
    
    # fig,ax = plt.subplots(1,1)
    # ax.hist(diff_H, bins = 'auto',histtype = 'step')
    # ax.axvline(np.mean(diff_H), color = 'k', ls = 'dashed', label = 'H offset = %.2f\n$\sigma$ = %.2f'%(off_s,np.std(diff_H)))
    # ax.axvline(l_lim, ls = 'dashed', color ='r')
    # ax.axvline(h_lim, ls = 'dashed', color ='r')
    # ax.legend() 
    
    
    
    
    
    # xy_gn2 = np.array([gns2_m['x'],gns2_m['y']]).T
    # xy_gn1 = np.array([gns1_m['x'],gns1_m['y']]).T
    lb_gn2 = np.array([gns2_m['l'],gns2_m['b']]).T
    lb_gn1 = np.array([gns1_m['l'],gns1_m['b']]).T
    
    N = 30
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
    gns1_m = gns1[idx[sep_constraint]]
    
    print(f'MATCHES i = {i}= {len(gns2_m)}')


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

bins = 20
# %%
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
max_sep = 50*u.mas
idx,d2d,d3d = gaia_c.match_to_catalog_sky(gns1_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
sep_constraint = d2d < max_sep
gaia_m = gaia[sep_constraint]
gg_m = gns1_m[idx[sep_constraint]]

diff_l = gaia_m['pm_l'] - gg_m['pm_l']
diff_b = gaia_m['pm_b'] - gg_m['pm_b']


m_dl, l_dl, h_dl = sigma_clip(diff_l, sigma = sig_pm, masked = True, return_bounds= True)
m_db, l_db, h_db = sigma_clip(diff_b, sigma = sig_pm, masked = True, return_bounds= True)
m_dpm = np.logical_and(np.logical_not(m_dl.mask),np.logical_not(m_db.mask))

diff_lm = diff_l[m_dpm]
diff_bm = diff_b[m_dpm]

# diff_lm = diff_l
# diff_bm = diff_b
# %
fig, (ax,ax2) = plt.subplots(1,2)
# ax.hist(d, bins = bins, color = 'k', alpha = 0.2)
# ax2.hist(pm_b, bins = bins,color = 'k', alpha = 0.2)
ax.hist(diff_lm, bins = bins, histtype = 'step', label = '$\Delta{\mu}_{l}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(diff_lm.value),np.std(diff_lm.value)))
ax2.hist(diff_bm, bins = bins, histtype = 'step',label = '$\Delta{\mu}_{b}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(diff_bm.value),np.std(diff_bm.value)))
ax.set_xlabel('$\Delta \mu_{l}$ [mas/yr]')
ax2.set_xlabel('$\Delta\mu_{b}$ [mas/yr]')
ax.axvline(l_pml.value, ls = 'dashed', color = 'r')
ax.axvline(h_pml.value, ls = 'dashed', color = 'r')
ax2.axvline(l_pmb.value, ls = 'dashed', color = 'r')
ax2.axvline(h_pmb.value, ls = 'dashed', color = 'r')
ax.legend()
ax2.legend()











