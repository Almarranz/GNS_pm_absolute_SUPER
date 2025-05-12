#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 10:30:27 2022

@author: amartinez
"""

# Generates offsets for Gaia stars and astroaligned with xy coordinates in GNS1

# Here we are going to align GNS (1 and 2) to Gaia reference frame for the 
# each of tha gns epochs
import numpy as np
from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import sys
from matplotlib import rcParams
from astroquery.gaia import Gaia
import IPython
import os
# import cluster_finder
from filters import filter_gaia_data
import Polywarp as pw
from alignator import alignator
from alignator_relative import alg_rel
import skimage as ski
from astropy.table import Table
from compare_lists import compare_lists
from astropy.stats import sigma_clip
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
# IPython.get_ipython().run_line_magic('matplotlib', 'auto')
IPython.get_ipython().run_line_magic('matplotlib', 'inline')
# %%
field_one = 60
chip_one = 0
field_two = 20
chip_two = 0


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

# %%
# ===============================Constants=====================================
max_sig = 0.5
d_m = 12 #!!! pixeles are in mas
max_sep = 15*u.mas#!!!
max_deg = 2
# transf = 'affine'
transf = 'similarity'
# transf = 'polynomial'
order_trans = 2
# clip_in_alig = 'yes' # Clipps the 3sigmas in position during the alignment
clip_in_alig = None
bad_sig  = 3
# 
# Ks_lim = [12,14.5]
Ks_lim = [0,999]

d_pm = 150#!!! this is mas. Maximun separation for computing the proper motion

align = 'Polywarp'
# align = '2DPoly'
f_mode = 'WnC'

# =============================================================================

GNS_1='/Users/amartinez/Desktop/PhD/HAWK/GNS_1/lists/%s/chip%s/'%(field_one, chip_one)
GNS_2='/Users/amartinez/Desktop/PhD/HAWK/GNS_2/lists/%s/chip%s/'%(field_two, chip_two)

pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1relative_SUPER/pruebas/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2relative_SUPER/pruebas/'


gns1 = Table.read(GNS_1 + 'stars_calibrated_H_chip%s.ecsv'%(chip_one),  format = 'ascii.ecsv')
gns2 = Table.read(GNS_2 + 'stars_calibrated_H_chip%s.ecsv'%(chip_two), format = 'ascii.ecsv')

# m_mask = (gns1['H']>14) & (gns1['H']<17)
# gns1 = gns1[m_mask]

unc_cut1 = (gns1['sl']<max_sig) & (gns1['sb']<max_sig)
gns1 = gns1[unc_cut1]

unc_cut2 = (gns2['sl']<max_sig) & (gns2['sb']<max_sig)
gns2 = gns2[unc_cut2]

center = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic')
# center_1 = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic')
# center_2 = SkyCoord(l = np.mean(gns2['l']), b = np.mean(gns2['b']), unit = 'degree', frame = 'galactic')

gns1_lb = SkyCoord(l = gns1['l'], b = gns1['b'], unit ='deg', frame = 'galactic')
gns2_lb = SkyCoord(l = gns2['l'], b = gns2['b'], unit ='deg', frame = 'galactic')

xg_1, yg_1 = center.spherical_offsets_to(gns1_lb)
xg_2, yg_2 = center.spherical_offsets_to(gns2_lb)


gns1['xp'] = xg_1.to(u.arcsec)
gns1['yp'] = yg_1.to(u.arcsec)
gns2['xp'] = xg_2.to(u.arcsec)
gns2['yp'] = yg_2.to(u.arcsec)

radius = abs(np.min(gns1['l'])-np.max(gns1['l']))*1*u.degree

# %%
try:
    
    gaia = Table.read(pruebas1  + 'gaia_f1%s_f2%s_r%.0f.ecsv'%(field_one,field_two,radius.to(u.arcsec).value))
    print('Gaia from table')
except:
    print('Gaia from web')
    # center = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic').icrs

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" # Select early Data Release 3
    Gaia.ROW_LIMIT = -1  # it not especifty, Default rows are limited to 50. 
    j = Gaia.cone_search_async(center, radius = abs(radius))
    gaia = j.get_results()
    os.makedirs(pruebas1, exist_ok=True)
    gaia.write(pruebas1  + 'gaia_f1%s_f2%s_r%.0f.ecsv'%(field_one,field_two,radius.to(u.arcsec).value))

gaia['id'] = np.arange(len(gaia))

e_pm = 0.3
mag_min = 17


fig, ax2 = plt.subplots(1,1)
ax2.scatter(gaia['phot_g_mean_mag'],gaia['pmra_error'], s= 2, label = 'Gaia $\delta \mu_{ra}$')
ax2.scatter(gaia['phot_g_mean_mag'],gaia['pmdec_error'], s= 2, label = 'Gaia $\delta \mu_{dec}$')
ax2.axvline(mag_min, color = 'r', ls = 'dashed', label = 'pm cuts')
ax2.axhline(e_pm, color = 'r', ls = 'dashed')

ax2.set_xlabel('[G]')
ax2.set_ylabel('$\delta \mu$ [mas/yr]')

fig.tight_layout(pad=1.0)
lg = ax2.legend(markerscale=3.0)




gaia = filter_gaia_data(
    gaia_table=gaia,
    astrometric_params_solved=31,
    duplicated_source= False,
    parallax_over_error_min=-10,
    astrometric_excess_noise_sig_max=2,
    phot_g_mean_mag_min= mag_min,
    phot_g_mean_mag_max=None,
    pm_min=0,
    pmra_error_max=e_pm,
    pmdec_error_max=e_pm
    )


gaia_lb = SkyCoord(ra = gaia['ra'], dec = gaia['dec'],
                   pm_ra_cosdec = gaia['pmra'],
                   pm_dec = gaia['pmdec'], 
                   frame = 'icrs', obstime = 'J2016').galactic
xp_g, yp_g = center.spherical_offsets_to(gaia_lb.frame)
gaia['xp'] = xp_g.to(u.arcsec)
gaia['yp'] = yp_g.to(u.arcsec)

tg = Time(['2016-01-01T00:00:00'],scale='utc')

# %%

# =============================================================================
# Aling with GNS1
# =============================================================================

dt1 = t1_gns-tg

gaia['xp'] = gaia['xp'].to(u.mas) + (gaia_lb.pm_l_cosb)*dt1.to(u.yr)
gaia['yp'] = gaia['yp'].to(u.mas) + (gaia_lb.pm_b)*dt1.to(u.yr)
ga_gtc = center.spherical_offsets_by(gaia['xp'], gaia['yp'])


gaia['l'] = ga_gtc.l
gaia['b'] = ga_gtc.b

gaia_c = SkyCoord(l = gaia['l'], b = gaia['b'], frame = 'galactic')
gns1_c = SkyCoord(l = gns1['l'], b = gns1['b'], frame = 'galactic')
# Gaia comparison

idx,d2d,d3d = gaia_c.match_to_catalog_sky(gns1_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
sep_constraint = d2d < max_sep
gaia_m = gaia[sep_constraint]
gns1_m = gns1[idx[sep_constraint]]



ga_w = gaia_c.l.wrap_at('360.1d')
gns1_w = gns1_c.l.wrap_at('360.1d')

fig, (ax, ax1) = plt.subplots(1,2)
fig.suptitle('GNS1')
ax.scatter(gns1_w[::100], gns1['b'][::100], alpha =0.1, color = 'k')
ax.scatter(ga_w, gaia['b'], label = 'Gaia', s= 10)
ax.set_title(f'Matches = {len(gns1_m)}\nMin dist = {max_sep} ')
ax.scatter(gns1_m['l'], gns1_m['b'], label = 'GNS1 Match')
ax.scatter(gaia_m['l'], gaia_m['b'],s =10, label = 'Gaia Match')
ax.set_xlabel('l')
ax.set_ylabel('b')
ax.legend()

if transf == 'polynomial':
    p = ski.transform.estimate_transform(transf,
                                        np.array([gns1_m['xp'],gns1_m['yp']]).T, 
                                        np.array([gaia_m['xp'],gaia_m['yp']]).T, order = order_trans)
else:    
    p = ski.transform.estimate_transform(transf,
                                        np.array([gns1_m['xp'],gns1_m['yp']]).T, 
                                        np.array([gaia_m['xp'],gaia_m['yp']]).T)    
print(p)

gns1_xyt = p(np.array([gns1['xp'],gns1['yp']]).T)

gns1['xp'] = gns1_xyt[:,0]
gns1['yp'] = gns1_xyt[:,1]


gns1_xy = np.array([gns1['xp'],gns1['yp']]).T
gaia_xy = np.array([gaia['xp'],gaia['yp']]).T
xy_mat = compare_lists(gns1_xy, gaia_xy, max_sep.to(u.arcsec).value)




ax1.set_title(f'Matches = {len(xy_mat)}\nMin dist = {max_sep} ')
ax1.scatter(gns1['xp'][::100], gns1['yp'][::100], alpha =0.1, color = 'k')
ax1.scatter(gaia['xp'], gaia['yp'],s =10, label = 'Gaia')
ax1.scatter(gns1['xp'][xy_mat['ind_1']], gns1['yp'][xy_mat['ind_1']], label = 'GNS1 match')
ax1.scatter(gaia['xp'][xy_mat['ind_2']], gaia['yp'][xy_mat['ind_2']],s =10, label = 'Gaia match')
ax1.set_xlabel('xp[arcsec]')
ax1.set_ylabel('yp [arcsec]')
ax1.legend()
fig.tight_layout()

# def alg_rel(gns_A, gns_B,col1, col2, align_by,max_deg,d_m, f_mode = None  ) :
gns1_al = alg_rel(gns1, gaia, 'xp', 'yp', 'Polywarp',max_deg,max_sep.to(u.arcsec).value)


gns1_xy = np.array([gns1_al['xp'],gns1_al['yp']]).T
xy_al = compare_lists(gns1_xy, gaia_xy, max_sep.to(u.arcsec).value)

d_x = (gaia['xp'][xy_al['ind_2']] - gns1_al['xp'][xy_al['ind_1']]).to(u.mas) 
d_y = (gaia['yp'][xy_al['ind_2']] -gns1_al['yp'][xy_al['ind_1']] ).to(u.mas)

sig_pm = 3
m_dx, l_dx, h_dx = sigma_clip(d_x, sigma = sig_pm, masked = True, return_bounds= True)
m_dy, l_dy, h_dy = sigma_clip(d_y, sigma = sig_pm, masked = True, return_bounds= True)
m_dxy = np.logical_and(np.logical_not(m_dx.mask),np.logical_not(m_dy.mask))

d_xm = d_x[m_dxy]
d_ym = d_y[m_dxy]

fig, (ax, ax1) = plt.subplots(1,2)
ax.set_title('Gaia vs GNS1 (proyected)')
ax1.set_title(f'Matching stars  = {len(d_xm)}')
ax.hist(d_x,  color = 'grey', alpha = 0.5)
ax1.hist(d_y, color = 'grey', alpha = 0.5)
ax.hist(d_xm, histtype = 'step',label = '$\overline{\Delta x}$ = %.2f mas\n$\sigma$ = %.2f'%(np.mean(d_xm.value),np.std(d_xm.value)))
ax1.hist(d_ym,histtype = 'step', label = '$\overline{\Delta y}$ = %.2f mas\n$\sigma$ = %.2f'%(np.mean(d_ym.value),np.std(d_ym.value)))
ax.legend()
ax1.legend()
ax.set_xlabel('$\Delta$xp [mas]')
ax1.set_xlabel('$\Delta$yp [mas]')




sys.exit(317)
# =============================================================================
# Alind with GNS2
# =============================================================================

dt2 = t2_gns-tg

l_offt = gaia['xp'].to(u.mas) + (gaia_lb.pm_l_cosb)*dt2.to(u.yr)
b_offt = gaia['yp'].to(u.mas) + (gaia_lb.pm_b)*dt2.to(u.yr)
ga_gtc = center.spherical_offsets_by(l_offt, b_offt)


gaia['l2'] = ga_gtc.l
gaia['b2'] = ga_gtc.b

gaia2_c = SkyCoord(l = gaia['l2'], b = gaia['b2'], frame = 'galactic')
gns2_c = SkyCoord(l = gns2['l'], b = gns2['b'], frame = 'galactic')
# Gaia comparison

idx,d2d,d3d = gaia2_c.match_to_catalog_sky(gns2_c,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 matchsep_constraint = d2d < max_sep
sep_constraint = d2d < max_sep
gaia2_m = gaia[sep_constraint]
gns2_m = gns1[idx[sep_constraint]]



ga_w = gaia2_c.l.wrap_at('360.1d')
gns2_w = gns2_c.l.wrap_at('360.1d')

fig, (ax, ax1) = plt.subplots(1,2)
fig.suptitle('GNS2')
ax.scatter(gns2_w[::100], gns2['b'][::100], alpha =0.1, color = 'k')
ax.scatter(ga_w, gaia['b'], label = 'Gaia', s= 10)
ax.set_title(f'Matches = {len(gns1_m)}\nMin dist = {max_sep} ')
ax.scatter(gns2_m['l'], gns2_m['b'], label = 'GNS1 Match')
ax.scatter(gaia2_m['l'], gaia2_m['b'],s =10, label = 'Gaia Match')
ax.set_xlabel('l')
ax.set_ylabel('b')
ax.legend()

if transf == 'polynomial':
    p = ski.transform.estimate_transform(transf,
                                        np.array([gns2_m['xp'],gns2_m['yp']]).T, 
                                        np.array([gaia2_m['xp'],gaia2_m['yp']]).T, order = order_trans)
else:    
    p = ski.transform.estimate_transform(transf,
                                        np.array([gns2_m['xp'],gns2_m['yp']]).T, 
                                        np.array([gaia2_m['xp'],gaia2_m['yp']]).T)    
print(p)

gns2_xyt = p(np.array([gns2['xp'],gns2['yp']]).T)

gns2['xp'] = gns2_xyt[:,0]
gns2['yp'] = gns2_xyt[:,1]


gns2_xy = np.array([gns2['xp'],gns2['yp']]).T
gaia2_xy = np.array([gaia['xp'],gaia['yp']]).T
xy_mat = compare_lists(gns2_xy, gaia2_xy, max_sep.to(u.arcsec).value)




ax1.set_title(f'Matches = {len(xy_mat)}\nMin dist = {max_sep} ')
ax1.scatter(gns1['xp'][::100], gns1['yp'][::100], alpha =0.1, color = 'k')
ax1.scatter(gaia['xp'], gaia['yp'],s =10, label = 'Gaia')
ax1.scatter(gns1['xp'][xy_mat['ind_1']], gns1['yp'][xy_mat['ind_1']], label = 'GNS1 match')
ax1.scatter(gaia['xp'][xy_mat['ind_2']], gaia['yp'][xy_mat['ind_2']],s =10, label = 'Gaia match')
ax1.set_xlabel('xp[arcsec]')
ax1.set_ylabel('yp [arcsec]')
ax1.legend()
fig.tight_layout()

# def alg_rel(gns_A, gns_B,col1, col2, align_by,max_deg,d_m, f_mode = None  ) :
gns1_al = alg_rel(gns1, gaia, 'xp', 'yp', 'Polywarp',max_deg,max_sep.to(u.arcsec).value)


gns1_xy = np.array([gns1_al['xp'],gns1_al['yp']]).T
xy_al = compare_lists(gns1_xy, gaia_xy, max_sep.to(u.arcsec).value)

d_x = (gaia['xp'][xy_al['ind_2']] - gns1_al['xp'][xy_al['ind_1']]).to(u.mas) 
d_y = (gaia['yp'][xy_al['ind_2']] -gns1_al['yp'][xy_al['ind_1']] ).to(u.mas)

sig_pm = 3
m_dx, l_dx, h_dx = sigma_clip(d_x, sigma = sig_pm, masked = True, return_bounds= True)
m_dy, l_dy, h_dy = sigma_clip(d_y, sigma = sig_pm, masked = True, return_bounds= True)
m_dxy = np.logical_and(np.logical_not(m_dx.mask),np.logical_not(m_dy.mask))

d_xm = d_x[m_dxy]
d_ym = d_y[m_dxy]

fig, (ax, ax1) = plt.subplots(1,2)
ax.set_title('Gaia vs GNS1 (proyected)')
ax1.set_title(f'Matching stars  = {len(d_xm)}')
ax.hist(d_x,  color = 'grey', alpha = 0.5)
ax1.hist(d_y, color = 'grey', alpha = 0.5)
ax.hist(d_xm, histtype = 'step',label = '$\overline{\Delta x}$ = %.2f mas\n$\sigma$ = %.2f'%(np.mean(d_xm.value),np.std(d_xm.value)))
ax1.hist(d_ym,histtype = 'step', label = '$\overline{\Delta y}$ = %.2f mas\n$\sigma$ = %.2f'%(np.mean(d_ym.value),np.std(d_ym.value)))
ax.legend()
ax1.legend()
ax.set_xlabel('$\Delta$xp [mas]')
ax1.set_xlabel('$\Delta$yp [mas]')












# %%


# ga_w = np.where(gaia['l'] > 180, gaia['l'] - 360, gaia['l'])
# g2_w = np.where(gns2['l'] > 180, gns2['l'] - 360, gns2['l'])
# g1_w = np.where(gns1['l'] > 180, gns1['l'] - 360, gns1['l'])

# fig, ax = plt.subplots(1,1)
# ax.scatter(g2_w,gns2['b'].value)
# ax.scatter(g1_w,gns1['b'])
# ax.scatter(ga_w,gaia['b'])





