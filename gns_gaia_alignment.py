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
from alignator_looping import alg_loop
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
field_one = 100
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
align_loop = 'yes'
# align_loop = 'no'
max_sep = 90*u.mas#!!!
max_deg = 5
# transf = 'affine'
transf = 'similarity'
# transf = 'polynomial'
order_trans = 3
# clip_in_alig = 'yes' # Clipps the 3sigmas in position during the alignment
clip_in_alig = None
bad_sig  = 3

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

bad = []
bad =  [15, 112, 189, 575, 689,65, 154]
if len(bad)>0:#!!!
    del_1 = np.isin(gaia['id'], bad)#!!!
    gaia = gaia[np.logical_not(del_1)]#!!!

e_pm = 0.1
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

gaia['pm_l'] = gaia_lb.pm_l_cosb
gaia['pm_b'] = gaia_lb.pm_b

tg = Time(['2016-01-01T00:00:00'],scale='utc')



# %%

def sig_cl(x, y,s):
    mx, lx, hx = sigma_clip(x , sigma = s, masked = True, return_bounds= True)
    my, ly, hy = sigma_clip(y , sigma = s, masked = True, return_bounds= True)
    m_xy = np.logical_and(np.logical_not(mx.mask),np.logical_not(my.mask))
    
    return m_xy, [lx,hx,ly,hy]

# Define catalogs and times as a list of tuples
catalogs = [
    {'name': 'GNS1', 'gns': gns1, 'time': t1_gns, 'tag': '1'},
    {'name': 'GNS2', 'gns': gns2, 'time': t2_gns, 'tag': '2'}
]

for c,cat in enumerate(catalogs):
    print(f"\n===== Aligning {cat['name']} =====")

    dt = cat['time'] - Time('2016-01-01T00:00:00', scale='utc')

    # Recalculate Gaia projected positions at catalog's epoch
    gaia_lb = SkyCoord(
        ra=gaia['ra'], dec=gaia['dec'],
        pm_ra_cosdec=gaia['pmra'], pm_dec=gaia['pmdec'],
        frame='icrs', obstime='J2016'
    ).galactic
    
    
    gaia_l = gaia_lb.l + gaia_lb.pm_l_cosb*dt.to(u.yr)
    gaia_b = gaia_lb.b + gaia_lb.pm_b*dt.to(u.yr)
    
    
    gaia_lbt = SkyCoord(l = gaia_l, b = gaia_b, frame = 'galactic')
    
    gaia[f'l{cat["tag"]}'] = gaia_lbt.l
    gaia[f'b{cat["tag"]}'] = gaia_lbt.b
    
    xp_g, yp_g = center.spherical_offsets_to(gaia_lbt.frame)
    
    gaia['xp'] = xp_g.to(u.arcsec) 
    gaia['yp'] = yp_g.to(u.arcsec) 
    
    gaia[f'xp_{c+1}'] = xp_g.to(u.arcsec) 
    gaia[f'yp_{c+1}'] = yp_g.to(u.arcsec) 
     

    # xp_g, yp_g = center.spherical_offsets_to(gaia_lb.frame)
    # gaia['xp'] = xp_g.to(u.arcsec) + gaia_lb.pm_l_cosb * dt.to(u.yr)
    # gaia['yp'] = yp_g.to(u.arcsec) + gaia_lb.pm_b * dt.to(u.yr)
    
    # gaia[f'xp_{c+1}'] = xp_g.to(u.arcsec) + gaia_lb.pm_l_cosb * dt.to(u.yr)
    # gaia[f'yp_{c+1}'] = yp_g.to(u.arcsec) + gaia_lb.pm_b * dt.to(u.yr)

    # ga_gtc = center.spherical_offsets_by(gaia['xp'], gaia['yp'])
    # gaia[f'l{cat["tag"]}'] = ga_gtc.l
    # gaia[f'b{cat["tag"]}'] = ga_gtc.b

    gns_cat = cat['gns']
    gaia_c = SkyCoord(l=gaia[f'l{cat["tag"]}'], b=gaia[f'b{cat["tag"]}'], frame='galactic')
    gns_c = SkyCoord(l=gns_cat['l'], b=gns_cat['b'], frame='galactic')

    idx, d2d, _ = gaia_c.match_to_catalog_sky(gns_c, nthneighbor=1)
    match_mask = d2d < max_sep
    gaia_m = gaia[match_mask]
    gns_m = gns_cat[idx[match_mask]]
    
    ga_w = gaia_c.l.wrap_at('360.1d')
    gns_w = gns_c.l.wrap_at('360.1d')
    
    fig, (ax, ax1, ax2) = plt.subplots(1,3, figsize = (15,5))
    fig.suptitle(f'GNS{c+1}')
    ax.scatter(gns_w[::100], gns_cat['b'][::100], alpha =0.1, color = 'k')
    ax.scatter(ga_w, gaia['b'], label = 'Gaia', s= 10)
    ax.set_title(f'Matches = {len(gns_m)}\nMin dist = {max_sep} ')
    ax.scatter(gns_m['l'], gns_m['b'], label = f'GNS{c} Match')
    ax.scatter(gaia_m['l'], gaia_m['b'],s =10, label = 'Gaia Match')
    ax.set_xlabel('l')
    ax.set_ylabel('b')
    ax.legend()

    # Fit transform
    if transf == 'polynomial':
        p = ski.transform.estimate_transform(
            transf, np.array([gns_m['xp'], gns_m['yp']]).T,
            np.array([gaia_m['xp'], gaia_m['yp']]).T, order=order_trans
        )
    else:
        p = ski.transform.estimate_transform(
            transf, np.array([gns_m['xp'], gns_m['yp']]).T,
            np.array([gaia_m['xp'], gaia_m['yp']]).T
        )
    print(p.params)


    gns_trans = p(np.array([gns_cat['xp'], gns_cat['yp']]).T)
    gns_cat['xp'] = gns_trans[:, 0]
    gns_cat['yp'] = gns_trans[:, 1]
    
    gns_xy = np.array([gns_cat['xp'],gns_cat['yp']]).T
    gaia_xy = np.array([gaia['xp'],gaia['yp']]).T
    xy_mat = compare_lists(gns_xy, gaia_xy, max_sep.to(u.arcsec).value)
    
    # for i in range(15):
        
        
    #     # p = ski.transform.estimate_transform(
    #     #     transf, np.array([gns_cat['xp'][xy_mat['ind_1']] , gns_cat['yp'][xy_mat['ind_1']] ]).T,
    #     #     np.array([gaia['xp'][xy_mat['ind_2']] , gaia['yp'][xy_mat['ind_2']] ]).T)
        
    #     p = ski.transform.estimate_transform(
    #         'polynomial', np.array([gns_cat['xp'][xy_mat['ind_1']] , gns_cat['yp'][xy_mat['ind_1']] ]).T,
    #         np.array([gaia['xp'][xy_mat['ind_2']] , gaia['yp'][xy_mat['ind_2']] ]).T, order = 2)
        
    #     gns_trans = p(np.array([gns_cat['xp'], gns_cat['yp']]).T)
    #     gns_cat['xp'] = gns_trans[:, 0]
    #     gns_cat['yp'] = gns_trans[:, 1]
    
    #     gns_xy = np.array([gns_cat['xp'],gns_cat['yp']]).T
    #     gaia_xy = np.array([gaia['xp'],gaia['yp']]).T
    #     xy_mat = compare_lists(gns_xy, gaia_xy, max_sep.to(u.arcsec).value)
        
    #     print(len(xy_mat))
    #     print(p.params)

        

    # sys.exit()
    ax1.set_title(f'Matches = {len(xy_mat)}\nMin dist = {max_sep} ')
    ax1.scatter(gns_cat['xp'][::100], gns_cat['yp'][::100], alpha =0.1, color = 'k')
    ax1.scatter(gaia['xp'], gaia['yp'],s =10, label = 'Gaia')
    ax1.scatter(gns_cat['xp'][xy_mat['ind_1']], gns_cat['yp'][xy_mat['ind_1']], label = 'GNS1 match')
    ax1.scatter(gaia['xp'][xy_mat['ind_2']], gaia['yp'][xy_mat['ind_2']],s =10, label = 'Gaia match')
    ax1.set_xlabel('xp[arcsec]')
    ax1.legend()
    
    if align_loop  == 'yes':
        # Optional final refinement
        # gns_cat = alg_rel(gns_cat, gaia, 'xp', 'yp', align, max_deg, max_sep.to(u.arcsec).value)
        gns_cat = alg_loop(gns_cat, gaia, 'xp', 'yp', align, max_deg, max_sep.to(u.arcsec).value)
        
    elif align_loop == 'no':
        p2 = ski.transform.estimate_transform('polynomial', 
            np.array([gns_cat['xp'][xy_mat['ind_1']], gns_cat['yp'][xy_mat['ind_1']]]).T,
            np.array([gaia['xp'][xy_mat['ind_2']], gaia['yp'][xy_mat['ind_2']]]).T, order=2)
        
        gns_trans = p2(np.array([gns_cat['xp'], gns_cat['yp']]).T)
        
        gns_cat['xp'] = gns_trans[:,0]
        gns_cat['yp'] = gns_trans[:,1]
    
    
    
    gns_xy = np.array([gns_cat['xp'],gns_cat['yp']]).T
    gaia_xy = np.array([gaia['xp'],gaia['yp']]).T
    xy_mat = compare_lists(gns_xy, gaia_xy, max_sep.to(u.arcsec).value)
    
    ax2.set_title(f'Matches = {len(xy_mat)}\nMin dist = {max_sep} ')
    ax2.scatter(gns_cat['xp'][::100], gns_cat['yp'][::100], alpha =0.1, color = 'k')
    ax2.scatter(gaia['xp'], gaia['yp'],s =10, label = 'Gaia')
    ax2.scatter(gns_cat['xp'][xy_mat['ind_1']], gns_cat['yp'][xy_mat['ind_1']], label = 'GNS1 match')
    ax2.scatter(gaia['xp'][xy_mat['ind_2']], gaia['yp'][xy_mat['ind_2']],s =10, label = 'Gaia match')
    ax2.set_xlabel('xp[arcsec]')
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('yp [arcsec]')
    ax2.legend()
    fig.tight_layout()

    # Residuals
    gns_xy = np.array([gns_cat['xp'], gns_cat['yp']]).T
    gaia_xy = np.array([gaia['xp'], gaia['yp']]).T
    xy_al = compare_lists(gns_xy, gaia_xy, max_sep.to(u.arcsec).value)

    d_x = (gaia['xp'][xy_al['ind_2']] - gns_cat['xp'][xy_al['ind_1']]).to(u.mas) 
    d_y = (gaia['yp'][xy_al['ind_2']] -gns_cat['yp'][xy_al['ind_1']] ).to(u.mas)
    
    sig_pm = 3
    # m_dx, l_dx, h_dx = sigma_clip(d_x, sigma = sig_pm, masked = True, return_bounds= True)
    # m_dy, l_dy, h_dy = sigma_clip(d_y, sigma = sig_pm, masked = True, return_bounds= True)
    # m_dxy = np.logical_and(np.logical_not(m_dx.mask),np.logical_not(m_dy.mask))
    
    m_dxy, lims = sig_cl(d_x,d_y, sig_pm)
    
    d_xm = d_x[m_dxy]
    d_ym = d_y[m_dxy]
    
    fig, (ax, ax1) = plt.subplots(1,2)
    ax.set_title(f'Gaia vs {cat['name']} (proyected)')
    ax1.set_title(f'Matching stars  = {len(d_xm)}')
    ax.hist(d_x,  color = 'grey', alpha = 0.5)
    ax1.hist(d_y, color = 'grey', alpha = 0.5)
    ax.hist(d_xm, histtype = 'step',label = '$\overline{\Delta x}$ = %.2f mas\n$\sigma$ = %.2f'%(np.mean(d_xm.value),np.std(d_xm.value)))
    ax1.hist(d_ym,histtype = 'step', label = '$\overline{\Delta y}$ = %.2f mas\n$\sigma$ = %.2f'%(np.mean(d_ym.value),np.std(d_ym.value)))
    ax.legend(); ax1.legend()
    ax.set_xlabel('$\Delta$xp [mas]')
    ax1.set_xlabel('$\Delta$yp [mas]')
    ax.axvline(lims[0].value, color = 'r', ls = 'dashed')
    ax.axvline(lims[1].value, color = 'r', ls = 'dashed')
    ax1.axvline(lims[2].value, color = 'r', ls = 'dashed')
    ax1.axvline(lims[3].value, color = 'r', ls = 'dashed')
    
    cat['gns']['xp'] =  gns_cat['xp']
    cat['gns']['yp'] =  gns_cat['yp']







# %%

# =============================================================================
# GNS proper motions
# =============================================================================
gns1_al = catalogs[0]['gns']
gns2_al = catalogs[1]['gns']
gns1_gxy  = np.array([gns1_al['xp'], gns1_al['yp']]).T 
gns2_gxy  = np.array([gns2_al['xp'], gns2_al['yp']]).T 

gns_com = compare_lists(gns1_gxy, gns2_gxy,0.150 )

gns1_gxy  = gns1_gxy[gns_com['ind_1']]  
gns2_gxy  = gns2_gxy[gns_com['ind_2']]  
gns1 = gns1_al[gns_com['ind_1']]
gns2 = gns2_al[gns_com['ind_2']]


pm_x = (gns_com['l2_x'] - gns_com['l1_x'])*u.arcsec.to(u.mas)/dt_gns.to(u.yr)
pm_y = (gns_com['l2_y'] - gns_com['l1_y'])*u.arcsec.to(u.mas)/dt_gns.to(u.yr)

gns1['pm_xp'] = pm_x
gns1['pm_yp'] = pm_y
gns2['pm_xp'] = pm_x
gns2['pm_yp'] = pm_y


m_pm, lpm = sig_cl(pm_x,pm_y, sig_pm)

pm_xm = pm_x[m_pm]
pm_ym = pm_y[m_pm]
# 
# %%
fig, (ax,ax2) = plt.subplots(1,2)


bins = 30
ax.hist(pm_x, bins = bins, color = 'grey', alpha = 0.3)
ax2.hist(pm_y, bins = bins, color = 'grey', alpha = 0.3)

ax.hist(pm_xm, bins = bins, histtype = 'step', label = '$\overline{\mu}_{xp}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(pm_xm.value),np.std(pm_xm.value)))
ax.axvline(lpm[0].value , ls = 'dashed', color = 'r')
ax.axvline(lpm[1].value , ls = 'dashed', color = 'r')

ax2.hist(pm_ym, bins = bins, histtype = 'step', label = '$\overline{\mu}_{yp}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(pm_ym.value),np.std(pm_ym.value)))
ax2.axvline(lpm[2].value , ls = 'dashed', color = 'r')
ax2.axvline(lpm[3].value , ls = 'dashed', color = 'r')

ax.legend()
ax2.legend()

ax.set_xlabel('$\mu_{xp}$ [mas/yr]')
ax2.set_xlabel('$\mu_{yp}$ [mas/yr]')
fig.tight_layout()

g_fac = 5# make the min distance 3 times bigger when comrin with Gaia

gns1_xy = np.array([gns1['xp'], gns1['yp']]).T
gaia1_xy = np.array([gaia['xp_1'], gaia['yp_1']]).T
gns1_ga = compare_lists(gns1_gxy, gaia1_xy, max_sep.to(u.arcsec).value*g_fac)

dpm_x = (gaia['pm_l'][gns1_ga['ind_2']] - gns1['pm_xp'][gns1_ga['ind_1']]) 
dpm_y = (gaia['pm_b'][gns1_ga['ind_2']] - gns1['pm_yp'][gns1_ga['ind_1']])
m_pm, lims = sig_cl(dpm_x, dpm_y, sig_pm)
dpm_xm = dpm_x[m_pm]
dpm_ym = dpm_y[m_pm]
bad_pm = np.logical_not(m_pm)



  
dx = (gaia['xp_1'][gns1_ga['ind_2']] - gns1['xp'][gns1_ga['ind_1']])*1e3
dy = (gaia['yp_1'][gns1_ga['ind_2']] - gns1['yp'][gns1_ga['ind_1']])*1e3    
m_xy, limx =  sig_cl(dx, dy, sig_pm)
dx_m = dx[m_xy]
dy_m = dy[m_xy]
bad_xy = np.logical_not(m_xy)


gns2_xy = np.array([gns2['xp'], gns2['yp']]).T
gaia2_xy = np.array([gaia['xp_2'], gaia['yp_2']]).T
gns2_ga = compare_lists(gns2_gxy, gaia2_xy, max_sep.to(u.arcsec).value*g_fac)
dx2 = (gaia['xp_2'][gns2_ga['ind_2']] - gns2['xp'][gns2_ga['ind_1']])*1e3
dy2 = (gaia['yp_2'][gns2_ga['ind_2']] - gns2['yp'][gns2_ga['ind_1']])*1e3
m_xy2, limx2 =  sig_cl(dx2, dy2, sig_pm)
dx_m2 = dx2[m_xy2]
dy_m2 = dy2[m_xy2]
bad_xy2 = np.logical_not(m_xy2)




# %
fig, (ax,ax2, ax3) = plt.subplots(1,3, figsize =(12,4))

ax.set_title('GNS-Gaia pm residuals')
ax.scatter(dpm_x,dpm_y, color = 'k', alpha = 0.3)
ax.scatter(dpm_xm,dpm_ym, label = '$\overline{\Delta \mu_{x}}$ = %.2f, $\sigma$ = %.2f\n''$\overline{\Delta \mu_{y}}$ = %.2f, $\sigma$ = %.2f'%(np.mean(dpm_xm),np.std(dpm_xm),np.mean(dpm_ym),np.std(dpm_ym)))
ax.axvline(lims[0], color = 'r', ls = 'dashed', label = f'{sig_pm}$\sigma$')
ax.axvline(lims[1], color = 'r', ls = 'dashed')
ax.axhline(lims[2], color = 'r', ls = 'dashed')
ax.axhline(lims[3], color = 'r', ls = 'dashed')


for x, y, label in zip(dpm_x[bad_pm],
                       dpm_y[bad_pm],
                       gaia['id'][gns1_ga['ind_2']][bad_pm]):
    print(x, y, label )
    ax.annotate(str(label), xy=(x, y), xytext=(1,1), textcoords='offset points',
                fontsize=8, color='black')

# ax.annotate(36, (5,5))
ax.legend()
ax.axis('equal')
ax.set_xlabel('$\Delta \mu_{x}$ [mas/yr]')
ax.set_ylabel('$\Delta \mu_{y}$ [mas/yr]')

for x, y, label in zip(dx[bad_xy],
                       dy[bad_xy],
                       gaia['id'][gns1_ga['ind_2']][bad_xy]):
    print(x, y, label )
    ax2.annotate(str(label), xy=(x, y), xytext=(1, 1), textcoords='offset points',
                fontsize=8, color='black')


ax2.set_title('GNS1-Gaia pos. residuals')
ax2.scatter(dx,dy, color = 'k', alpha = 0.3)
ax2.scatter(dx_m,dy_m, label = '$\overline{\Delta x}$ = %.2f, $\sigma$ = %.2f\n''$\overline{\Delta y}$ = %.2f, $\sigma$ = %.2f'%(np.mean(dx_m),np.std(dx_m),np.mean(dy_m),np.std(dy_m)))
ax2.axvline(limx[0], color = 'r', ls = 'dashed', label = f'{sig_pm}$\sigma$')
ax2.axvline(limx[1], color = 'r', ls = 'dashed')
ax2.axhline(limx[2], color = 'r', ls = 'dashed')
ax2.axhline(limx[3], color = 'r', ls = 'dashed')
ax2.legend()
ax2.axis('equal')
ax2.set_xlabel('$\Delta$x [mas]')
ax2.set_ylabel('$\Delta$x [mas]')

for x, y, label in zip(dx2[bad_xy2],
                       dy2[bad_xy2],
                       gaia['id'][gns2_ga['ind_2']][bad_xy2]):
    print(x, y, label )
    ax3.annotate(str(label), xy=(x, y), xytext=(1, 1), textcoords='offset points',
                fontsize=8, color='black')

ax3.set_title('GNS2-Gaia pos. residuals')
ax3.scatter(dx2,dy2, color = 'k', alpha = 0.3)
ax3.scatter(dx_m2,dy_m2, label = '$\overline{\Delta x}$ = %.2f, $\sigma$ = %.2f\n''$\overline{\Delta y}$ = %.2f, $\sigma$ = %.2f'%(np.mean(dx_m2),np.std(dx_m2),np.mean(dy_m2),np.std(dy_m2)))
ax3.axvline(limx2[0], color = 'r', ls = 'dashed', label = f'{sig_pm}$\sigma$')
ax3.axvline(limx2[1], color = 'r', ls = 'dashed')
ax3.axhline(limx2[2], color = 'r', ls = 'dashed')
ax3.axhline(limx2[3], color = 'r', ls = 'dashed')
ax3.legend()
ax3.axis('equal')
ax3.set_xlabel('$\Delta$x [mas]')
ax3.set_ylabel('$\Delta$x [mas]')

fig.tight_layout()

# %%
bad = []

for i in gaia['id'][gns1_ga['ind_2']][bad_pm]:
    bad.append(i)
    
for i in gaia['id'][gns1_ga['ind_2']][bad_xy]:
    bad.append(i)
for i in gaia['id'][gns2_ga['ind_2']][bad_xy2]:
    bad.append(i)




print(30*'☠️')
print('bad = ',[x.tolist() for x in np.unique(bad)])
print(30*'☠️')

# %%



