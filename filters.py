#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:13:35 2024

@author: amartinez
Filter astropy Table data based on provided criteria.

Returns:
- Filtered Astropy Table.
"""


# gaia_filters.py

import numpy as np

def filter_gaia_data(gaia_table, 
                     astrometric_params_solved=None, 
                     duplicated_source=None, 
                     parallax_over_error_min=None, 
                     astrometric_excess_noise_sig_max=None, 
                     phot_g_mean_mag_min=None, 
                     phot_g_mean_mag_max=None, 
                     pm_min=None, 
                     pmra_error_max=None, 
                     pmdec_error_max=None):
    
    mask = np.ones(len(gaia_table), dtype=bool)
    print('pmra_error_max',pmra_error_max)
    if astrometric_params_solved is not None:
        mask &= (gaia_table['astrometric_params_solved'] == astrometric_params_solved)
        
    if duplicated_source is not None:
        mask &= (gaia_table['duplicated_source'] == duplicated_source)
        
    if parallax_over_error_min is not None:
        mask &= (gaia_table['parallax_over_error'] >= parallax_over_error_min)
        
    if astrometric_excess_noise_sig_max is not None:
        mask &= (gaia_table['astrometric_excess_noise_sig'] <= astrometric_excess_noise_sig_max)
        
    if phot_g_mean_mag_min is not None:
        mask &= (gaia_table['phot_g_mean_mag'] < phot_g_mean_mag_min)
        
    if phot_g_mean_mag_max is not None:
        mask &= (gaia_table['phot_g_mean_mag'] > phot_g_mean_mag_max)
        
    if pm_min is not None:
        mask &= (gaia_table['pm'] > pm_min)
        
    if pmra_error_max is not None:
        mask &= (gaia_table['pmra_error'] < pmra_error_max)
        
    if pmdec_error_max is not None:
        mask &= (gaia_table['pmdec_error'] < pmdec_error_max)
    
    return gaia_table[mask]

    
def filter_hosek_data(hosek_table,
                      max_e_pos = None,
                      max_e_pm = None,
                      min_mag = None,
                      max_mag = None,
                      max_Pclust = None,
                      center = None):

    mask = np.ones(len(hosek_table), dtype=bool)
    
    if max_e_pos is not None:
        mask &= (hosek_table['e_dRA'] < max_e_pos) & (hosek_table['e_dDE'] < max_e_pos) 
        
    if max_e_pm is not None:
        mask &= (hosek_table['e_pmRA']< max_e_pm) & (hosek_table['e_pmDE']< max_e_pm)
        
    if min_mag is not None:
        mask &= (hosek_table['F127M'] < min_mag)
        
    if max_mag is not None:
        mask &= (hosek_table['F127M'] > max_mag)
        
    if max_Pclust is not None:
        mask &= (hosek_table['Pclust'] < max_Pclust)
        
    if center is not None:
        mask &= (hosek_table['F127M'] - hosek_table['F153M'] > 1.7)
        
    
    return hosek_table[mask]
    

def filter_gns_data(gns_table,
                      max_e_pos = None,
                      max_e_pm = None,
                      min_mag = None,
                      max_mag = None,
                      max_Pclust = None,
                      center = None):

    mask = np.ones(len(gns_table), dtype=bool)
    
    if max_e_pos is not None:
        mask &= (gns_table['dx1'] < max_e_pos) & (gns_table['dy1'] < max_e_pos) 
    
    if min_mag is not None:
        mask &= (gns_table['H1'] < min_mag) 
    
    if max_mag is not None:
        mask &= (gns_table['H1'] > max_mag) 
        
    if center is not None:
        mask &= (gns_table['H1'] - gns_table['Ks1'] > 1.3)

    return gns_table[mask]

def filter_vvv_data(vvv_table,
                    pmRA = None,
                    pmDE = None,
                    epm = None,
                    ok = None,
                    max_Ks = None,
                    min_Ks = None,
                    J = None,
                    center = None
                    ):
    mask = np.ones(len(vvv_table), dtype = bool)
        
    if center is not None:
        mask &= (vvv_table['J'] - vvv_table['Ks'] > 2)
        
    if pmRA is not None:
        mask &= (vvv_table['pmRA'] < 900)
        
    if max_Ks is not None:
        mask &= (vvv_table['Ks'] > max_Ks)
        
    if min_Ks is not None:
        mask &= (vvv_table['Ks'] < min_Ks)
        
    if epm is not None:
        mask &= (vvv_table['epmRA'] <epm) & (vvv_table['epmDEC'] <epm)
        
    if ok is not None:
        mask &= (vvv_table['ok'] != 0)
        
        
    return vvv_table[mask]

    
    
    