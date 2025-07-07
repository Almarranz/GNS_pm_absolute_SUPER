#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:44:21 2024

@author: amartinez
"""
import numpy as np
from compare_lists import compare_lists
import Polywarp as pw
import matplotlib.pyplot as plt
import sys
from astropy.stats import sigma_clip
from astropy.modeling.models import Polynomial2D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling import models, fitting
# from grid import grid_stars
"""
 Iteratively aling gns_A (source) to gns_B (destination)
 Parameters:
 -----------
 table : astropy.table.Table
     Input table containing star data.
 gns_A : table 
     Astropy Table, or similar 
 gns_B : table 
     Astropy Table, or similar 
 col1, col2:
     str: names  of the columns to be use for the alignment
 align_by : str
     'Polywarp' or '2DPoly'
 max_deg: int
     the maximun degree for the aligment -1 (if you set it at 3 it will align with a polynomial of grade 2)
 d_m: int of float
     max distance for the fine alignment betwenn list A and list B
 grid: 'yes' or 'no'
     use a grid for reference stars
 f_mode: W, WnC, WC, 'NWnC' or   'NW'
     different types of alignmet for the 2DPoli methin only
 Returns:
 --------
 astropy.table.Table
     aligned gns_A table
 """

def alg_loop(gns_A, gns_B,col1, col2, align_by,max_deg,d_m,max_loop, f_mode = None  ) :
    loop = 0
    deg = 1
    # max_loop= 10
    sig_cl = 3
    comom_ls = []
   
    l2_xy = np.array([gns_B[col1],gns_B[col2]]).T
    
    while deg < max_deg:
        loop += 1 
        l1_xy = np.array([gns_A[col1],gns_A[col2]]).T
        
        comp = compare_lists(l1_xy,l2_xy,d_m)
        if len(comom_ls) >1:
            # if comom_ls[-1] < comom_ls[-2]:
            if (comom_ls[-1] <= comom_ls[-2]) and loop >= max_loop:
                deg += 1
                loop = 0
                comom_ls =[]
                continue
               
        comom_ls.append(len(comp))
        print(f'Common in loop {loop}, degree {deg} = %s'%(len(comp['ind_1'])))
        # if loop == 1:
        #     with open(pruebas + 'sig_and_com.txt', 'a') as file:
        #         file.write('%.1f %.0f\n'%(sig_cl, len(comp['ind_1'])))
    
        l1_com = gns_A[comp['ind_1']]
       
        l2_com = gns_B[comp['ind_2']]
        
        l2_clip = l2_com
        l1_clip = l1_com
        
# =============================================================================
#         diff_mag = l1_com['H'] - l2_com['H'] 
#         # diff_mag1 = l1_com['IB230_diff'] - l2_com['IB230_diff'] 
#         diff_x =  l2_com['x'] - l1_com['x'] 
#         diff_y =  l2_com['y'] - l1_com['y'] 
#         diff_xy = (diff_x**2 + diff_y**2)**0.5
#         mask_m, l_lim,h_lim = sigma_clip(diff_mag, sigma=sig_cl, masked = True, return_bounds= True)
#         
#         l1_clip = l1_com[np.logical_not(mask_m.mask)]
#         l2_clip = l2_com[np.logical_not(mask_m.mask)]
#         
# =============================================================================
       
        
        
        
    # =============================================================================
    #         fig, (ax,ax1) = plt.subplots(1,2)
    #         fig.suptitle(f'Degree = {deg}. Loop = {loop}')
    #         ax.set_xlabel('$\Delta$ H')
    #         ax.hist(diff_mag, label = 'matches = %s\ndist = %.2f arcsec'%(len(comp['ind_1']), d_m*pix_scale))
    #         ax.axvline(l_lim, color = 'red', ls = 'dashed', label = '$\pm$%s$\sigma$'%(sig_cl))
    #         ax.axvline(h_lim, color = 'red', ls = 'dashed')
    #         ax.legend()
    #         
    #         ax1.hist(diff_x, label = '$\overline{\Delta x} = %.2f\pm%.2f$'%(np.mean(diff_x),np.std(diff_x)), histtype = 'step')
    #         ax1.hist(diff_y, label = '$\overline{\Delta y} = %.2f\pm%.2f$'%(np.mean(diff_y),np.std(diff_y)), histtype = 'step')
    #     
    #         ax1.set_xlabel('$\Delta$ pixel')
    #         ax1.legend()
    # =============================================================================
        
        
        
        
        xy_1c = np.array([l1_clip[col1],l1_clip[col2]]).T
        xy_2c = np.array([l2_clip[col1],l2_clip[col2]]).T
        
        if align_by == 'Polywarp':
            Kx,Ky=pw.polywarp(xy_2c[:,0],xy_2c[:,1],xy_1c[:,0],xy_1c[:,1],degree=deg)
            
            xi=np.zeros(len(gns_A))
            yi=np.zeros(len(gns_A))
            
            for k in range(deg+1):
                        for m in range(deg+1):
                            xi=xi+Kx[k,m]*gns_A[col1]**k*gns_A[col2]**m
                            yi=yi+Ky[k,m]*gns_A[col1]**k*gns_A[col2]**m
        elif align_by == '2DPoly':
            model_x = Polynomial2D(degree=deg)
            model_y = Polynomial2D(degree=deg)
            
            # Linear least-squares fitter
            fitter = LinearLSQFitter()
            # fitter = fitting.LMLSQFitter()
            
            if f_mode == 'W':
                #==============
                # Fit weighted
                #==============
                fit_xw = fitter(model_x, xy_1c[:,0],xy_1c[:,1], xy_2c[:,0], weights= 1/np.sqrt(l1_clip['sl']**2 + l1_clip['sb']**2))  # Fit x-coordinates
                fit_yw = fitter(model_y, xy_1c[:,0],xy_1c[:,1],  xy_2c[:,1],weights= 1/np.sqrt(l1_clip['sl']**2 + l1_clip['sb']**2)) 
            
            elif f_mode == 'NW':
                #==============
                # Fit not weighted
                #==============
                fit_xw = fitter(model_x, xy_1c[:,0],xy_1c[:,1], xy_2c[:,0])  # Fit x-coordinates
                fit_yw = fitter(model_y, xy_1c[:,0],xy_1c[:,1],  xy_2c[:,1]) 
            
            elif f_mode == 'WnC':
                #==============
                #Fit weighted and clipped
                #==============
                or_fit = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0)
                
                fit_xw, fm_x = or_fit(model_x, xy_1c[:,0],xy_1c[:,1], xy_2c[:,0], weights= 1/np.sqrt(l1_clip['sl']**2 + l1_clip['sb']**2))   # Fit x-coordinates
                fit_yw, fm_y = or_fit(model_y, xy_1c[:,0],xy_1c[:,1],  xy_2c[:,1],weights= 1/np.sqrt(l1_clip['sl']**2 + l1_clip['sb']**2)) 
                
                fm_xy = np.logical_not(np.logical_and(np.logical_not(fm_x),np.logical_not(fm_y)))
                
                fig, ax = plt.subplots(1,1)
                plt_per = 20
                ax.set_title(f'Degree = {deg}')
                ax.scatter(xy_1c[:,0][::plt_per],xy_1c[:,1][::plt_per], label = 'Used (%s%%)'%(plt_per))
                ax.scatter(xy_1c[:,0][fm_xy],xy_1c[:,1][fm_xy],s =200, label = 'Clipped')
                ax.legend()
            elif f_mode == 'NWnC':
                or_fit = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0)
                
                fit_xw, fm_x = or_fit(model_x, xy_1c[:,0],xy_1c[:,1], xy_2c[:,0])  # Fit x-coordinates
                fit_yw, fm_y = or_fit(model_y, xy_1c[:,0],xy_1c[:,1],  xy_2c[:,1]) 
                
                fm_xy = np.logical_not(np.logical_and(np.logical_not(fm_x),np.logical_not(fm_y)))
                
                fig, ax = plt.subplots(1,1)
                plt_per = 20
                ax.set_title(f'Degree = {deg}')
                ax.scatter(xy_1c[:,0][::plt_per],xy_1c[:,1][::plt_per], label = 'Used (%s%%)'%(plt_per))
                ax.scatter(xy_1c[:,0][fm_xy],xy_1c[:,1][fm_xy],s =200, label = 'Clipped')
                ax.legend()
                
            # sys.exit(308)
            
            xi = fit_xw(gns_A[col1], gns_A[col2])
            yi = fit_yw(gns_A[col1], gns_A[col2])# Fit y-coordinates
            
        
        
        # print(Kx[0][0])
        gns_A[col1] = xi
        gns_A[col2] = yi
        
    return gns_A    
   