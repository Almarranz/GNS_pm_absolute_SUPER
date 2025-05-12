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

def alignator(survey, gns,gaia,s_ls, d_m,max_deg, align_by = None,f_mode = None, plot = None, clipping = None, sig_clip = None):
    # Proceed with the iterative alignment process from an initial common lists
    
    # id1 = f'x{survey}'
    # id2 = f'y{survey}'
    # dx = f'dx{survey}'
    # dy = f'dy{survey}'
    id1 = 'xp'
    id2 = 'yp'
    dx = 'sl'
    dy = 'sb'
    loop =1
    deg = 1

    comom_ls =[]
    comom_ls.append(len(s_ls))
    dic_xy = {} 
    dic_xy['trans_0'] = np.array([gns[id1],gns[id2]]).T
    # for loop in range(1,10):
    if sig_clip == None:
        sig_cl = 3#!!!
    elif sig_clip is not None:
        sig_cl = sig_clip
    
    while deg < max_deg:
        
        diff_x =s_ls['l2_x'] - s_ls['l1_x']  
        diff_y =s_ls['l2_y'] - s_ls['l1_y']  
        
        mask_x, lx_lim,hx_lim = sigma_clip(diff_x, sigma=sig_cl, masked = True, return_bounds= True)
        mask_y, ly_lim,hy_lim = sigma_clip(diff_y, sigma=sig_cl, masked = True, return_bounds= True)
        
        # mask_xy = mask_x & mask_y # mask_xy = np.logical(mx, my)
        mask_xy = np.logical_and(np.logical_not(mask_x.mask), np.logical_not(mask_y.mask))
        
        
        
        if plot == 'yes':
            fig, (ax,ax1)  = plt.subplots(1,2)
            ax1.set_title(f'GNS{survey}. Degree = %s'%(deg))
            ax.set_title(f'Loop = {loop-1}. Matching stars = {len(s_ls)}')
            ax.hist(diff_x, histtype = 'step', label = '$\overline{x} = %.2f$\n$\sigma x$ =%.2f'%(np.mean(diff_x),np.std(diff_x)))
            ax.axvline(lx_lim, color = 'red', ls = 'dashed', label = '$\pm$%s$\sigma$'%(sig_cl))
            ax.axvline(hx_lim, color = 'red', ls = 'dashed')
            ax1.axvline(ly_lim, color = 'red', ls = 'dashed')
            ax1.axvline(hy_lim, color = 'red', ls = 'dashed')
           
            ax1.hist(diff_y, histtype = 'step', label = '$\overline{y} = %.2f$\n$\sigma y$ =%.2f'%(np.mean(diff_y),np.std(diff_y)))
            
            ax.set_xlabel('$\Delta x$ [mas]')
            ax1.set_xlabel('$\Delta y$ [mas]')
           
            if np.all(mask_xy) == False:
                
                diff_mx = diff_x[np.logical_not(mask_x.mask)]
                diff_my = diff_y[np.logical_not(mask_y.mask)]
                ax.hist(diff_mx , color = 'k', alpha = 0.5, lw = 10,label = '$\overline{x} = %.2f$\n$\sigma x$ =%.2f'%(np.mean(diff_mx),np.std(diff_mx)))
                ax1.hist(diff_my, color = 'k', alpha = 0.5, lw = 10,label = '$\overline{y} = %.2f$\n$\sigma y$ =%.2f'%(np.mean(diff_my),np.std(diff_my)))
            ax.legend()
            ax1.legend()
        if clipping is not None:
            if np.all(mask_xy) == False:
                # This line elimates the 3sigma alignment stars 
                s_ls = s_ls[mask_xy]
                print(10*'💀' + f'\nThere are {sig_cl} \u03C3 alignmet stars\n' + 10*'💀')
                    
    
        if align_by == 'Polywarp':
            Kx,Ky=pw.polywarp(s_ls['l2_x'],s_ls['l2_y'],s_ls['l1_x'],s_ls['l1_y'],degree=deg)
            
            xi=np.zeros(len(gns))
            yi=np.zeros(len(gns))
            
            for k in range(deg+1):
                        for m in range(deg+1):
                            xi=xi+Kx[k,m]*gns[id1]**k*gns[id2]**m
                            yi=yi+Ky[k,m]*gns[id1]**k*gns[id2]**m
            
            # dic_xy[f'trans_{loop}'] = np.array([xi,yi]).T    
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
                fit_xw = fitter(model_x, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_x'], weights= 1/np.sqrt(gns[dx][s_ls['ind_1']]**2 + gns[dy][s_ls['ind_1']]**2))  # Fit x-coordinates
                fit_yw = fitter(model_y, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_y'],weights= 1/np.sqrt(gns[dx][s_ls['ind_1']]**2 + gns[dy][s_ls['ind_1']]**2)) 
            
            elif f_mode == 'NW':
                #==============
                # Fit not weighted
                #==============
                fit_xw = fitter(model_x, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_x']) # Fit x-coordinates
                fit_yw = fitter(model_y, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_y']) 
            
            elif f_mode == 'WnC':
                #==============
                #Fit weighted and clipped
                #==============
                or_fit = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0)
                
                fit_xw, fm_x = or_fit(model_x, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_x'], weights= 1/np.sqrt(gns[dx][s_ls['ind_1']]**2 + gns[dy][s_ls['ind_1']]**2)) # Fit x-coordinates
                fit_yw, fm_y = or_fit(model_y, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_y'],weights= 1/np.sqrt(gns[dx][s_ls['ind_1']]**2 + gns[dy][s_ls['ind_1']]**2))
                
                fm_xy = np.logical_not(np.logical_and(np.logical_not(fm_x),np.logical_not(fm_y)))
                
                fig, ax = plt.subplots(1,1)
                plt_per = 20
                ax.set_title(f'Degree = {deg}')
                ax.scatter(s_ls['l1_x'][::plt_per],s_ls['l1_y'][::plt_per], label = 'Used (%s%%)'%(plt_per))
                ax.scatter(s_ls['l1_x'][fm_xy],s_ls['l1_y'][fm_xy],s =200, label = 'Clipped')
                ax.legend()
            elif f_mode == 'NWnC':
                or_fit = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0)
                
                fit_xw, fm_x = or_fit(model_x, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_x']) # Fit x-coordinates
                fit_yw, fm_y = or_fit(model_y, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_y']) 
                
                fm_xy = np.logical_not(np.logical_and(np.logical_not(fm_x),np.logical_not(fm_y)))
                
                fig, ax = plt.subplots(1,1)
                plt_per = 20
                ax.set_title(f'Degree = {deg}')
                ax.scatter(s_ls['l1_x'][::plt_per],s_ls['l1_y'][::plt_per], label = 'Used (%s%%)'%(plt_per))
                ax.scatter(s_ls['l1_x'][fm_xy],s_ls['l1_y'][fm_xy],s =200, label = 'Clipped')
                ax.legend()
                
            # sys.exit(308)
            
            xi = fit_xw(gns[id1], gns[id2])
            yi = fit_yw(gns[id1], gns[id2])# Fit y-coordinates
        
        dic_xy[f'trans_{loop}'] = np.array([xi,yi]).T    
            
        gns[id1] = xi
        gns[id2] = yi
            
            # l1_xy = np.array([gns[id1],gns[id2]]).T
            # g1_xy = np.array([gaia['x'], gaia['y']]).T
        
        
        
        l_xy = np.array([gns[id1],gns[id2]]).T
        ga_xy = np.array([gaia['x'], gaia['y']]).T
        
        
        
        s_ls = compare_lists(l_xy,ga_xy,d_m)
        print(f'\nCommon GNS{survey} and Gaia after loop{loop} = {len(s_ls)}')
                
        comom_ls.append(len(s_ls))
        # if comom_ls[-1] <= comom_ls[-2] :
        # # if comom_ls[-1] < comom_ls[-2]:
        #     if len(comom_ls)>2:
                
        #         gns[id1] = dic_xy[f'trans_{loop-1}'][:,0]
        #         gns[id2] = dic_xy[f'trans_{loop-1}'][:,1]
        #         deg +=1
        #         print(30*'-'+f'\nBreaking after loop = {loop-1}\nStarting alignment degree = {deg}\n' + 30*'-')
        #         continue
        #     else:
        #         print(f'Polynomial of degree {deg} does not work')
        #         gns[id1] = dic_xy['trans_0'][:,0]
        #         gns[id2] = dic_xy['trans_0'][:,1]
        #         break
        if comom_ls[-1] <= comom_ls[-2] :
            if len(comom_ls)>2:
            # if len(comom_ls)>10:
                
                # gns[id1] = dic_xy[f'trans_{loop-1}'][:,0]
                # gns[id2] = dic_xy[f'trans_{loop-1}'][:,1]
                deg +=1
                print(30*'-'+f'\nBreaking after loop = {loop-1}\nStarting alignment degree = {deg}\n' + 30*'-')
                continue
            # else:
            #     print(f'Polynomial of degree {deg} does not work')
            #     gns[id1] = dic_xy['trans_0'][:,0]
            #     gns[id2] = dic_xy['trans_0'][:,1]
            #     break
       
        
        
        # ax.legend()
        # ax1.legend()
        loop +=1     
        
    
    return gns