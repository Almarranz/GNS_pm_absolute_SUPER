#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:43:58 2024

@author: amartinez
"""

import numpy as np
from astropy.modeling.models import Polynomial2D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
import sys
from astropy.table import Table
from scipy.stats import binned_statistic_2d
from compare_lists import compare_lists 
# %%

field_one, chip_one, field_two, chip_two,t1,t2,max_sig = np.loadtxt('/Users/amartinez/Desktop/PhD/HAWK/GNS_1relative_python/lists/fields_and_chips.txt', 
                                                       unpack=True)
field_one = field_one.astype(int)
chip_one = chip_one.astype(int)
field_two = field_two.astype(int)
chip_two = chip_two.astype(int)

pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2relative/pruebas/'
pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1relative/pruebas/'

GNS_1relative = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1relative_python/lists/%s/chip%s/'%(field_one, chip_one)
GNS_2relative = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2relative_python/lists/%s/chip%s/'%(field_two, chip_two)

# gns1 = Table.read(GNS_1relative + 'stars_calibrated_HK_chip%s_on_gns2_f%sc%s_sxy%s.txt'%(chip_one,field_two,chip_two,max_sig), format = 'ascii')
# gns2 = Table.read(GNS_2relative +'stars_calibrated_H_chip%s_on_gns1_f%sc%s_sxy%s.txt'%(chip_two,field_one,chip_one,max_sig), format = 'ascii')


gns1 = Table.read(pruebas1 + 'gns1_trans.txt', format = 'ascii')
gns2 = Table.read(pruebas2 + 'gns2_trans.txt', format = 'ascii')

c_lst = compare_lists(np.array([gns1['x1'],gns1['y1']]).T,
                             np.array([gns2['x2'],gns2['y2']]).T,
                             1)

# %%
Xo = gns2['x2'][c_lst['ind_2']]
Yo = gns2['y2'][c_lst['ind_2']]
Xi = gns1['x1'][c_lst['ind_1']]
Yi = gns1['y1'][c_lst['ind_1']]
unc_x = gns1['dx1'][c_lst['ind_1']]
unc_y = gns1['dy1'][c_lst['ind_1']]

unc = np.sqrt(unc_x**2 + unc_y**2)

# Combine X and Y for fitting
X_input = np.array([Xi, Yi])
X_output = np.array([Xo, Yo])

# Polynomial degree
degree = 1

# Create the Polynomial2D model
model_x = Polynomial2D(degree=degree)
model_y = Polynomial2D(degree=degree)

# Linear least-squares fitter
fitter = LinearLSQFitter()
# fitter = fitting.LMLSQFitter()

# Fit the x and y transformations
fit_xw = fitter(model_x, Xi, Yi, Xo, weights= 1/unc)  # Fit x-coordinates
fit_yw = fitter(model_y, Xi, Yi, Yo, weights= 1/unc)    # Fit y-coordinates
fit_x = fitter(model_x, Xi, Yi, Xo, )  # Fit x-coordinates
fit_y = fitter(model_y, Xi, Yi, Yo, )    # Fit y-coordinates

# Apply the fitted model
Xi_c = fit_x(Xi, Yi)
Yi_c = fit_y(Xi, Yi)
Xi_cw = fit_xw(Xi, Yi)
Yi_cw = fit_yw(Xi, Yi)

fig, ax = plt.subplots()
ax.set_title('ORIGINAL')
# ax.hist(Xo-Xi,histtype = 'step',label = '$\overline{\Delta}=%.2e$\n$\sigma$=%.2e'%(np.mean(Xo-Xi),np.std(Xo-Xi)))
# ax.hist(Yo-Yi,histtype = 'step',label = '$\overline{\Delta}=%.2e$\n$\sigma$=%.2e'%(np.mean(Yo-Yi),np.std(Yo-Yi)))
ax.scatter(Xo,Yo)
ax.scatter(Xi,Yi, s=0.5)

fig, (ax, ax1) = plt.subplots(1,2)
ax.set_title('No weighted')
ax1.set_title('Weighted')
ax.hist(Xo-Xi_c,histtype = 'step',label = '$\overline{\Delta}=%.2e$\n$\sigma$=%.2e'%(np.mean(Xo-Xi_c),np.std(Xo-Xi_c)))
ax.hist(Yo-Yi_c,histtype = 'step',label = '$\overline{\Delta}=%.2e$\n$\sigma$=%.2e'%(np.mean(Yo-Yi_c),np.std(Yo-Yi_c)))

ax1.hist(Xo-Xi_cw,histtype = 'step',label = '$\overline{\Delta}=%.2e$\n$\sigma$=%.2e'%(np.mean(Xo-Xi_cw),np.std(Xo-Xi_cw)))
ax1.hist(Yo-Yi_cw,histtype = 'step',label = '$\overline{\Delta}=%.2e$\n$\sigma$=%.2e'%(np.mean(Yo-Yi_cw),np.std(Yo-Yi_cw)))

ax.legend()
ax1.legend()

gns1['x1_n'] = fit_x(gns1['x1'],gns1['y1']) 
gns1['y1_n'] = fit_y(gns1['x1'],gns1['y1']) 
gns1['x1_w'] = fit_xw(gns1['x1'],gns1['y1']) 
gns1['y1_w'] = fit_yw(gns1['x1'],gns1['y1']) 

c_lst_n = compare_lists(np.array([gns1['x1_n'],gns1['y1_n']]).T,
                             np.array([gns2['x2'],gns2['y2']]).T,
                             1)
c_lst_w = compare_lists(np.array([gns1['x1_w'],gns1['y1_w']]).T,
                             np.array([gns2['x2'],gns2['y2']]).T,
                             1)






# # %%
# v_min = 0
# v_max = 0.50
# num_bins = 10
# statistic, x_edges, y_edges, binnumber = binned_statistic_2d(Xo, Yo, abs(Xo-Xi_c), statistic='median', bins=(num_bins,int(num_bins/2)))

# # Create a meshgrid for plotting
# X, Y = np.meshgrid(x_edges, y_edges)

# # Plot the result
# fig, (ax,ax1) = plt.subplots(1,2)
# fig.suptitle('NO WEIGHTED')
# c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',vmin = v_min, vmax = v_max)
# fig.colorbar(c, ax=ax, label='$\Delta $ [pix]', shrink = 1)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.axis('equal')

# statistic, x_edges, y_edges, binnumber = binned_statistic_2d(Xo, Yo, abs(Yo-Yi_c), statistic='median', bins=(num_bins,int(num_bins/2)))
# X, Y = np.meshgrid(x_edges, y_edges)
# c = ax1.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',vmin = v_min, vmax = v_max)
# fig.colorbar(c, ax=ax1, label='$\Delta X$ [pix]', shrink = 1)
# ax1.set_xlabel('x')

# ax1.axis('equal')
# ax1.set_yticklabels([])


# num_bins = 10
# statistic, x_edges, y_edges, binnumber = binned_statistic_2d(Xo, Yo, abs(Xo-Xi_cw), statistic='median', bins=(num_bins,int(num_bins/2)))

# # Create a meshgrid for plotting
# X, Y = np.meshgrid(x_edges, y_edges)

# # Plot the result
# fig, (ax,ax1) = plt.subplots(1,2)
# fig.suptitle('WEIGHTED')

# c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r', vmin = v_min, vmax = v_max)
# fig.colorbar(c, ax=ax, shrink = 1)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.axis('equal')

# statistic, x_edges, y_edges, binnumber = binned_statistic_2d(Xo, Yo, abs(Yo-Yi_cw), statistic='median', bins=(num_bins,int(num_bins/2)))
# X, Y = np.meshgrid(x_edges, y_edges)
# c = ax1.pcolormesh(X, Y, statistic.T, cmap='Spectral_r', vmin = v_min, vmax = v_max)
# fig.colorbar(c, ax=ax1, label='$\Delta$ [pix]', shrink = 1)
# ax1.set_xlabel('x')
# # ax1.set_ylabel('y')
# ax1.axis('equal')
# ax1.set_yticklabels([])












