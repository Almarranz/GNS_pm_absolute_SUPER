#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:18:16 2023

@author: amartinez
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import sys
from astropy.table import Table
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FormatStrFormatter
from astropy.io import ascii
import astropy.coordinates as ap_coor
from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
# import alphashape
from astropy.table import Table
# import species
# %%plotting pa    metres
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
rcParams.update({'figure.figsize': (10, 5)})
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})


def finder(pmra, pmdec, x, y, Ra, Dec, clustered_by,color_A, color_B,
           samples_dist, gen_sim, sim_lim):

    coordenadas = SkyCoord(ra=Ra, dec=Dec, unit='degree')
    pmra_kernel, pmdec_kernel = gaussian_kde(pmra), gaussian_kde(pmdec)
    x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
    colorines  = color_A-color_B
   
    if clustered_by == 'pm_xy_color':
        X = np.array([pmra, pmdec, x, y, colorines]).T
        X = np.array([pmra, pmdec, x, y, colorines]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2)
        # DistNnce to the 1,2,3...k neighbour
        dist, ind = tree.query(X_stad, k=samples_dist)
        d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour
    elif clustered_by == 'pm_xy':
        X = np.array([pmra, pmdec, x, y]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2)
        # DistNnce to the 1,2,3...k neighbour
        dist, ind = tree.query(X_stad, k=samples_dist)
        d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour
    elif clustered_by == 'pm_color':
        X = np.array([pmra, pmdec, colorines]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2)
        # DistNnce to the 1,2,3...k neighbour
        dist, ind = tree.query(X_stad, k=samples_dist)
        d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour

    elif clustered_by == 'pm':
        X = np.array([pmra, pmdec]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2)
        # DistNnce to the 1,2,3...k neighbour
        dist, ind = tree.query(X_stad, k=samples_dist)
        d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour

    lst_d_KNN_sim = []
    if gen_sim == 'kernnel':
        for d in range(20):
            mudec_sim,  mura_sim = pmdec_kernel.resample(
                len(pmdec)), pmra_kernel.resample(len(pmra))
            x_sim, y_sim = x_kernel.resample(len(x)), y_kernel.resample(len(y))
            
            if clustered_by == 'pm_xy_color':
                color_kernel = gaussian_kde(colorines)
                color_sim = color_kernel.resample(len(pmdec))
                X_sim = np.array([mura_sim[0], mudec_sim[0],
                                 x_sim[0], y_sim[0], color_sim[0]]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim = KDTree(X_stad_sim, leaf_size=2)

                # DistNnce to the 1,2,3...k neighbour
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
                # distance to the Kth neighbour
                d_KNN_sim = sorted(dist_sim[:, -1])

                lst_d_KNN_sim.append(min(d_KNN_sim))
            elif clustered_by == 'pm_xy':
                X_sim = np.array(
                    [mura_sim[0], mudec_sim[0], x_sim[0], y_sim[0]]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim = KDTree(X_stad_sim, leaf_size=2)

                # DistNnce to the 1,2,3...k neighbour
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
                # distance to the Kth neighbour
                d_KNN_sim = sorted(dist_sim[:, -1])

                lst_d_KNN_sim.append(min(d_KNN_sim))
            elif clustered_by == 'pm_color':
                color_kernel = gaussian_kde(colorines)
                color_sim = color_kernel.resample(len(pmdec))
                X_sim = np.array([mura_sim[0], mudec_sim[0], color_sim[0]]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim = KDTree(X_stad_sim, leaf_size=2)

                # DistNnce to the 1,2,3...k neighbour
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
                # distance to the Kth neighbour
                d_KNN_sim = sorted(dist_sim[:, -1])

                lst_d_KNN_sim.append(min(d_KNN_sim))

            elif clustered_by == 'pm':
                X_sim = np.array([mura_sim[0], mudec_sim[0]]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim = KDTree(X_stad_sim, leaf_size=2)

                # DistNnce to the 1,2,3...k neighbour
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
                # distance to the Kth neighbour
                d_KNN_sim = sorted(dist_sim[:, -1])

                lst_d_KNN_sim.append(min(d_KNN_sim))

    d_KNN_sim_av = np.mean(lst_d_KNN_sim)
    if sim_lim == 'mean':
        eps_av = round((min(d_KNN)+d_KNN_sim_av)/2, 3)
        valor = d_KNN_sim_av
    elif sim_lim == 'minimun':
        eps_av = round((min(d_KNN)+min(lst_d_KNN_sim))/2, 6)
        valor = min(lst_d_KNN_sim)
    elif sim_lim == 'maximun':
        eps_av = round((min(d_KNN)+max(lst_d_KNN_sim))/2, 3)
        valor = min(lst_d_KNN_sim)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(d_KNN, bins='auto', histtype='step', color='k')
    ax.hist(d_KNN_sim, bins='auto', histtype='step', color='r')
    ax.set_xlabel('%s-NN distance' % (samples_dist))
    ax.set_xlim(0,2)
    
    clus_method = 'dbs'

    clustering = DBSCAN(eps=eps_av, min_samples=samples_dist).fit(X_stad)
    # clustering = DBSCAN(eps=0.56, min_samples=samples_dist).fit(X_stad)
    l = clustering.labels_

    n_clusters = len(set(l)) - (1 if -1 in l else 0)
    
    print('n_clusters =',n_clusters)
    n_noise = list(l).count(-1)
    # %
    u_labels = set(l)
    # Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
    colors = [plt.cm.rainbow(i) for i in np.linspace(0, 1, len(set(l)))]
    # %

    # %
    for k in range(len(colors)):  # give noise color black with opacity 0.1
        if list(u_labels)[k] == -1:
            colors[k] = [0, 0, 0, 0.1]
    # %
    colores_index = []

    for c in u_labels:
        cl_color = np.where(l == c)
        colores_index.append(cl_color)

    gr_alpha, gr_color, gr_size = 0.2, 'r', 50  # TODO
    bg_alpha = 0.05
    cl_size = 300
    ms_size, ms_color = 100, '#1f77b4'
    if len(set(l)) > 1:
        for i in range(len(set(l))-1):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            ax.hist(d_KNN, bins='auto', histtype='step', color='k')
            ax.hist(d_KNN_sim, bins='auto', histtype='step', color='r')
            ax.set_xlabel('%s-NN distance' % (samples_dist))

            texto = '\n'.join(('min real d_KNN = %s' % (round(min(d_KNN), 3)),
                               'limit set for sim d_KNN =%s' % (
                                   round(valor, 3)),
                               'average = %s' % (eps_av), '%s' % (sim_lim), '%s' % (gen_sim), clustered_by))

            props = dict(boxstyle='round', facecolor='w', alpha=0.5)
            # place a text box in upper left in axes coords
            ax.text(0.25, 0.50, texto, transform=ax.transAxes, fontsize=20,
                    verticalalignment='top', bbox=props)

            ax.set_ylabel('N')
            fig, ax = plt.subplots(1, 3, figsize=(30, 10))
            
            # color_de_cluster = '#ff7f0e'
            color_de_cluster = 'lime'  # TODO
            # fig, ax = plt.subplots(1,3,figsize=(30,10))
            # ax[1].invert_xaxis()

            # ax[0].set_title('Min %s-NN= %s. cluster by: %s '%(samples_dist,round(min(d_KNN),3),clustered_by))
            # t_gal['l'] = t_gal['l'].wrap_at('180d')
            ax[0].scatter(X[:, 0][colores_index[-1]], X[:, 1][colores_index[-1]],
                          color=colors[-1], s=50, zorder=1, alpha=bg_alpha)

            ax[0].scatter(X[:, 0], X[:, 1], color=colors[-1], s=50, zorder=1)
            # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])

            ax[0].scatter(X[:, 0][colores_index[i]], X[:, 1][colores_index[i]],
                          color=color_de_cluster, s=cl_size, zorder=2, edgecolor='k')
            
            ax[0].set_xlim(-15, 10)
            ax[0].set_ylim(-20, 10)
            ax[0].set_xlabel('$\\mu_{ra*}$ (mas yr$^{-1})$', fontsize=30)
            ax[0].set_ylabel('$\\mu_{dec}$ (mas yr$^{-1})$', fontsize=30)
            ax[0].invert_xaxis()
            # ax[0].hlines(0,-10,10,linestyle = 'dashed', color ='red')

            mul_sig, mub_sig = np.std(X[:, 0][colores_index[i]]), np.std(
                X[:, 1][colores_index[i]])
            mul_mean, mub_mean = np.mean(X[:, 0][colores_index[i]]), np.mean(
                X[:, 1][colores_index[i]])

            mul_sig_all, mub_sig_all = np.std(X[:, 0]), np.std(X[:, 1])
            mul_mean_all, mub_mean_all = np.mean(X[:, 0]), np.mean(X[:, 1])

            # Print pm in galactic

            vel_txt = '\n'.join(('$\overline{\mu}_{ra*}$ = %.2f, $\overline{\mu}_{dec}$  =  %.2f' % (mul_mean, mub_mean),
                                 '$\sigma_{\mu ra*}$ =  %.2f, $\sigma_{\mu dec}$ =  %.2f' % (mul_sig, mub_sig)))
            vel_txt_all = '\n'.join(('mura =  %.2f, $\mu_{dec}$ =  %.2f' % (round(mul_mean_all, 2), round(mub_mean_all, 2)),
                                     '$\sigma_{\mu ra*}$ =  %.2f, $\sigma_{\mu dec}$ =  %.2f' % (round(mul_sig_all, 2), round(mub_sig_all, 2))))

            propiedades = dict(
                boxstyle='round', facecolor=color_de_cluster, alpha=0.2)
            propiedades_all = dict(
                boxstyle='round', facecolor=colors[-1], alpha=0.1)
            propiedades_gr = dict(
                boxstyle='round', facecolor=gr_color, alpha=0.2)
            ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
                       verticalalignment='top', bbox=propiedades)
            # ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
            #     verticalalignment='top', bbox=propiedades_all)

            # This calcualte the maximun distance between cluster members to have a stimation of the cluster radio
            c2 = SkyCoord(ra=Ra[colores_index[i]]*u.deg,
                          dec=Dec[colores_index[i]]*u.deg)
            sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
            rad = max(sep)/2

            radio_MS = max(sep)

            # Added this bit to calculate the half-light radio (eff radio)
            # arches_dbs = arches[colores_index[i]]
            # all_mag_clus = arches_dbs['F153M']
            clus_cent = SkyCoord(ra=[np.mean(Ra[colores_index[i]])], dec=[np.mean(Dec[colores_index[i]])],
                                 unit='degree')
            clus_coord = SkyCoord(
                ra=Ra[colores_index[i]], dec=Dec[colores_index[i]], unit='degree')

            clus_mag = color_B[colores_index[i]]

            # species.SpeciesInit()
            # synphot = species.SyntheticPhotometry('Paranal/HAWKI.Ks')
            # flux = np.array([synphot.magnitude_to_flux(clus_mag[mag], error=0.2, zp_flux=None)[0] for mag in range(len(clus_mag))])
            # light = sum(flux)

            # cent_sep = clus_cent.separation(clus_coord)
            # clus_sep =np.c_[Ra[colores_index[i]],Dec[colores_index[i]],color_B[colores_index[i]],flux,cent_sep.value]

            # clus_sep = clus_sep[clus_sep[:, -1].argsort()]
            # cum = np.cumsum(clus_sep[:,3])

            # hl_ind = np.where(cum < light/2)
            # hl_ind = np.where(cum < light)

            # eff_rad = clus_sep[hl_ind[0][-1]][-1]*3600

            m_point = SkyCoord(ra=[np.mean(c2.ra)], dec=[np.mean(c2.dec)])

            idxc, group_md, d2d, d3d = ap_coor.search_around_sky(
                m_point, coordenadas, rad*1.5)

            mul_sig_gr, mub_sig_gr = np.std(
                X[:, 0][group_md]), np.std(X[:, 1][group_md])
            mul_mean_gr, mub_mean_gr = np.mean(
                X[:, 0][group_md]), np.mean(X[:, 1][group_md])

            vel_txt_gr = '\n'.join(('$\overline{\mu}_{ra*}$ = %.2f, $\overline{\mu}_{dec}$ = %.2f' % (mul_mean_gr, mub_mean_gr),
                                    '$\sigma_{\mu ra*}$ = %.2f, $\sigma_{\mu dec}$ = %.2f' % (mul_sig_gr, mub_sig_gr)))

            # ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
            #     verticalalignment='top', bbox=propiedades_gr)

            ax[0].text(0.05, 0.15, vel_txt_gr, transform=ax[0].transAxes, fontsize=20,
                       verticalalignment='top', bbox=propiedades_gr)

            # For plotting the whole catalog
            # results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
            # catal=np.loadtxt(results + 'sec_B_match_GNS_and_WFC3IR_refined_galactic.txt')
            # ax[1].scatter(catal[:,0], catal[:,1], color = 'k', alpha = 0.01)

           
            # ax[1].set_title('ID %.0f'%(st_name), fontsize = 30 )
            ax[1].scatter(Ra[group_md], Dec[group_md], color=gr_color,
                          s=gr_size, zorder=1, marker='x', alpha=gr_alpha)
            ax[0].scatter(X[:, 0][group_md], X[:, 1][group_md], color=gr_color,
                          s=gr_size, zorder=1, marker='x', alpha=gr_alpha)
            prop = dict(boxstyle='round',
                        facecolor=color_de_cluster, alpha=0.2)
            # ax[1].text(0.05, 0.15, 'hl radius $\sim$ %.2f"\n stars = %s '%(eff_rad,len(colores_index[i][0])),len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
            #                         verticalalignment='top', bbox=prop)
            # ax[1].text(0.05, 0.15, 'Radius $\sim$ %.0f"\n # stars = %s '%(rad.to(u.arcsec).value,len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
            #                         verticalalignment='top', bbox=prop)
            ax[1].text(0.57, 0.95, 'Radius $\sim$ %.0f"\n # stars = %s ' % (rad.to(u.arcsec).value, len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
                       verticalalignment='top', bbox=prop)

            

            # ax[1].scatter(catal[:,7], catal[:,8], color='k',s=50,zorder=1,alpha=0.01)#plots in galactic
            ax[1].scatter(Ra, Dec, color=colors[-1], s=50,
                          zorder=1, alpha=bg_alpha)  # plots in galactic
            ax[1].invert_xaxis()

            # print('youmamma')

            ax[1].scatter(Ra[colores_index[i]], Dec[colores_index[i]], color=color_de_cluster,
                          s=cl_size, zorder=2, edgecolor='k')  # plots in galactic
            # ax[1].quiver(Ra[colores_index[i]], Dec[colores_index[i]], X[:,0][colores_index[i]], X[:,1][colores_index[i]]*-1, alpha=1, color='black',angles = 'uv',scale = 20 )#colors[i]

            # ax[1].quiver(Ra[colores_index[i]], Dec[colores_index[i]], X[:,0][colores_index[i]], X[:,1][colores_index[i]], alpha=1, color='black',angles = 'uv',scale = 20 )#colors[i]
            # ax[1].scatter(datos[:,7][group_md],datos[:,8][group_md],s=50,color='r',alpha =0.1,marker ='x')

            ax[1].set_xlabel('Ra (°)', fontsize=30)
            ax[1].set_ylabel('Dec (°)', fontsize=30)
            # ax[1].yaxis.set_label_coords(-.05, .58)
            ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            x_ticks = np.round(ax[1].get_xticks(), 2)
            ax[1].set_xticks(np.unique(x_ticks))

            y_ticks = np.round(ax[1].get_yticks(), 2)
            ax[1].set_yticklabels(y_ticks, rotation=90)
            ax[1].set_yticks(y_ticks)
            
            plt.show()
            guarda = input('Do ypu want to save this cluster? (y/n)')
            if guarda == 'y':
                clus_np = np.c_[Ra[colores_index[i]], Dec[colores_index[i]],
                                X[:, 2][colores_index[i]], X[:, 3][colores_index[i]],
                               X[:, 0][colores_index[i]], X[:, 1][colores_index[i]],
                               color_A[colores_index[i]]]
                               
                clu_t = Table(clus_np, names = ('RA','Dec','x','y','pm_x(mas/yr)','pm_y(mas(yr)','IB224'))
                pruebas = '/Users/amartinez/Desktop/for_people/for_Laly/pruebas/'
                clu_t.write(pruebas + f'clus_{i}_knn{samples_dist}.txt', format='ascii', overwrite=True)
# sys.exit(161)
# =============================================================================
# def dbscan_GC(pmra, pmdec, x, y, Ra, Dec, color_A, color_B, color_C, clustered_by,
#               samples_dist, Ms_match, gen_sim, sim_lim, ref_f, isochrone, densidad, save_im):
# 
#     pruebas = pruebas = '/Users/amartinez/Desktop/for_people/for_Laly/pruebas/'
# 
#     import alphashape
# 
#     coordenadas = SkyCoord(ra=Ra, dec=Dec, unit='degree')
#     colorines = color_A-color_B
#     pmra_kernel, pmdec_kernel = gaussian_kde(pmra), gaussian_kde(pmdec)
#     x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
#     color_kernel = gaussian_kde(colorines)
#     if clustered_by == 'all_color':
#         X = np.array([pmra, pmdec, x, y, colorines]).T
#         X = np.array([pmra, pmdec, x, y, colorines]).T
#         X_stad = StandardScaler().fit_transform(X)
#         tree = KDTree(X_stad, leaf_size=2)
#         # DistNnce to the 1,2,3...k neighbour
#         dist, ind = tree.query(X_stad, k=samples_dist)
#         d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour
#     elif clustered_by == 'all':
#         X = np.array([pmra, pmdec, x, y]).T
#         X_stad = StandardScaler().fit_transform(X)
#         tree = KDTree(X_stad, leaf_size=2)
#         # DistNnce to the 1,2,3...k neighbour
#         dist, ind = tree.query(X_stad, k=samples_dist)
#         d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour
#     elif clustered_by == 'vel_col':
#         X = np.array([pmra, pmdec, colorines]).T
#         X_stad = StandardScaler().fit_transform(X)
#         tree = KDTree(X_stad, leaf_size=2)
#         # DistNnce to the 1,2,3...k neighbour
#         dist, ind = tree.query(X_stad, k=samples_dist)
#         d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour
# 
#     elif clustered_by == 'vel':
#         X = np.array([pmra, pmdec]).T
#         X_stad = StandardScaler().fit_transform(X)
#         tree = KDTree(X_stad, leaf_size=2)
#         # DistNnce to the 1,2,3...k neighbour
#         dist, ind = tree.query(X_stad, k=samples_dist)
#         d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour
# 
#     lst_d_KNN_sim = []
#     if gen_sim == 'kernnel':
#         for d in range(20):
#             mudec_sim,  mura_sim = pmdec_kernel.resample(
#                 len(pmdec)), pmra_kernel.resample(len(pmra))
#             x_sim, y_sim = x_kernel.resample(len(x)), y_kernel.resample(len(y))
#             color_sim = color_kernel.resample(len(pmdec))
#             if clustered_by == 'all_color':
#                 X_sim = np.array([mura_sim[0], mudec_sim[0],
#                                  x_sim[0], y_sim[0], color_sim[0]]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
#             elif clustered_by == 'all':
#                 X_sim = np.array(
#                     [mura_sim[0], mudec_sim[0], x_sim[0], y_sim[0]]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
#             elif clustered_by == 'vel_col':
#                 X_sim = np.array([mura_sim[0], mudec_sim[0], color_sim[0]]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
# 
#             elif clustered_by == 'vel':
#                 X_sim = np.array([mura_sim[0], mudec_sim[0]]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
#     if gen_sim == 'shuffle':
#         for d in range(5):
#             randomize = np.arange(len(pmdec))
#             np.random.shuffle(randomize)
#             mudec_sim,  mura_sim = pmdec[randomize], pmra[randomize]
#             x_sim, y_sim = x, y
# 
#             random_col = np.arange(len(pmdec))
#             np.random.shuffle(random_col)
# 
#             if clustered_by == 'all_color':
#                 X_sim = np.array(
#                     [mura_sim, mudec_sim, x_sim, y_sim, color_sim]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
#             elif clustered_by == 'all':
#                 X_sim = np.array([mura_sim, mudec_sim, x_sim, y_sim]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
#             elif clustered_by == 'vel_col':
#                 X_sim = np.array([mura_sim, mudec_sim, color_sim]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
# 
#             elif clustered_by == 'vel':
#                 X_sim = np.array([mura_sim, mudec_sim]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
#     d_KNN_sim_av = np.mean(lst_d_KNN_sim)
#     if sim_lim == 'mean':
#         eps_av = round((min(d_KNN)+d_KNN_sim_av)/2, 3)
#         valor = d_KNN_sim_av
#     elif sim_lim == 'minimun':
#         eps_av = round((min(d_KNN)+min(lst_d_KNN_sim))/2, 6)
#         valor = min(lst_d_KNN_sim)
#     elif sim_lim == 'maximun':
#         eps_av = round((min(d_KNN)+max(lst_d_KNN_sim))/2, 3)
#         valor = min(lst_d_KNN_sim)
# 
#     clus_method = 'dbs'
# 
#     clustering = DBSCAN(eps=eps_av, min_samples=samples_dist).fit(X_stad)
#     # clustering = DBSCAN(eps=0.56, min_samples=samples_dist).fit(X_stad)
#     l = clustering.labels_
# 
#     n_clusters = len(set(l)) - (1 if -1 in l else 0)
#     # print('Group %s.Number of cluster, eps=%s and min_sambles=%s: %s'%(group,round(epsilon,2),samples,n_clusters))
#     n_noise = list(l).count(-1)
#     # %
#     u_labels = set(l)
#     # Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
#     colors = [plt.cm.rainbow(i) for i in np.linspace(0, 1, len(set(l)))]
#     # %
# 
#     # %
#     for k in range(len(colors)):  # give noise color black with opacity 0.1
#         if list(u_labels)[k] == -1:
#             colors[k] = [0, 0, 0, 0.1]
#     # %
#     colores_index = []
# 
#     for c in u_labels:
#         cl_color = np.where(l == c)
#         colores_index.append(cl_color)
# 
#     gr_alpha, gr_color, gr_size = 0.2, 'r', 50  # TODO
#     bg_alpha = 0.05
#     cl_size = 300
#     ms_size, ms_color = 100, '#1f77b4'
#     if len(set(l)) > 1:
#         for i in range(len(set(l))-1):
# 
#             ms_uno = np.zeros(len(Ra[colores_index[i]]))
#             ms_uno[ms_in_clus] = 1
#             print('This is index in cluster:', ms_in_clus)
# 
#             fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# 
#             ax.hist(d_KNN, bins='auto', histtype='step', color='k')
#             ax.hist(d_KNN_sim, bins='auto', histtype='step', color='r')
#             ax.set_xlabel('%s-NN distance' % (samples_dist))
# 
#             texto = '\n'.join(('min real d_KNN = %s' % (round(min(d_KNN), 3)),
#                                'limit set for sim d_KNN =%s' % (
#                                    round(valor, 3)),
#                                'average = %s' % (eps_av), '%s' % (sim_lim), '%s' % (gen_sim), clustered_by))
# 
#             props = dict(boxstyle='round', facecolor='w', alpha=0.5)
#             # place a text box in upper left in axes coords
#             ax.text(0.25, 0.50, texto, transform=ax.transAxes, fontsize=20,
#                     verticalalignment='top', bbox=props)
# 
#             ax.set_ylabel('N')
#             fig, ax = plt.subplots(1, 3, figsize=(30, 10))
#             if Ms_match[16] == 14996:
# 
#                 ms_dos = np.where((Ra[colores_index[i]] == 266.38873000) & (
#                     Dec[colores_index[i]] == -28.94578110))
#                 ax[1].set_title('ID %.0f & %s' % (14996, 954199), fontsize=30)
#                 ax[0].scatter(X[:, 0][colores_index[i]][ms_dos], X[:, 1][colores_index[i]]
#                               [ms_dos], color=ms_color, s=ms_size, zorder=3, marker='^')
#                 ax[1].scatter(Ra[colores_index[i]][ms_dos], Dec[colores_index[i]][ms_dos],
#                               s=ms_size, c=ms_color, zorder=3, marker='^')  # plots in galactic
#                 ax[2].scatter(color_A[colores_index[i]][ms_dos]-color_B[colores_index[i]][ms_dos],
#                               color_B[colores_index[i]][ms_dos], s=ms_size, c=ms_color, zorder=3, marker='^')
# 
#                 # sys.exit()
#                 if len(ms_dos) < 1:
#                     sys.exit('ID 954199 is not there')
# 
#             # color_de_cluster = '#ff7f0e'
#             color_de_cluster = 'lime'  # TODO
#             # fig, ax = plt.subplots(1,3,figsize=(30,10))
#             # ax[1].invert_xaxis()
# 
#             # ax[0].set_title('Min %s-NN= %s. cluster by: %s '%(samples_dist,round(min(d_KNN),3),clustered_by))
#             # t_gal['l'] = t_gal['l'].wrap_at('180d')
#             ax[0].scatter(X[:, 0][colores_index[-1]], X[:, 1][colores_index[-1]],
#                           color=colors[-1], s=50, zorder=1, alpha=bg_alpha)
# 
#             ax[0].scatter(X[:, 0], X[:, 1], color=colors[-1], s=50, zorder=1)
#             # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
# 
#             ax[0].scatter(X[:, 0][colores_index[i]], X[:, 1][colores_index[i]],
#                           color=color_de_cluster, s=cl_size, zorder=2, edgecolor='k')
#             ax[0].scatter(Ms_match[inds[0]], Ms_match[inds[1]],
#                           s=ms_size, c=ms_color, zorder=3)
# 
#             ax[0].set_xlim(-15, 10)
#             ax[0].set_ylim(-20, 10)
#             ax[0].set_xlabel('$\\mu_{ra*}$ (mas yr$^{-1})$', fontsize=30)
#             ax[0].set_ylabel('$\\mu_{dec}$ (mas yr$^{-1})$', fontsize=30)
#             ax[0].invert_xaxis()
#             # ax[0].hlines(0,-10,10,linestyle = 'dashed', color ='red')
# 
#             mul_sig, mub_sig = np.std(X[:, 0][colores_index[i]]), np.std(
#                 X[:, 1][colores_index[i]])
#             mul_mean, mub_mean = np.mean(X[:, 0][colores_index[i]]), np.mean(
#                 X[:, 1][colores_index[i]])
# 
#             mul_sig_all, mub_sig_all = np.std(X[:, 0]), np.std(X[:, 1])
#             mul_mean_all, mub_mean_all = np.mean(X[:, 0]), np.mean(X[:, 1])
# 
#             # Print pm in galactic
# 
#             vel_txt = '\n'.join(('$\overline{\mu}_{ra*}$ = %.2f, $\overline{\mu}_{dec}$  =  %.2f' % (mul_mean, mub_mean),
#                                  '$\sigma_{\mu ra*}$ =  %.2f, $\sigma_{\mu dec}$ =  %.2f' % (mul_sig, mub_sig)))
#             vel_txt_all = '\n'.join(('mura =  %.2f, $\mu_{dec}$ =  %.2f' % (round(mul_mean_all, 2), round(mub_mean_all, 2)),
#                                      '$\sigma_{\mu ra*}$ =  %.2f, $\sigma_{\mu dec}$ =  %.2f' % (round(mul_sig_all, 2), round(mub_sig_all, 2))))
# 
#             propiedades = dict(
#                 boxstyle='round', facecolor=color_de_cluster, alpha=0.2)
#             propiedades_all = dict(
#                 boxstyle='round', facecolor=colors[-1], alpha=0.1)
#             propiedades_gr = dict(
#                 boxstyle='round', facecolor=gr_color, alpha=0.2)
#             ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
#                        verticalalignment='top', bbox=propiedades)
#             # ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
#             #     verticalalignment='top', bbox=propiedades_all)
# 
#             # This calcualte the maximun distance between cluster members to have a stimation of the cluster radio
#             c2 = SkyCoord(ra=Ra[colores_index[i]]*u.deg,
#                           dec=Dec[colores_index[i]]*u.deg)
#             sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
#             rad = max(sep)/2
# 
#             radio_MS = max(sep)
# 
#             # Added this bit to calculate the half-light radio (eff radio)
#             # arches_dbs = arches[colores_index[i]]
#             # all_mag_clus = arches_dbs['F153M']
#             clus_cent = SkyCoord(ra=[np.mean(Ra[colores_index[i]])], dec=[np.mean(Dec[colores_index[i]])],
#                                  unit='degree')
#             clus_coord = SkyCoord(
#                 ra=Ra[colores_index[i]], dec=Dec[colores_index[i]], unit='degree')
# 
#             clus_mag = color_B[colores_index[i]]
# 
#             # species.SpeciesInit()
#             # synphot = species.SyntheticPhotometry('Paranal/HAWKI.Ks')
#             # flux = np.array([synphot.magnitude_to_flux(clus_mag[mag], error=0.2, zp_flux=None)[0] for mag in range(len(clus_mag))])
#             # light = sum(flux)
# 
#             # cent_sep = clus_cent.separation(clus_coord)
#             # clus_sep =np.c_[Ra[colores_index[i]],Dec[colores_index[i]],color_B[colores_index[i]],flux,cent_sep.value]
# 
#             # clus_sep = clus_sep[clus_sep[:, -1].argsort()]
#             # cum = np.cumsum(clus_sep[:,3])
# 
#             # hl_ind = np.where(cum < light/2)
#             # hl_ind = np.where(cum < light)
# 
#             # eff_rad = clus_sep[hl_ind[0][-1]][-1]*3600
# 
#             m_point = SkyCoord(ra=[np.mean(c2.ra)], dec=[np.mean(c2.dec)])
# 
#             idxc, group_md, d2d, d3d = ap_coor.search_around_sky(
#                 m_point, coordenadas, rad*1.5)
# 
#             mul_sig_gr, mub_sig_gr = np.std(
#                 X[:, 0][group_md]), np.std(X[:, 1][group_md])
#             mul_mean_gr, mub_mean_gr = np.mean(
#                 X[:, 0][group_md]), np.mean(X[:, 1][group_md])
# 
#             vel_txt_gr = '\n'.join(('$\overline{\mu}_{ra*}$ = %.2f, $\overline{\mu}_{dec}$ = %.2f' % (mul_mean_gr, mub_mean_gr),
#                                     '$\sigma_{\mu ra*}$ = %.2f, $\sigma_{\mu dec}$ = %.2f' % (mul_sig_gr, mub_sig_gr)))
# 
#             # ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
#             #     verticalalignment='top', bbox=propiedades_gr)
# 
#             ax[0].text(0.05, 0.15, vel_txt_gr, transform=ax[0].transAxes, fontsize=20,
#                        verticalalignment='top', bbox=propiedades_gr)
# 
#             # For plotting the whole catalog
#             # results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
#             # catal=np.loadtxt(results + 'sec_B_match_GNS_and_WFC3IR_refined_galactic.txt')
#             # ax[1].scatter(catal[:,0], catal[:,1], color = 'k', alpha = 0.01)
# 
#             st_name = Ms_match[16]
#             # ax[1].set_title('ID %.0f'%(st_name), fontsize = 30 )
#             ax[1].scatter(Ra[group_md], Dec[group_md], color=gr_color,
#                           s=gr_size, zorder=1, marker='x', alpha=gr_alpha)
#             ax[0].scatter(X[:, 0][group_md], X[:, 1][group_md], color=gr_color,
#                           s=gr_size, zorder=1, marker='x', alpha=gr_alpha)
#             prop = dict(boxstyle='round',
#                         facecolor=color_de_cluster, alpha=0.2)
#             # ax[1].text(0.05, 0.15, 'hl radius $\sim$ %.2f"\n stars = %s '%(eff_rad,len(colores_index[i][0])),len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
#             #                         verticalalignment='top', bbox=prop)
#             # ax[1].text(0.05, 0.15, 'Radius $\sim$ %.0f"\n # stars = %s '%(rad.to(u.arcsec).value,len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
#             #                         verticalalignment='top', bbox=prop)
#             ax[1].text(0.57, 0.95, 'Radius $\sim$ %.0f"\n # stars = %s ' % (rad.to(u.arcsec).value, len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
#                        verticalalignment='top', bbox=prop)
# 
#             ax[1].scatter(Ms_match[0], Ms_match[1],
#                           s=ms_size, c=ms_color, zorder=3)
# 
#             # ax[1].scatter(catal[:,7], catal[:,8], color='k',s=50,zorder=1,alpha=0.01)#plots in galactic
#             ax[1].scatter(Ra, Dec, color=colors[-1], s=50,
#                           zorder=1, alpha=bg_alpha)  # plots in galactic
#             ax[1].invert_xaxis()
# 
#             # print('youmamma')
# 
#             ax[1].scatter(Ra[colores_index[i]], Dec[colores_index[i]], color=color_de_cluster,
#                           s=cl_size, zorder=2, edgecolor='k')  # plots in galactic
#             # ax[1].quiver(Ra[colores_index[i]], Dec[colores_index[i]], X[:,0][colores_index[i]], X[:,1][colores_index[i]]*-1, alpha=1, color='black',angles = 'uv',scale = 20 )#colors[i]
# 
#             # ax[1].quiver(Ra[colores_index[i]], Dec[colores_index[i]], X[:,0][colores_index[i]], X[:,1][colores_index[i]], alpha=1, color='black',angles = 'uv',scale = 20 )#colors[i]
#             # ax[1].scatter(datos[:,7][group_md],datos[:,8][group_md],s=50,color='r',alpha =0.1,marker ='x')
# 
#             ax[1].set_xlabel('Ra (°)', fontsize=30)
#             ax[1].set_ylabel('Dec (°)', fontsize=30)
#             # ax[1].yaxis.set_label_coords(-.05, .58)
#             ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#             ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#             x_ticks = np.round(ax[1].get_xticks(), 2)
#             ax[1].set_xticks(np.unique(x_ticks))
# 
#             y_ticks = np.round(ax[1].get_yticks(), 2)
#             ax[1].set_yticklabels(y_ticks, rotation=90)
#             ax[1].set_yticks(y_ticks)
#             # sys.exit(408)
# 
#             # y_ticks = np.round(ax[1].get_yticks(),2)
#             # ax[1].set_yticks(np.unique(y_ticks),rotation =30)
#             # ax[1].set_yticklabels(np.unique(y_ticks), rotation=90, va = 'center')
#             # ymin, ymax = ax[1].get_ylim()
#             # y_ticks =np.round(np.linspace(ymin, ymax, 5), 2)
#             # ax[1].set_yticklabels(np.unique(y_ticks), rotation=90)
# 
#             # ax[1].set_yticks(np.round(np.linspace(ymin, ymax, 5), 2))
# 
# 
# # =============================================================================
# #                 # This plots the contorno
# # =============================================================================
# 
#             if densidad == 'yes':
#                 p2d = np.array([Ra[colores_index[i]], Dec[colores_index[i]]]).T
#                 al_sh = alphashape.alphashape(p2d, 700)
#                 contorno = np.array(al_sh.exterior.coords[:])
#                 np.savetxt(pruebas + '%.0f_contorno.txt' % (st_name),
#                            np.array([contorno[:, 0], contorno[:, 1]]).T, fmt='%.8f')
#                 densidad = len(colores_index[i][0])/(al_sh.area*3600)
#                 ax[1].plot(contorno[:, 0], contorno[:, 1], color='k',
#                            linewidth=5, label='$\\rho$ = %.0f star/min2' % (densidad))
#                 ax[1].legend()
#             ax[1].set_xlim(min(Ra), max(Ra))
#             ax[1].set_ylim(min(Dec), max(Dec))
# 
#             # ms_in_clus = np.where((Ra[colores_index[i]] == Ms_match[0]) & (Dec[colores_index[i]] == Ms_match[1]))
#             # ms_in_clus = np.where((Ra[colores_index[i]] == Ra[colores_index[i]][0]) & (Dec[colores_index[i]] == Dec[colores_index[i]][0]))
# 
# # =============================================================================
# #                 if len(ms_in_clus[0]) > 0:
# #                     ax[1].set_facecolor('lavender')
# #                     # print(12*'*'+'\nSOMETHIG!!!!\n'+12*'*')
# #                     # ax[1].set_facecolor('mistyrose')
# # =============================================================================
# 
#             # return len(ms_in_clus[0])
# 
#             if isochrone == 'yes':
#                 import spisea
#                 from spisea import synthetic, evolution, atmospheres, reddening, ifmr
#                 from spisea.imf import imf, multiplicity
# # =============================================================================
# #
# #                     maps = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
# #                     # cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
# #
# #                     AKs = fits.open(maps + 'K2HK_C.fit',memmap=True)
# #                     AH = fits.open(maps + 'H2HK_C.fit',memmap=True)
# #
# #                     H_map = WCS(maps +  'H2HK_C.fit')
# #                     Ks_map = WCS(maps + 'K2HK_C.fit')
# #
# #                     H_map = WCS(maps +  'H1HK_C.fit')
# #                     Ks_map = WCS(maps + 'K1HK_C.fit')
# #
# #
# #                     ra_dec =[[Ra[colores_index[i]][j],Dec[colores_index[i]][j]] for j in range(len(Ra[colores_index[i]]))]
# #
# #                     pix_Ks =  Ks_map.wcs_world2pix(ra_dec,1)
# #                     AKs_clus =[AKs[0].data[pix_Ks[i][1].astype(int)][pix_Ks[i][0].astype(int)] for i in range(len(pix_Ks))]
# #                     AKs_iso = np.mean(AKs_clus)
# #                     std_AKs = np.std(AKs_clus)
# #                     # AKs_iso = 1.77
# #                     print(AKs_clus)
# # =============================================================================
# 
#                 gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
#                 # Aks_gns = pd.read_fwf(gns_ext + 'central.txt', sep =' ',header = None)
#                 # gns_HJK = ascii.read(gns_ext + 'central_ascii.txt')
#                 gns_jhk = ascii.read(gns_ext + 'central_lib_B.txt')
#                 # This generates de CMD in J - Ks
#                 center = np.where(gns_jhk['C7']-gns_jhk['C9'] > 1.3)  # TODO
#                 AKs_center = gns_jhk[center]  # TODO
#                 gns_coord = SkyCoord(ra=AKs_center['C1']*u.degree, dec=AKs_center['C3']
#                                      * u.degree, frame='icrs', equinox='J2000', obstime='J2015.5')
#                 clus_coord = SkyCoord(ra=Ra[colores_index[i]]*u.degree, dec=Dec[colores_index[i]]
#                                       * u.degree, frame='icrs', equinox='J2000', obstime='J2014.2')
# 
#                 ax[2].scatter(Ms_match[2] - Ms_match[4],
#                               Ms_match[4], s=ms_size, c=ms_color, zorder=3)
# 
#                 ax[2].scatter(color_C-color_B, color_B, alpha=0.05, color='k')
#                 ax[2].scatter(color_C[group_md]-color_B[group_md], color_B[group_md],
#                               s=gr_size, alpha=gr_alpha, c=gr_color, marker='x')
#                 txt_around = '\n'.join(('$\overline{J-Ks}$ =%.2f' % (np.median(color_C[group_md]-color_B[group_md])),
#                                         '$\sigma_{(J-Ks)}$ = %.2f' % (
#                                             np.std(color_C[group_md]-color_B[group_md])),
#                                         '$\Delta$ (J-Ks) = %.2f' % (max(color_C[group_md]-color_B[group_md])-min(color_C[group_md]-color_B[group_md]))))
#                 props_arou = dict(boxstyle='round',
#                                   facecolor=gr_color, alpha=0.2)
#                 ax[2].text(0.65, 0.25, txt_around, transform=ax[2].transAxes, fontsize=20,
#                            verticalalignment='top', bbox=props_arou)
#                 valid_color = np.where(
#                     np.isnan(color_C[colores_index[i]]) == False)
#                 color_Cg = color_C[colores_index[i]][valid_color]
#                 color_Bg = color_B[colores_index[i]][valid_color]
#                 txt_color = '\n'.join(('$\overline{J-Ks}$ =%.2f' % (np.median(color_Cg - color_Bg)),
#                                        '$\sigma_{(J-Ks)}$ = %.2f' % (
#                                            np.std(color_Cg - color_Bg)),
#                                        '$\Delta(J-Ks)$ = %.2f' % (max(color_Cg - color_Bg)-min(color_Cg - color_Bg))))
#                 # txt_color = '\n'.join(('$\overline{J-Ks}$ =%.2f'%(np.median(color_C[colores_index[i]]-color_B[colores_index[i]])),
#                 #                      '$\sigma_{(J-Ks)}$ = %.2f'%(np.std(color_C[colores_index[i]]-color_B[colores_index[i]])),
#                 #                      '$\Delta(J-Ks)$ = %.2f'%(max(color_C[colores_index[i]]-color_B[colores_index[i]])-min(color_C[colores_index[i]]-color_B[colores_index[i]]))))
#                 props = dict(boxstyle='round',
#                              facecolor=color_de_cluster, alpha=0.2)
#                 # # place a text box in upper left in axes coords
#                 ax[2].text(0.50, 0.95, txt_color, transform=ax[2].transAxes, fontsize=30,
#                            verticalalignment='top', bbox=props)
# 
#                 ax[2].scatter(color_C[colores_index[i]]-color_B[colores_index[i]], color_B[colores_index[i]],
#                               alpha=1, c=color_de_cluster, s=cl_size, zorder=2, edgecolor='k')
#                 ax[2].scatter(color_C[colores_index[i]][ms_dos]-color_B[colores_index[i]][ms_dos],
#                               color_B[colores_index[i]][ms_dos], s=ms_size*100, c=ms_color, zorder=3)
# 
#                 ax[2].set_xlabel('J$-$Ks', fontsize=30)
#                 ax[2].set_ylabel('Ks', fontsize=30)
# 
#                 # This generates the isochrone at certain extiction
# 
#                 AKs_clus_all = []
#                 idx = clus_coord.match_to_catalog_sky(gns_coord)
#                 validas = np.where(idx[1] < 0.5*u.arcsec)
#                 gns_match = AKs_center[idx[0][validas]]
# 
#                 for member in range(len(gns_match)):
#                     if gns_match[member]['C17'] != '---' and gns_match[member]['C19'] != '---' and gns_match[member]['C19'] != '-1.00000':
#                         AKs_clus_all.append(float(gns_match[member]['C19']))
# 
#                 AKs_list1 = np.arange(1.6, 2.11, 0.01)
#                 AKs_list = np.append(AKs_list1, 0)
# 
#                 AKs_clus, std_AKs = np.median(
#                     AKs_clus_all), np.std(AKs_clus_all)
# 
#                 def absolute_difference_function(
#                     list_value): return abs(list_value - AKs_clus)
#                 AKs_iso = min(AKs_list, key=absolute_difference_function)
# 
#                 print(AKs_iso)
#                 print(np.std(AKs_clus_all))
# 
#                 dist = 8200  # distance in parsec
#                 metallicity = 0.30  # Metallicity in [M/H]
#                 # # logAge_600 = np.log10(0.61*10**9.)
#                 # logAge = np.log10(5*10**6.)
#                 # logAge_30 = np.log10(0.030*10**9.)
#                 logAge = np.log10(8e9)
#                 # logAge_90 = np.log10(0.090*10**9.)
#                 evo_model = evolution.MISTv1()
#                 atm_func = atmospheres.get_merged_atmosphere
#                 red_law = reddening.RedLawNoguerasLara18()
#                 filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']
# 
#                 imf_multi = multiplicity.MultiplicityUnresolved()
# 
#                 iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'
#                 iso = synthetic.IsochronePhot(logAge, AKs_iso, dist, metallicity=metallicity,
#                                               evo_model=evo_model, atm_func=atm_func,
#                                               red_law=red_law, filters=filt_list,
#                                               iso_dir=iso_dir)
#                 ax[2].plot(iso.points['m_hawki_J'] - iso.points['m_hawki_Ks'],
#                            iso.points['m_hawki_Ks'], 'b-',  label='%.2fMyr\nAKs = %.2f $\pm$ %.2f ' % (10**logAge/1e6, AKs_iso, std_AKs))
#                 ax[2].legend(loc=5)
#                 ax[2].set_ylim(min(color_B), max(color_B))
#                 ax[2].invert_yaxis()
#             else:
#                 ax[2].scatter(Ms_match[3] - Ms_match[4],
#                               Ms_match[4], s=ms_size, c=ms_color, zorder=3)
# 
#                 print('yomamma')
#                 ax[2].scatter(color_A-color_B, color_B, alpha=0.05, color='k')
#                 ax[2].scatter(color_A[group_md]-color_B[group_md], color_B[group_md],
#                               s=gr_size, alpha=gr_alpha, c=gr_color, marker='x')
#                 txt_around = '\n'.join(('$\overline{H-Ks}$ =%.2f' % (np.median(color_A[group_md]-color_B[group_md])),
#                                         '$\sigma_{(H-Ks)}$ = %.2f' % (
#                                             np.std(color_A[group_md]-color_B[group_md])),
#                                         '$\Delta$ (H-Ks) = %.2f' % (max(color_A[group_md]-color_B[group_md])-min(color_A[group_md]-color_B[group_md]))))
#                 props_arou = dict(boxstyle='round',
#                                   facecolor=gr_color, alpha=0.2)
#                 ax[2].text(0.65, 0.19, txt_around, transform=ax[2].transAxes, fontsize=20,
#                            verticalalignment='top', bbox=props_arou)
# 
#                 ax[2].scatter(color_A[colores_index[i]]-color_B[colores_index[i]], color_B[colores_index[i]],
#                               alpha=1, c=color_de_cluster, s=cl_size, zorder=2, edgecolor='k')
# 
#                 # txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),
#                 #                      'age = %.0f Myr'%(10**logAge/10**6)))
#                 txt_color = '\n'.join(('$\overline{H-Ks}$ =%.2f' % (np.median(color_A[colores_index[i]]-color_B[colores_index[i]])),
#                                        '$\sigma_{(H-Ks)}$ = %.2f' % (
#                                            np.std(color_A[colores_index[i]]-color_B[colores_index[i]])),
#                                        '$\Delta(H-Ks)$ = %.2f' % (max(color_A[colores_index[i]]-color_B[colores_index[i]])-min(color_A[colores_index[i]]-color_B[colores_index[i]]))))
#                 props = dict(boxstyle='round',
#                              facecolor=color_de_cluster, alpha=0.2)
#                 # # place a text box in upper left in axes coords
#                 ax[2].text(0.50, 0.95, txt_color, transform=ax[2].transAxes, fontsize=30,
#                            verticalalignment='top', bbox=props)
#                 # ax[2].text(0.65, 0.85, txt_srn, transform=ax[2].transAxes, fontsize=14,
#                 #     verticalalignment='top', bbox=props)
#                 ax[2].set_xlabel('H$-$Ks', fontsize=30)
#                 ax[2].set_ylabel('Ks', fontsize=30)
#                 # ax[2].set_xlim(1.1,4)
#                 ax[2].set_ylim(min(color_B), max(color_B))
#                 ax[2].invert_yaxis()
# 
#                 if save_im == 'yes':
#                     # from MS_analysis import radio
#                     radio = 65*u.arcsec
#                     for_saving = '/Users/amartinez/Desktop/PhD/My_papers/Libralato/'
#                     plt.savefig(for_saving + 'ms_id_%.0f_knn%s_rad%s_%s.png' % (
#                         Ms_match[16], samples_dist, radio.value, sim_lim), dpi=300, bbox_inches='tight')
#                 plt.show()
#                 return len(ms_in_clus[0]), Ra[colores_index[i]], Dec[colores_index[i]], X[:, 0][colores_index[i]], X[:, 1][colores_index[i]], color_A[colores_index[i]], color_B[colores_index[i]], color_C[colores_index[i]], ms_uno
#             else:
#                 return [0, 0, 0, 0, 0]
#     else:
#         return [0, 0, 0, 0, 0]
# # %%
# # This one searchs all around the catalog.
# 
# 
# def dbscan_GC_all(pmra, pmdec, x, y, Ra, Dec, color_A, color_B, clustered_by, samples_dist, ms_match, gen_sim, sim_lim, ref_f):
#     mix_color = 'yes'
#     inds = [17, 18] if ref_f == 'gal' else [9, 11]
# 
#     ms_coord = SkyCoord(
#         ra=ms_match[:, 0], dec=ms_match[:, 1], unit='degree', frame='icrs')
# 
#     coordenadas = SkyCoord(ra=Ra, dec=Dec, unit='degree')
#     colorines = color_A-color_B
#     pmra_kernel, pmdec_kernel = gaussian_kde(pmra), gaussian_kde(pmdec)
#     x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
#     color_kernel = gaussian_kde(colorines)
#     if clustered_by == 'all_color':
#         X = np.array([pmra, pmdec, x, y, colorines]).T
#         X = np.array([pmra, pmdec, x, y, colorines]).T
#         X_stad = StandardScaler().fit_transform(X)
#         tree = KDTree(X_stad, leaf_size=2)
#         # DistNnce to the 1,2,3...k neighbour
#         dist, ind = tree.query(X_stad, k=samples_dist)
#         d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour
#     elif clustered_by == 'all':
#         X = np.array([pmra, pmdec, x, y]).T
#         X_stad = StandardScaler().fit_transform(X)
#         tree = KDTree(X_stad, leaf_size=2)
#         # DistNnce to the 1,2,3...k neighbour
#         dist, ind = tree.query(X_stad, k=samples_dist)
#         d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour
#     elif clustered_by == 'vel_col':
#         X = np.array([pmra, pmdec, colorines]).T
#         X_stad = StandardScaler().fit_transform(X)
#         tree = KDTree(X_stad, leaf_size=2)
#         # DistNnce to the 1,2,3...k neighbour
#         dist, ind = tree.query(X_stad, k=samples_dist)
#         d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour
# 
#     elif clustered_by == 'vel':
#         X = np.array([pmra, pmdec]).T
#         X_stad = StandardScaler().fit_transform(X)
#         tree = KDTree(X_stad, leaf_size=2)
#         # DistNnce to the 1,2,3...k neighbour
#         dist, ind = tree.query(X_stad, k=samples_dist)
#         d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour
# 
#     lst_d_KNN_sim = []
#     if gen_sim == 'kernnel':
#         for d in range(5):
#             mudec_sim,  mura_sim = pmdec_kernel.resample(
#                 len(pmdec)), pmra_kernel.resample(len(pmra))
#             x_sim, y_sim = x_kernel.resample(len(x)), y_kernel.resample(len(y))
#             color_sim = color_kernel.resample(len(pmdec))
#             if clustered_by == 'all_color':
#                 X_sim = np.array([mura_sim[0], mudec_sim[0],
#                                  x_sim[0], y_sim[0], color_sim[0]]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
#             elif clustered_by == 'all':
#                 X_sim = np.array(
#                     [mura_sim[0], mudec_sim[0], x_sim[0], y_sim[0]]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
#             elif clustered_by == 'vel_col':
#                 X_sim = np.array([mura_sim[0], mudec_sim[0], color_sim[0]]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
# 
#             elif clustered_by == 'vel':
#                 X_sim = np.array([mura_sim[0], mudec_sim[0]]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
#     if gen_sim == 'shuffle':
#         for d in range(10):
#             randomize = np.arange(len(pmdec))
#             np.random.shuffle(randomize)
#             mudec_sim,  mura_sim = pmdec[randomize],     [randomize]
#             x_sim, y_sim = x, y
# 
#             random_col = np.arange(len(pmdec))
#             np.random.shuffle(random_col)
#             if mix_color == 'yes':
#                 H_sim, K_sim = color_A[random_col], color_B[random_col]
#             elif mix_color == 'no':
#                 H_sim, K_sim = color_A, color_B
#             color_sim = H_sim-K_sim
#             if clustered_by == 'all_color':
#                 X_sim = np.array(
#                     [mura_sim, mudec_sim, x_sim, y_sim, color_sim]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
#             elif clustered_by == 'all':
#                 X_sim = np.array([mura_sim, mudec_sim, x_sim, y_sim]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
#             elif clustered_by == 'vel_col':
#                 X_sim = np.array([mura_sim, mudec_sim, color_sim]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
# 
#             elif clustered_by == 'vel':
#                 X_sim = np.array([mura_sim, mudec_sim]).T
#                 X_stad_sim = StandardScaler().fit_transform(X_sim)
#                 tree_sim = KDTree(X_stad_sim, leaf_size=2)
# 
#                 # DistNnce to the 1,2,3...k neighbour
#                 dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
#                 # distance to the Kth neighbour
#                 d_KNN_sim = sorted(dist_sim[:, -1])
# 
#                 lst_d_KNN_sim.append(min(d_KNN_sim))
#     d_KNN_sim_av = np.mean(lst_d_KNN_sim)
#     if sim_lim == 'mean':
#         eps_av = round((min(d_KNN)+d_KNN_sim_av)/2, 3)
#         valor = d_KNN_sim_av
#     elif sim_lim == 'minimun':
#         eps_av = round((min(d_KNN)+min(lst_d_KNN_sim))/2, 3)
#         valor = min(lst_d_KNN_sim)
#     elif sim_lim == 'maximun':
#         eps_av = round((min(d_KNN)+max(lst_d_KNN_sim))/2, 3)
#         valor = min(lst_d_KNN_sim)
# 
#     clus_method = 'dbs'
# 
#     clustering = DBSCAN(eps=eps_av, min_samples=samples_dist).fit(X_stad)
#     l = clustering.labels_
# 
#     n_clusters = len(set(l)) - (1 if -1 in l else 0)
#     # print('Group %s.Number of cluster, eps=%s and min_sambles=%s: %s'%(group,round(epsilon,2),samples,n_clusters))
#     n_noise = list(l).count(-1)
#     # %
#     u_labels = set(l)
#     # Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
#     colors = [plt.cm.rainbow(i) for i in np.linspace(0, 1, len(set(l)))]
#     # %
# 
#     # %
#     for k in range(len(colors)):  # give noise color black with opacity 0.1
#         if list(u_labels)[k] == -1:
#             colors[k] = [0, 0, 0, 0.1]
#     # %
#     colores_index = []
# 
#     for c in u_labels:
#         cl_color = np.where(l == c)
#         colores_index.append(cl_color)
#     if len(set(l)) > 1:
#         for i in range(len(set(l))-1):
#             print('number of found clusters = %s' % (len(set(l))-1))
# 
#             fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#             ax.hist(d_KNN, bins='auto', histtype='step', color='k')
#             ax.hist(d_KNN_sim, bins='auto', histtype='step', color='r')
#             ax.set_xlabel('%s-NN distance' % (samples_dist))
# 
#             texto = '\n'.join(('min real d_KNN = %s' % (round(min(d_KNN), 3)),
#                                'limit set for sim d_KNN =%s' % (
#                                    round(valor, 3)),
#                                'average = %s' % (eps_av), '%s' % (sim_lim), '%s' % (gen_sim)))
# 
#             props = dict(boxstyle='round', facecolor='w', alpha=0.5)
#             # place a text box in upper left in axes coords
#             ax.text(0.65, 0.25, texto, transform=ax.transAxes, fontsize=20,
#                     verticalalignment='top', bbox=props)
# 
#             ax.set_ylabel('N')
#             fig, ax = plt.subplots(1, 3, figsize=(30, 10))
#             color_de_cluster = '#ff7f0e'
#             # fig, ax = plt.subplots(1,3,figsize=(30,10))
#             # ax[2].invert_yaxis()
# 
#             ax[0].set_title('Min %s-NN= %s. cluster by: %s ' %
#                             (samples_dist, round(min(d_KNN), 3), clustered_by))
#             # t_gal['l'] = t_gal['l'].wrap_at('180d')
#             ax[0].scatter(X[:, 0][colores_index[-1]], X[:, 1]
#                           [colores_index[-1]], color=colors[-1], s=50, zorder=1)
# 
#             ax[0].scatter(X[:, 0], X[:, 1], color=colors[-1], s=50, zorder=1)
#             # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
#             ax[0].scatter(X[:, 0][colores_index[i]], X[:, 1]
#                           [colores_index[i]], color=color_de_cluster, s=50, zorder=3)
#             # ax[0].set_xlim(-10,10)
#             # ax[0].set_ylim(-10,10)
#             ax[0].set_xlabel(
#                 r'$\mathrm{\mu_{ra} (mas\ yr^{-1})}$', fontsize=30)
#             ax[0].set_ylabel(
#                 r'$\mathrm{\mu_{dec} (mas\ yr^{-1})}$', fontsize=30)
#             ax[0].invert_xaxis()
#             # ax[0].hlines(0,-10,10,linestyle = 'dashed', color ='red')
# 
#             mul_sig, mub_sig = np.std(X[:, 0][colores_index[i]]), np.std(
#                 X[:, 1][colores_index[i]])
#             mul_mean, mub_mean = np.mean(X[:, 0][colores_index[i]]), np.mean(
#                 X[:, 1][colores_index[i]])
# 
#             mul_sig_all, mub_sig_all = np.std(X[:, 0]), np.std(X[:, 1])
#             mul_mean_all, mub_mean_all = np.mean(X[:, 0]), np.mean(X[:, 1])
# 
#             vel_txt = '\n'.join(('mura = %s, mudec = %s' % (round(mul_mean, 3), round(mub_mean, 3)),
#                                  '$\sigma_{mura}$ = %s, $\sigma_{mudec}$ = %s' % (round(mul_sig, 3), round(mub_sig, 3))))
#             vel_txt_all = '\n'.join(('mura = %s, mudec = %s' % (round(mul_mean_all, 3), round(mub_mean_all, 3)),
#                                      '$\sigma_{mura}$ = %s, $\sigma_{mudec}$ = %s' % (round(mul_sig_all, 3), round(mub_sig_all, 3))))
# 
#             propiedades = dict(
#                 boxstyle='round', facecolor=color_de_cluster, alpha=0.2)
#             propiedades_all = dict(
#                 boxstyle='round', facecolor=colors[-1], alpha=0.1)
#             ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
#                        verticalalignment='top', bbox=propiedades)
#             ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
#                        verticalalignment='top', bbox=propiedades_all)
# 
#             # This calcualte the maximun distance between cluster members to have a stimation of the cluster radio
#             c2 = SkyCoord(ra=Ra[colores_index[i]]*u.deg,
#                           dec=Dec[colores_index[i]]*u.deg)
#             sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
#             rad = max(sep)/2
# 
#             radio_MS = max(sep)
# 
#             m_point = SkyCoord(ra=[np.mean(c2.ra)], dec=[np.mean(c2.dec)])
# 
#             idxc, group_md, d2d, d3d = ap_coor.search_around_sky(
#                 m_point, coordenadas, rad*2)
# 
#             ax[1].scatter(Ra[group_md], Dec[group_md], color='red',
#                           s=50, zorder=1, marker='x', alpha=0.7)
#             ax[0].scatter(X[:, 0][group_md], X[:, 1][group_md],
#                           color='red', s=50, zorder=1, marker='x', alpha=0.7)
#             prop = dict(boxstyle='round',
#                         facecolor=color_de_cluster, alpha=0.2)
#             ax[1].text(0.15, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s ' % (round(rad.to(u.arcsec).value, 2), len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
#                        verticalalignment='top', bbox=prop)
# 
#             # ax[1].scatter(catal[:,7], catal[:,8], color='k',s=50,zorder=1,alpha=0.01)#plots in galactic
#             ax[1].scatter(Ra, Dec, color=colors[-1], s=50,
#                           zorder=1, alpha=0.01)  # plots in galactic
# 
#             ax[1].scatter(Ra[colores_index[i]], Dec[colores_index[i]],
#                           color=color_de_cluster, s=50, zorder=3)  # plots in galactic
#             # ax[1].quiver(Ra[colores_index[i]], Dec[colores_index[i]], X[:,0][colores_index[i]], X[:,1][colores_index[i]], alpha=1, color='black' )#colors[i]
#             # ax[1].scatter(datos[:,7][group_md],datos[:,8][group_md],s=50,color='r',alpha =0.1,marker ='x')
# 
#             ax[1].scatter(ms_match[:, 0], ms_match[:, 1], color='cyan', s=200)
# 
#             ax[1].invert_xaxis()
#             ax[1].set_xlabel('Ra', fontsize=30)
#             ax[1].set_ylabel('Dec', fontsize=30)
#             ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#             ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# 
# # =============================================================================
# #             for ms_star in range(len(ms_match)):
# #                 sep_ms = c2.separation(ms_coord[ms_star])
# #                 if min(sep_ms.value) <5/3600:
# #                     ax[1].set_facecolor('lavender')
# # =============================================================================
# 
#             # for ms_star in range(len(ms_match)):
#             #     clus_w_ms = np.where((Ra[colores_index[i]] == ms_match[ms_star][0]) & (Ra[colores_index[i]] == ms_match[ms_star][1]))
#             #     if len(clus_w_ms[0])>0:
#             #         ax[1].set_facecolor('lavender')
#             #         sys.exit('620')
# 
#             # ms_in_clus = np.where((Ra[colores_index[i]] == Ms_match[0]) & (Dec[colores_index[i]] == Ms_match[1]))
#             # ms_in_clus = np.where((Ra[colores_index[i]] == Ra[colores_index[i]][0]) & (Dec[colores_index[i]] == Dec[colores_index[i]][0]))
# 
#             ax[2].scatter(color_A-color_B, color_B, alpha=0.1, color='k')
#             # ax[2].scatter(color_A[group_md]-color_B[group_md], color_B[group_md],alpha=0.7,c='r',marker = 'x')
#             txt_around = '\n'.join(('H-Ks =%.3f' % (np.median(color_A[group_md]-color_B[group_md])),
#                                     '$\sigma_{H-Ks}$ = %.3f' % (
#                                         np.std(color_A[group_md]-color_B[group_md])),
#                                     'diff_color = %.3f' % (max(color_A[group_md]-color_B[group_md])-min(color_A[group_md]-color_B[group_md]))))
#             props_arou = dict(boxstyle='round', facecolor='r', alpha=0.3)
#             ax[2].text(0.50, 0.25, txt_around, transform=ax[2].transAxes, fontsize=30,
#                        verticalalignment='top', bbox=props_arou)
# 
#             ax[2].scatter(color_A[colores_index[i]]-color_B[colores_index[i]],
#                           color_B[colores_index[i]], alpha=1, c=color_de_cluster)
# 
#             # txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),
#             #                      'age = %.0f Myr'%(10**logAge/10**6)))
#             txt_color = '\n'.join(('H-Ks =%.3f' % (np.median(color_A[colores_index[i]]-color_B[colores_index[i]])),
#                                    '$\sigma_{H-Ks}$ = %.3f' % (
#                                        np.std(color_A[colores_index[i]]-color_B[colores_index[i]])),
#                                    'diff_color = %.3f' % (max(color_A[colores_index[i]]-color_B[colores_index[i]])-min(color_A[colores_index[i]]-color_B[colores_index[i]]))))
#             props = dict(boxstyle='round',
#                          facecolor=color_de_cluster, alpha=0.3)
#             # # place a text box in upper left in axes coords
#             ax[2].text(0.50, 0.95, txt_color, transform=ax[2].transAxes, fontsize=30,
#                        verticalalignment='top', bbox=props)
#             # ax[2].text(0.65, 0.85, txt_srn, transform=ax[2].transAxes, fontsize=14,
#             #     verticalalignment='top', bbox=props)
#             ax[2].set_xlabel('H - Ks', fontsize=30)
#             ax[2].set_ylabel('Ks', fontsize=30)
#             ax[2].set_xlim(1.3, 4)
#             ax[2].invert_yaxis()
#             # return len(ms_in_clus[0])
#             plt.show()
#             # return Ra[colores_index[i]], Dec[colores_index[i]], X[:,0][colores_index[i]],X[:,1][colores_index[i]]
#         # else:
#             # return [0,0,0,0,0]
# 
# 
# =============================================================================
