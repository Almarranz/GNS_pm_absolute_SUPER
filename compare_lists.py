"""
compare_lists.py
Almarranz 2024 May 28
Extracts the nearest neighbors between two lists that are closer that the value of 'dis_min'
l1 and l2 should be np arrays of coordinates [x,y] 
Returns: Coord. of l1 matches, Coord. of l2 matches, l1 ind. of matches, l2 ind. of matches, distances 
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
from astropy.table import Table
def compare_lists(l1,l2, dis_min):
    nbrs1 = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(l2)
    dis1, ind1 = nbrs1.kneighbors(l1)

    nbrs2 = NearestNeighbors(n_neighbors=1, algorithm='auto',metric='euclidean').fit(l1)
    dis2, ind2 = nbrs2.kneighbors(l2)

    dis1 = np.array(dis1)

    matc1  = np.empty((0,2),float)
    matc2  = np.empty((0,2),float)
    d1 = np.empty((0,1),float)
    d2 = np.empty((0,1),float)
    l1_i = np.empty((0,1),int)
    # l1_i = []
    l2_i = np.empty((0,1),int)
    # l2_i = []
    for i, idx1 in enumerate(ind1):
        # idx1 is the index of the nearest neighbor in list2 for point i in list1
        if ind2[idx1[0]] == i and dis1[i]< dis_min:
          
            matc1 = np.append(matc1, [l1[i]], axis= 0)
            matc2 = np.append(matc2,[l2[idx1[0]]], axis= 0)
            d1 = np.append(d1, [dis1[i]], axis = 0)
            d2 = np.append(d2, [dis2[idx1[0]]], axis = 0)
            l1_i = np.append(l1_i, [[i]], axis = 0)
            l2_i = np.append(l2_i, [[idx1[0]]], axis = 0)

    comp = np.c_[matc1,matc2,l1_i.astype(int),l2_i.astype(int),d1]
    comp_t = Table(comp, names =('l1_x','l1_y','l2_x','l2_y','ind_1','ind_2','dist'))
    comp_t['ind_1'] = comp_t['ind_1'].astype(int)
    comp_t['ind_2'] = comp_t['ind_2'].astype(int)

    
    return comp_t
    # return comp
    # return matc1,matc2,l1_i.astype(int).T,l2_i.astype(int).T,d1.T