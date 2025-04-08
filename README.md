        
# GNS_pm_absolute_offsets using just python scripts (and no astroalign)

## APROACH A: Obtaining Absolute Proper Motions from GNS1 and GNS2 Using GAIA as the Reference Frame

### GRF (ICRS): Offsets (GNS xy coordinates and Deprojected Gaia Coordinates)

0. **`gns1_B_lists.py`**  
   Combines the H and Ks lists from GNS1.
1. **`gns_gaia_alignment.py**

- aligns GNS1 and GNS2 with Gaia. Then computes the propermotions and compares with Gaia pms
- It is an interactive non autmatic process. After the first loop the 3σ star will be printed out in the console. Copy and paste this numbers in the `bad1` and `bad1`and rerun the program


### APROACH B:Obtaing the pm by aligning GNS1 to Gaia and the GNS2 to GNS1 (in the Gaia RF)

0. **`gns1_B_lists.py`**  
   Combines the H and Ks lists from GNS1.
1. **`gns1_gaia_gns2.py**


