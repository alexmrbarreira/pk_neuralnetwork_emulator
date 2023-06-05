import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import camb
from camb import model, initialpower
import time
plt.rcParams.update({'text.usetex': True, 'mathtext.fontset': 'stix'}) 

# ===================================================================
# Set global parameters and emulator priors
# ===================================================================

# neutrino masses
mnu     = 0.0
# curvature
omk     = 0
# k_min [h/Mpc]
kmin    = 1.0e-4
# k_max [h/Mpc]
kmax    = 10.
# number k-values in spectra P(k)
nkpoints = 100
# k array to use in plots
kk = 10.**np.linspace(np.log10(kmin), np.log10(kmax), nkpoints)
# emulation dimension (number of parameters to emulate)
n_dim = 6
# number of training/validation/testing points
n_train = int(1e3)
n_valid = int(n_train*0.2)
n_test  = int(1e2)

# Cosmological parameter priors (redshift z, is treated as a parameter to emulate too)
# Parameter order [H0, Omega_b h^2, Omega_c h^2, ln(1e10 As), ns,  z]
#                 [ 0,      1     ,      2     ,       3    ,  4,  5] 
# H0
H0_min      = 50.
H0_max      = 100.
# Omega_b h^2
ombh2_min   = 0.020
ombh2_max   = 0.024
# Omega_c h^2
omch2_min   = 0.100
omch2_max   = 0.150
# ln(1e10 As)
lnAse10_min = 1.5
lnAse10_max = 4.0
# ns
ns_min      = 0.90
ns_max      = 1.10
# z
z_min       = 0.0
z_max       = 10. 


