# car_website_scraping
A set of python scripts for astrophysicists/cosmologists that utilizes a neural network to emulate the power spectrum of nonlinear matter density fluctuations in the Universe. It works as a high-dimensional parameter space interpolator aimed to speed up predictions as a function of cosmological parameters.

The model takes as input the values of the parameters H0, $\Omega$

NOTE: this was not extensively optimized, and so as is, it should not be used to produce results for research papers. Rather it can work as a starting point to explore other architectures and training strategies, generalize to other cosmological quantities (polyspectra, halo mass function, etc.), or simply illustrating how to use *python* and *tensorflow* in a concrete cosmology emulation application.

### Table of contents
- [Dependencies](#dependencies)
- [Script overview](#script-overview)
- [Performance overview](#performance-overview)

### Dependencies

- CAMB code (*pip install camb* ; https://camb.readthedocs.io/en/latest/)
- numpy, scipy and matplotlib
- tensorflow

### Overview of the scripts

##### commons.py
This is a parameter file "common" to all other scripts (imported by all). Modify this to set prior parameter ranges, number of training examples and other spectra details.

##### generate_data.py
Executing this script as *python generate_data.py* will generate three latin hypercubes for training, validation and testing sets.

