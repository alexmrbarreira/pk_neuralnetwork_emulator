# car_website_scraping
A set of python scripts for astrophysicists/cosmologists that utilizes a neural network to emulate the power spectrum of nonlinear matter density fluctuations in the Universe. It works as a high-dimensional parameter space interpolator aimed to speed up predictions as a function of cosmological parameters.

NOTE: this code was not extensively optimized, and so as is, it should not be used to produce results for research papers. Rather it can work as a solid starting point to explore other architectures and  training strategies, as well as to generalize it to other types of key cosmological quantities (other spectra, polyspectra, halo mass function, etc.)

### Table of contents
- [Dependencies](#dependencies)
- [Running the scripts](#running-the-scripts)
- [Emulator performance](#results-overview-for-different-cars-same-country)

### Dependencies

- numpy and matplotlib
- pandas
- bs4 - python web scraping library
- firefox browser (optional)
