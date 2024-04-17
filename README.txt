Tools to calculate observables like correlation functions and lensing cross-powers in the halo model
As many steps as possible implemented with numpy for speed

Uses Core Cosmology Library (CCL) as backend to compute matter power spectra,
this frontend transforms to observables

Likely most useful for observational astrophysics purposes rather than cosmology,
i.e. inferring host halo properties of a sample of galaxies
because we assume a fixed cosmology and transform directly from matter power to real-space correlations unlike CCL

Includes least-squares fitting routines to estimate effective bias / host halo mass / minimum host mass
from clustering or lensing measurements, also cross-correlations
Also includes MCMC fitting routines to constrain HOD parameters
All inputs and outputs have little h units, conversion is done internally to interface with CCL



Installation:

1. Make environment (anaconda) with numpy, scipy, astropy, pyccl, optionally emcee, corner
2. pip install colossus, mcfit
3. Set params in params.py

See notebooks/showcase.ipynb for usage