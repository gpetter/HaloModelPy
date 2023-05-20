import numpy as np
import os
import astropy.units as u
from colossus.cosmology import cosmology
import astropy.cosmology.units as cu
cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()
from colossus.lss import mass_function



shen2020dir = '/home/graysonpetter/ssd/Dartmouth/common_tools/quasarlf/'


def int_lf_over_z_and_l(dndL, dndz, nu=None):
    """
    Integrate Shen 2020 quasar luminosity function over luminosity and redshift distributions.
    Luminosity is spectral luminosity at frequency nu
    If nu not given, assume bolometric luminosity
    Parameters
    ----------
    dndL: tuple, (centers of luminosity bins, normalized dN/dL). luminosities given in log10(erg/s)
    dndz: tuple (redshift bin centers, normalized dN/dz)
    nu: float, if given, the frequency (Hz) at which luminosity was observed, otherwise assume bolometric luminosity

    Returns
    -------
    A space density of quasars predicted by Shen model for given L and z distributions
    Non log, (little h / Mpc)^3 units

    """
    curdir = os.getcwd()
    # use code from Shen et al 2020
    os.chdir(shen2020dir + 'pubtools/')
    import utilities
    zs, dndz = dndz
    ls, dndL = dndL
    ints_at_zs = []
    # for each redshift in grid
    for z in zs:
        # if no frequency given, integrate bolometric LF
        if nu is None:
            lgrid, lf = utilities.return_bolometric_qlf(redshift=z)
        else:
            # get luminosity function at redshift z and in band
            lgrid, lf = utilities.return_qlf_in_band(redshift=z, nu=nu)
        # interpolate QLF at positions of observed luminosity bins
        lf_at_ls = 10 ** np.interp(ls, lgrid, lf)
        # integrate over luminosity distribution
        ints_at_zs.append(np.trapz(lf_at_ls * dndL, x=ls))
    # integrate over redshift distribution
    dens = np.trapz(np.array(ints_at_zs)*dndz, x=zs) * (u.Mpc**-3)
    # convert to little h units for comparision with HMF
    dens_hunit = dens.to((cu.littleh/u.Mpc)**3, cu.with_H0(apcosmo.H0)).value
    os.chdir(curdir)
    return dens_hunit


#
def int_hmf_z(dndz, logminmass, massgrid=np.logspace(11, 16, 5000)):
    """
    integrate HMF over redshift for average space density of halos
    Parameters
    ----------
    dndz
    logminmass: float, minimum mass to intergrate HMF above, in log(Msun/h) units
    massgrid: mass grid to perform integral over

    Returns
    -------
    Space density of halos more massive than minmass, over givne redshift distribution
    Non log, (little h / Mpc)^3 units
    """
    zs, dndz = dndz
    ints_at_zs = []
    # for each redshift in grid
    for z in zs:
        # get HMF(z)
        mfunc_so = mass_function.massFunction(massgrid, z, mdef='200c', model='tinker08', q_out='dndlnM')
        # number of halos more massive than M is integral of HMF from M to inf
        occupiedidxs = np.where(np.log10(massgrid) > logminmass)
        mfunc_so, newgrid = mfunc_so[occupiedidxs], massgrid[occupiedidxs]
        ints_at_zs.append(np.trapz(mfunc_so, x=np.log(newgrid)))

    return np.trapz(np.array(ints_at_zs)*dndz, x=zs)


def occupation_fraction(dndL, dndz, logminmasses, nu=None, logmin_errs=None):
    """
    Occupation fraction is space density of quasars over space density of halos more massive than threshold
    Parameters
    ----------
    dndL
    dndz
    logminmasses
    nu

    Returns
    -------

    """
    spc_density = int_lf_over_z_and_l(dndL=dndL, dndz=dndz, nu=nu)  # / 2. for obscured
    minmasses = np.atleast_1d(logminmasses)

    halodensities = []
    for logmass in minmasses:
        halodensities.append(int_hmf_z(dndz, logminmass=logmass))

    return spc_density / np.array(halodensities)    # occupation fractions for minmasses
