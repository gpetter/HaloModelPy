import numpy as np
import os
import astropy.units as u
from . import cosmo
from . import hubbleunits
from . import redshift_helper



shen2020dir = '/home/graysonpetter/ssd/Dartmouth/common_tools/quasarlf/'

def int_qsolf(logLbolmin, z):
    """
    Integrate the QSO luminosity function above a bolometric luminosity at redshift z
    :param logLbolmin: e.g. 45
    :param z:
    :return: density in hubble units
    """
    originaldir = os.getcwd()
    # use code from Shen et al 2020
    os.chdir(shen2020dir + 'pubtools/')
    import utilities
    lgrid, lf = utilities.return_bolometric_qlf(redshift=z)
    moreluminous = np.where(lgrid > logLbolmin)

    density = hubbleunits.add_h_to_density(np.trapz(10**lf[moreluminous], x=lgrid[moreluminous]))
    os.chdir(originaldir)
    return density

def qso_luminosity_density(logLbolmin, z):
    """
    Integrate QSO luminosity function times luminosity. L * phi
    :param logLbolmin: e.g. 45
    :param z:
    :return: luminosity (erg/s) density in hubble units
    """
    originaldir = os.getcwd()
    # use code from Shen et al 2020
    os.chdir(shen2020dir + 'pubtools/')
    import utilities
    lgrid, lf = utilities.return_bolometric_qlf(redshift=z)
    moreluminous = np.where(lgrid > logLbolmin)

    lphi = (10 ** lf[moreluminous]) * (10 ** lgrid[moreluminous])
    density = hubbleunits.add_h_to_density(np.trapz(lphi, x=lgrid[moreluminous]))
    os.chdir(originaldir)
    return density


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
    dens_hunit = hubbleunits.add_h_to_density(dens.value)
    os.chdir(curdir)
    return dens_hunit

def int_hmf(z, logminmass, massgrid=np.logspace(11, 16, 5000)):
    mfunc = cosmo.hmf_z(np.log10(massgrid), z)
    # number of halos more massive than M is integral of HMF from M to inf
    occupiedidxs = np.where(np.log10(massgrid) > logminmass)
    mfunc, newgrid = mfunc[occupiedidxs], massgrid[occupiedidxs]
    int_at_z = np.trapz(mfunc, x=np.log(newgrid))
    return int_at_z

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
    dndz = redshift_helper.norm_z_dist(dndz)
    zs, dndz = dndz
    ints_at_zs = []
    # for each redshift in grid
    for z in zs:

        ints_at_zs.append(int_hmf(z, logminmass, massgrid=massgrid))

    return np.trapz(np.array(ints_at_zs)*dndz, x=zs)

def hmf_zrange(zrange, logminmass):
    """
    Integrate HMF above log min mass, over a redshift range, assuming constant dN/dz
    :param zrange:
    :param logminmass:
    :return:
    """
    dndz = redshift_helper.dndz_from_z_list(np.random.uniform(zrange[0], zrange[1], 10000), 10)
    return int_hmf_z(dndz=dndz, logminmass=logminmass)

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
