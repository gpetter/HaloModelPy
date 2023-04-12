import numpy as np
from . import params
from . import hubbleunits
from . import bias_tools
import astropy.units as u
paramobj = params.param_obj()
col_cosmo = paramobj.col_cosmo
apcosmo = paramobj.apcosmo

def growthrate(m, z, wantmean=True):
    if wantmean:
        a = 46.1 * u.solMass / u.yr
        b = 1.11
    else:
        a = 25.3 * u.solMass / u.yr
        b = 1.65
    return a * ((m/1e12)**(1.1)) * (1 + b*z)*apcosmo.efunc(z)


def evolve_halo_mass(logmh_init, z_init, z_fin, wantmean=True):
    """
    Evolve a halo mass from an initial to final redshift using the mean or median growth rate of Fakhouri+2010
    :param logmh_init: Initial halo mass at z=z_init, in log10(Msun/h) units
    :param z_init: initial z
    :param z_fin: final z
    :param wantmean: If true, evolve using mean growth rate of halos, otherwise use median growth rate
    :return:
    """
    m_init = hubbleunits.remove_h_from_mass(10 ** logmh_init)
    zgrid = np.flip(np.logspace(np.log10(z_fin), np.log10(z_init), 1000))
    agegrid = apcosmo.age(zgrid)
    agediff = np.diff(agegrid)
    m = m_init
    totms = [m_init]
    for j in range(len(zgrid) - 1):
        growrate = growthrate(m, zgrid[j], wantmean=wantmean)
        m += (growrate * agediff[j]).to('solMass').value
        totms.append(m)
    totms = hubbleunits.add_h_to_mass(np.flip(totms))
    return np.flip(zgrid), np.log10(totms)


def evolve_halo_bias(b_init, z_init, z_fin, wantmean=True):
    """
    Same as evolve_halo_mass, but calculate evolution of bias given an inital bias
    """
    # convert initial bias to halo mass
    logmh_init = bias_tools.bias2mass(inputbias=b_init, z=z_init)
    # evolve the halo over redshift
    zgrid, logms = evolve_halo_mass(logmh_init=logmh_init, z_init=z_init, z_fin=z_fin, wantmean=wantmean)
    # convert mass at each time step back to bias
    b_z = bias_tools.mass2bias(logms, zgrid)

    return np.array(zgrid), np.array(b_z)