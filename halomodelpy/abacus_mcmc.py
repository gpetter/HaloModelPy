import numpy as np
import yaml
from abacusnbody.hod.abacus_hod import AbacusHOD
import emcee
from . import mcmc

path2config = '/home/graysonpetter/ssd/Dartmouth/data/simulations/.config/abacus_hod.yaml'
config = yaml.safe_load(open(path2config))
sim_params = config['sim_params']
HOD_params = config['HOD_params']
clustering_params = config['clustering_params']
want_rsd = HOD_params['want_rsd']
write_to_disk = HOD_params['write_to_disk']
#newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
#mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk, Nthread = 16)

def ln_prior(theta, param_names):

    param_bounds = {'logM_cut': (11.9, 12.4), 'logM1': (13.3, 14.5), 'alpha_c': (0., 5.), 's': (-1., 1.),
                    's_v': (-1., 1.), 'Bcent': (-1., 1.), 'Bsat': (-1., 1.)}
    priorsum = 0
    for j in range(len(theta)):
        paramval = theta[j]
        if (paramval >= param_bounds[param_names[j]][0]) and (paramval <= param_bounds[param_names[j]][1]):
            priorsum += 0.
        else:
            priorsum += -np.inf
    return priorsum


def ln_prob_abacushod(theta, param_names, cf2d_observed, ball, force_qso_satellites, nthread=24):
    rp_bins = cf2d_observed['rp_bins']
    pimax = int(cf2d_observed['pimax'])
    pi_bin_size = int(cf2d_observed['dpi'])
    cf_obs = np.array(cf2d_observed['xi_rp_pi']).ravel()
    cf_obs_err = np.array(cf2d_observed['xi_rp_pi_err']).ravel()

    ball.tracers['QSO']['sigma'] = 1e-4

    prior = ln_prior(theta, param_names)
    if np.isfinite(prior):

        for j in range(len(theta)):
            ball.tracers['QSO'][param_names[j]] = theta[j]
        mock_dict = ball.run_hod(ball.tracers, want_rsd=True, write_to_disk=False, Nthread=nthread, verbose=False,
                                 force_qso_satellites=force_qso_satellites)
        xirppi = ball.compute_xirppi(mock_dict=mock_dict, rpbins=rp_bins,
                                     pimax=pimax, pi_bin_size=pi_bin_size)['QSO_QSO'].ravel()
        residual = cf_obs - xirppi
        likely = mcmc.ln_likelihood(residual=residual, yerr=cf_obs_err)

    else:
        likely = -np.inf

    return likely + prior

def sample_abacus_hod(nwalkers, niter, cf2d, param_names, ball, initial_params=None, pool=None,
                      force_qso_satellites=False, nthread=1):

    ndim = len(param_names)

    #newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    #mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk, Nthread=nthread)


    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    ln_prob_abacushod,
                                    args=[param_names, cf2d, ball, force_qso_satellites, nthread], pool=pool)


    # start walkers near least squares fit position with random gaussian offsets
    pos = np.array(initial_params) + 5e-1 * np.random.normal(size=(sampler.nwalkers, sampler.ndim))

    sampler.run_mcmc(pos, niter, progress=True)


    flatchain = sampler.get_chain(discard=10, flat=True)
    #blobs = sampler.get_blobs(discard=10, flat=True)



    centervals, lowerrs, higherrs = [], [], []
    for i in range(ndim):
        post = np.percentile(flatchain[:, i], [16, 50, 84])
        q = np.diff(post)
        centervals.append(post[1])
        lowerrs.append(q[0])
        higherrs.append(q[1])



    return flatchain, centervals, lowerrs, higherrs