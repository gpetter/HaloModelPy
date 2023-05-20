from corrfunc_helper import twoPointCFs
from . import clustering_fit
from . import redshift_helper
import numpy as np

def measure_and_fit_autocf(scales, datcat, randcat, nthreads=None, estimator='LS', pimax=40.,
					 dpi=1., mubins=None, nbootstrap=500, oversample=1,
					 wedges=None, nzbins=15, dndL=None, dndz=None):

		cf = twoPointCFs.autocorr_cat(scales, datcat, randcat,
										nthreads=nthreads, estimator=estimator, pimax=pimax, dpi=dpi,
										mubins=mubins, nbootstrap=nbootstrap, oversample=oversample,
										wedges=wedges)
		if dndz is None:
			dndz = redshift_helper.dndz_from_z_list(randcat['Z'], nzbins)
		bestfit = clustering_fit.fit_pipeline(dndz, cf, dndL=dndL)
		return bestfit, cf


def measure_and_fit_xcf(autocf, scales, datcat1, datcat2, randcat1, randcat2=None, nthreads=None, estimator='Peebles',
						pimax=40., dpi=1., mubins=None, nbootstrap=500, oversample=1, wedges=None, nzbins=15,
						dndz_auto=None, dndz_cross=None):

	xcf = twoPointCFs.crosscorr_cats(scales=scales, datcat1=datcat1, datcat2=datcat2,
										randcat1=randcat1, randcat2=randcat2,
										nthreads=nthreads, estimator=estimator, pimax=pimax, dpi=dpi, mubins=mubins,
										nbootstrap=nbootstrap, oversample=oversample, wedges=wedges)

	if dndz_auto is None:
		dndz_auto = redshift_helper.dndz_from_z_list(randcat1['Z'], nzbins)
		minz, maxz = np.min(randcat1['Z']), np.max(randcat1['Z'])
		dndz_cross = redshift_helper.dndz_from_z_list(datcat2['Z'], nzbins, (minz, maxz))
	bestfit = clustering_fit.xfit_pipeline(dndz_cross, xcf, dndz_auto, autocf)
	return bestfit, xcf


def measure_and_fit_auto_and_xcf(scales, datcat1, datcat2, randcat1, randcat2=None, nthreads=None, estimator='Peebles',
						pimax=40., dpi=1., mubins=None, nbootstrap=500, oversample=1, wedges=None, nzbins=15,
						dndz_auto=None, dndz_cross=None):
	autocf = twoPointCFs.autocorr_cat(scales, datcat1, randcat1, nthreads=nthreads,
										estimator='LS', pimax=pimax,
										dpi=dpi, mubins=mubins, nbootstrap=nbootstrap,
										oversample=oversample, wedges=wedges)
	xcf = twoPointCFs.crosscorr_cats(scales=scales, datcat1=datcat1, datcat2=datcat2,
										randcat1=randcat1, randcat2=randcat2,
										nthreads=nthreads, estimator=estimator, pimax=pimax, dpi=dpi, mubins=mubins,
										nbootstrap=nbootstrap, oversample=oversample, wedges=wedges)
	if dndz_auto is None:
		dndz_auto = redshift_helper.dndz_from_z_list(randcat1['Z'], nzbins)
		minz, maxz = np.min(randcat1['Z']), np.max(randcat1['Z'])
		dndz_cross = redshift_helper.dndz_from_z_list(datcat2['Z'], nzbins, (minz, maxz))
	bestfit = clustering_fit.xfit_pipeline(dndz_cross, xcf, dndz_auto, autocf)
	return bestfit, autocf, xcf