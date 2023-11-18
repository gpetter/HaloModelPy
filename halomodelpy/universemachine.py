from . import cosmo, hubbleunits
import numpy as np
import sys
import math
import re

def mpeak2m200c(log_mpeaks, z):
	"""
	UniverseMachine reports halo mass in Mpeak, virial overdensity, with no hubble units
	Convert halo masses to M200c, with hubble units
	:param log_mpeaks:
	:param z:
	:return:
	"""
	mpeak = 10**log_mpeaks
	mpeak_h = hubbleunits.add_h_to_mass(mpeak)
	m200c_h = cosmo.convert_halomass(mpeak_h, z, 'vir', '200c')
	return np.log10(m200c_h)

parampath = '/home/graysonpetter/ssd/Dartmouth/data/universemachine/umachine-dr1/data/smhm/params/'



def smhm(z, convertmass=True, truemass=False, centrals=False, satellites=False, quiescent=False, starforming=False,
		 icl=False):
	"""
	Adapted from code at https://www.peterbehroozi.com/data.html
	:param z:
	:param convertmass: Convert mass from virial to m200c with little h?
	:param truemass: "True" stellar mass - Section 5 Wang+21
	:param centrals: Only centrals
	:param satellites: Only satellites
	:param quiescent: Only red galaxies
	:param starforming: Only blue galaxies
	:param icl: Including intracluster light, stars at large radii counted in stellar mass
	:return:
	"""
	filename = parampath + 'smhm_'
	if truemass:
		filename += 'true_med_'
	else:
		filename += 'med_'

	if centrals:
		filename += 'cen_'
	if satellites:
		filename += 'sat_'
	if quiescent:
		filename += 'q_'
	if starforming:
		filename += 'sf_'
	if icl:
		filename += 'icl_'
	filename += 'params.txt'


	# Load params
	param_file = open(filename, "r")
	param_list = []
	allparams = []
	for line in param_file:
		param_list.append(float((line.split(" "))[1]))
		allparams.append(line.split(" "))

	if (len(param_list) != 20):
		print("Parameter file not correct length.  (Expected 20 lines, got %d)." % len(param_list))
		quit()

	names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(
		" ");
	params = dict(zip(names, param_list))

	# Decide whether to print tex or evaluate SMHM parameter
	try:
		z = float(z)
	except:
		# print TeX
		for x in allparams[0:10:1]:
			x[3] = -float(x[3])
			sys.stdout.write('& $%.3f^{%+.3f}_{%+.3f}$' % tuple(float(y) for y in x[1:4]))
		sys.stdout.write("\\\\\n & & & ")
		for x in allparams[10:19:1]:
			x[3] = -float(x[3])
			sys.stdout.write('& $%.3f^{%+.3f}_{%+.3f}$' % tuple(float(y) for y in x[1:4]))
		#    sys.stdout.write("\\\\\n & & & ")
		#    for x in allparams[16:19:1]:
		#        x[3] = -float(x[3])
		#        sys.stdout.write('& $%.3f^{%+.3f}_{%+.3f}$' % tuple(float(y) for y in x[1:4]))
		sys.stdout.write(' & %.0f' % float(allparams[19][1]))
		if (float(allparams[19][1]) > 200):
			sys.stdout.write('$\dag$')
		print('\\\\[2ex]')
		quit()

	# Print SMHM relation
	a = 1.0 / (1.0 + z)
	a1 = a - 1.0
	lna = math.log(a)
	zparams = {}
	zparams['m_1'] = params['M_1'] + a1 * params['M_1_A'] - lna * params['M_1_A2'] + z * params['M_1_Z']
	zparams['sm_0'] = zparams['m_1'] + params['EFF_0'] + a1 * params['EFF_0_A'] - lna * params['EFF_0_A2'] + z * params[
		'EFF_0_Z']
	zparams['alpha'] = params['ALPHA'] + a1 * params['ALPHA_A'] - lna * params['ALPHA_A2'] + z * params['ALPHA_Z']
	zparams['beta'] = params['BETA'] + a1 * params['BETA_A'] + z * params['BETA_Z']
	zparams['delta'] = params['DELTA']
	zparams['gamma'] = 10 ** (params['GAMMA'] + a1 * params['GAMMA_A'] + z * params['GAMMA_Z'])

	smhm_max = 14.5 - 0.35 * z
	# print('#Log10(Mpeak/Msun) Log10(Median_SM/Msun) Log10(Median_SM/Mpeak)')
	# print('#Mpeak: peak historical halo mass, using Bryan & Norman virial overdensity.')
	print('#Overall fit chi^2: %f' % params['CHI2'])
	if (params['CHI2'] > 200):
		print(
			'#Warning: chi^2 > 200 implies that not all features are well fit.  Comparison with the raw data (in data/smhm/median_raw/) is crucial.')
	hms, sms = [], []
	for m in [x * 0.05 for x in range(int(10.5 * 20), int(smhm_max * 20 + 1), 1)]:
		hms.append(m)
		dm = m - zparams['m_1'];
		dm2 = dm / zparams['delta'];
		sm = zparams['sm_0'] - math.log10(10 ** (-zparams['alpha'] * dm) + 10 ** (-zparams['beta'] * dm)) + zparams[
			'gamma'] * math.exp(-0.5 * (dm2 * dm2))
		sms.append(sm)
	hms = np.array(hms)
	if convertmass:
		hms = mpeak2m200c(hms, z)

	outdict = {'Mh': hms, 'SM': np.array(sms)}
	return outdict


def smhm_unc(z, convertmass=True, truemass=False, centrals=False, satellites=False, quiescent=False, starforming=False,
			 icl=False):
	"""
	Adapted from code at https://www.peterbehroozi.com/data.html
	:param z:
	:param convertmass: Convert mass from virial to m200c with little h?
	:param truemass: "True" stellar mass - Section 5 Wang+21
	:param centrals: Only centrals
	:param satellites: Only satellites
	:param quiescent: Only red galaxies
	:param starforming: Only blue galaxies
	:param icl: Including intracluster light, stars at large radii counted in stellar mass
	:return:
	"""
	filename = parampath + 'uncertainties_smhm_'
	if truemass:
		filename += 'true_med_'
	else:
		filename += 'med_'

	if centrals:
		filename += 'cen_'
	if satellites:
		filename += 'sat_'
	if quiescent:
		filename += 'q_'
	if starforming:
		filename += 'sf_'
	if icl:
		filename += 'icl_'
	filename += 'params.txt'

	# Load params
	param_file = open(filename, "r")
	param_list = []
	all_params = []
	for line in param_file:
		if (re.search('^#', line)):
			continue
		param_list = [float(x) for x in line.split(" ")]
		all_params.append(param_list)

	if (len(param_list) < 20):
		print("Parameter file not correct length.  (Expected 20 lines, got %d)." % len(param_list))
		quit()

	names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(
		" ");
	# params = dict(zip(names, param_list))

	# Decide whether to print tex or evaluate SMHM parameter
	try:
		z = float(z)
	except:
		print("Usage: %s z uncertainties_smhm_parameter_file.txt" % sys.argv[0])
		quit()

	def gen_smhm(param_list, z):
		params = dict(zip(names, param_list))
		a = 1.0 / (1.0 + z)
		a1 = a - 1.0
		lna = math.log(a)
		zparams = {}
		zparams['m_1'] = params['M_1'] + a1 * params['M_1_A'] - lna * params['M_1_A2'] + z * params['M_1_Z']
		zparams['sm_0'] = zparams['m_1'] + params['EFF_0'] + a1 * params['EFF_0_A'] - lna * params['EFF_0_A2'] + z * \
						  params['EFF_0_Z']
		zparams['alpha'] = params['ALPHA'] + a1 * params['ALPHA_A'] - lna * params['ALPHA_A2'] + z * params['ALPHA_Z']
		zparams['beta'] = params['BETA'] + a1 * params['BETA_A'] + z * params['BETA_Z']
		zparams['delta'] = params['DELTA']
		zparams['gamma'] = 10 ** (params['GAMMA'] + a1 * params['GAMMA_A'] + z * params['GAMMA_Z'])
		smhm_max = 14.5 - 0.35 * z
		sms = []
		for m in [x * 0.05 for x in range(int(10.5 * 20), int(smhm_max * 20 + 1), 1)]:
			dm = m - zparams['m_1'];
			dm2 = dm / zparams['delta'];
			sm = zparams['sm_0'] - math.log10(10 ** (-zparams['alpha'] * dm) + 10 ** (-zparams['beta'] * dm)) + zparams[
				'gamma'] * math.exp(-0.5 * (dm2 * dm2));
			sms.append(sm);
		return sms

	best_sms = gen_smhm(all_params[0], z)
	all_sms = []
	for params in (all_params):
		sms = gen_smhm(params, z)
		all_sms.append(sms)

	params = dict(zip(names, all_params[0]))
	print('#Best fit chi^2: %f' % params['CHI2'])
	if (params['CHI2'] > 200):
		print(
			'#Warning: chi^2 > 200 implies that not all features are well fit.  Comparison with the raw data (in data/smhm/median_raw/) is crucial.')

	smhm_max = 14.5 - 0.35 * z
	ms = [x * 0.05 for x in range(int(10.5 * 20), int(smhm_max * 20 + 1), 1)]
	hms, sms, uperrs, loerrs = [], [], [], []
	for i in range(0, len(ms)):
		m = ms[i]
		hms.append(m)
		sm_best = best_sms[i]
		sms.append(sm_best)
		all_sms.sort(key=lambda x: x[i])
		sm_up = all_sms[int((1 + 0.6827) * len(all_sms) / 2.0)][i]
		sm_down = all_sms[int((1 - 0.6827) * len(all_sms) / 2.0)][i]
		uperrs.append(sm_up - sm_best)
		loerrs.append(sm_best - sm_down)

	hms = np.array(hms)
	if convertmass:
		hms = mpeak2m200c(hms, z)
	outdict = {'Mh': hms, 'SM': np.array(sms), 'SM_upp': np.array(uperrs), 'SM_low': np.array(loerrs)}
	return outdict



