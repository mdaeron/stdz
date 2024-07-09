"""
Standardize stable isotope measurements by comparison with reference materials of known composition
"""

__author__    = 'Mathieu Daëron'
__contact__   = 'daeron@lsce.ipsl.fr'
__copyright__ = 'Copyright (c) 2024 Mathieu Daëron'
__license__   = 'MIT License - https://opensource.org/licenses/MIT'
__date__      = '2024-05-21'
__version__   = '0.0.1'


import numpy
import pandas
import lmfit
import scipy.stats
import scipy.linalg
from sklearn.covariance import MinCovDet
from matplotlib import pyplot
from matplotlib.patches import Ellipse


def estimate_covariance(X, robust = True, support_fraction = None, assume_centered = True):
	# X.shape must be (N,2)
	if robust:
		return MinCovDet(
			support_fraction = support_fraction,
			assume_centered = assume_centered
		).fit(X).covariance_
	else:
		return numpy.cov(X.T)

def sanitize(x):
	return x.replace('-', '').replace('.', '')


def cov_ellipse(CM, p = None, r = None):
	"""
	Parameters
	----------
	CM : (2, 2) array
		Covariance matrix.
	p : float
		Confidence level, should be in (0, 1)

	Returns
	-------
	width, height, rotation :
		 The lengths of two axes and the rotation angle in degree
	for the ellipse.
	"""

	assert r is None or p is None

	if r is None and p is None:
		p = 0.95
		r2 = scipy.stats.chi2.ppf(p, 2)
	if r is not None and p is None:
		r2 = r**2
	if p is not None and r is None:
		r2 = scipy.stats.chi2.ppf(p, 2)

	val, vec = scipy.linalg.eigh(CM)
	width, height = 2 * (val[:, None] * r2)**.5
	rotation = numpy.degrees(numpy.arctan2(*vec[::-1, 0]))

	return width, height, rotation


class Dataset:

	def __init__(self, data = [], index_field = 'UID'):
		"""
		Hold data to be standardized
		"""
		if isinstance(data, pandas.DataFrame):
			self.data = data.copy()
		elif isinstance(data, list):
			self.data = pandas.DataFrame(data)
		else:
			raise ValueError("Invalid data type. Allowed data types are pandas.DataFrame and list of dictionnaries.")
		if index_field:
			self.data.set_index(index_field, inplace = True)


	def read_csv(self, filepath_or_buffer, *args, **kwargs):
		"""
		Read additional data from csv file
		"""
		df = pandas.read_csv(filepath_or_buffer, *args, **kwargs)
		self.data = pandas.concat([self.data, df], ignore_index = True)


	def to_csv(self, filepath_or_buffer, *args, **kwargs):
		"""
		Write data to csv file
		"""
		self.data.to_csv(filepath_or_buffer, *args, **kwargs)


	def standardize(
		self,
		k_in,
		k_out,
		anchors,
		method = 'correct_observations',
		constraints = {},
	):
		"""
		Standardize data based on a set of anchors, i.e. a dict of the form:
		anchors = {
			'RefMaterial_1': RM1_value,
			'RefMaterial_2': RM2_value, ...
		}
		"""
		
		fitparams = lmfit.Parameters()

		for s in self.data.Session.unique():
			fitparams.add('delta_scaling_' + sanitize(s), value = 1.0)
			fitparams.add(  'delta_of_wg_' + sanitize(s), value = 0.0)
		for s in self.data.Sample.unique():
			if s not in anchors:
				fitparams.add('x_' + sanitize(s), value = 0.0)

		for p in fitparams:
			if p in constraints:
				fitparams[p].expr = constraints[p]
		
		def observations():
			return self.data[k_in]

		def truevalues(p):
			return self.data.Sample.map({
				s: anchors[s] if s in anchors else float(p[f"x_{sanitize(s)}"])
				for s in self.data.Sample.unique()
			})

		def correct_observations(p):
			delta_scalinp_map = {
				s: float(p[f"delta_scaling_{sanitize(s)}"])
				for s in self.data.Session.unique()
				}
			delta_of_wg_map = {
				s: float(p[f"delta_of_wg_{sanitize(s)}"])
				for s in self.data.Session.unique()
				}
			return 1e3 * (
				(1 + self.data[k_in] / 1e3 / self.data.Session.map(delta_scalinp_map))
				* (1 + self.data.Session.map(delta_of_wg_map) / 1e3)
				- 1
			)

		def predict_observations(p):
			delta_scalinp_map = {
				s: float(p[f"delta_scaling_{sanitize(s)}"])
				for s in self.data.Session.unique()
				}
			delta_of_wg_map = {
				s: float(p[f"delta_of_wg_{sanitize(s)}"])
				for s in self.data.Session.unique()
				}
			return (
				(
					(1 + truevalues(p) / 1e3)
					/ (1 + self.data.Session.map(delta_of_wg_map) / 1e3)
					- 1
				)
				* 1e3
				* self.data.Session.map(delta_scalinp_map)
			)

		def prediction_residuals(p):
			return (observations() - predict_observations(p)).array

		def correction_residuals(p):
			return (correct_observations(p) - truevalues(p)).array

		residuals = {
			'correct_observations': correction_residuals,
			'predict_observations': prediction_residuals,
		}[method]

		fitresult = lmfit.minimize(
			residuals, fitparams, method = 'least_squares', scale_covar = True
		)

		self.data[k_out+'_anchor'] = self.data['Sample'].isin(anchors)	
		self.data[k_in+'_predicted'] = predict_observations(fitresult.params)
		self.data[k_in+'_residual'] = self.data[k_in] - self.data[k_in+'_predicted']
		self.data[k_out+'_corrected'] = correct_observations(fitresult.params)
		self.data[k_out+'_true'] = truevalues(fitresult.params)
		self.data[k_out+'_residual'] = self.data[k_out+'_corrected'] - self.data[k_out+'_true']

		if not hasattr(self, 'standardization'):
			self.standardization = dict()

		self.standardization[k_out] = dict(
			fitresult = fitresult,
			report = lmfit.fit_report(fitresult),
			Nf = fitresult.nfree,
			t95 = scipy.stats.t.ppf(1 - 0.05 / 2, fitresult.nfree)
		)		

	
		if not hasattr(self, 'sessions'):
			self.sessions = dict()

		self.sessions[k_out] = pandas.DataFrame()
		sessions = self.sessions[k_out]

		sessions['N'] = self.data.value_counts(subset='Session')
		NaNu = self.data.groupby(['Session', k_out+'_anchor']).size().to_dict()
		sessions['Na'] = sessions.index.map({
			s: NaNu[(s, True)] for s in sessions.index
		})
		sessions['Nu'] = sessions.index.map({
			s: NaNu[(s,  False)] for s in sessions.index
		})
		sessions[f'{k_in}_scaling'] = sessions.index.map({
			s: fitresult.params[f"delta_scaling_{sanitize(s)}"].value
			for s in sessions.index
		})
		sessions[f'SE_{k_in}_scaling'] = sessions.index.map({
			s: fitresult.params[f"delta_scaling_{sanitize(s)}"].stderr
			for s in sessions.index
		})
		sessions[f'{k_out}_of_wg'] = sessions.index.map({
			s: fitresult.params[f"delta_of_wg_{sanitize(s)}"].value
			for s in sessions.index
		})
		sessions[f'SE_{k_out}_of_wg'] = sessions.index.map({
			s: fitresult.params[f"delta_of_wg_{sanitize(s)}"].stderr
			for s in sessions.index
		})


		samples = pandas.DataFrame()
		samples['N'] = self.data.value_counts(subset='Sample')
		samples[k_in] = (
			self.data
			.groupby('Sample')
			.agg({k_in: 'mean'})
		)
		samples[k_out] = samples.index.map({
			s: anchors[s] if s in anchors else fitresult.params[f"x_{sanitize(s)}"].value
			for s in samples.index
		})
		samples['SE_'+k_out] = samples.index.map({
			s: None if s in anchors else fitresult.params[f"x_{sanitize(s)}"].stderr
			for s in samples.index
		})
		samples['95CL_'+k_out] = samples['SE_'+k_out] * self.standardization[k_out]['t95']
		samples['SD_'+k_out] = (
			self.data
			.groupby('Sample')
			.agg({k_out+'_corrected': 'std'})
		)

		if hasattr(self, 'samples'):
			columns = self.samples.columns.tolist()
			for c in samples.columns:
				if c not in columns:
					columns.append(c)
			self.samples = self.samples.combine_first(samples)[columns]
		else:
			self.samples = samples.copy()
		
		df_anchors = pandas.DataFrame({k_out: [anchors[s] for s in anchors]}, index = anchors.keys())

		if hasattr(self, 'anchors'):
			self.anchors = self.anchors.combine_first(df_anchors)
		else:
			self.anchors = df_anchors.copy()
			self.anchors.index.name = 'Sample'
		
		# TODO: compute anchor mismatch ie mean and SE of residuals


	def standardize_D17O(
		self,
		d18O_key_in,
		d17O_key_in,
		d18O_key_out,
		d17O_key_out,
		anchors,
		method = 'correct_observations', # 'correct_observations' | 'predict_observations'
		constraints = {},
		residuals_17 = 'd17', # 'D17' | 'd17'
		lambda_17 = 0.528,
		robust_cov_estimator = True,
		relative_sigma_precision = 1e-5,
		max_iter = 10,
	):
		"""
		Standardize triple oxygen isotopes based on a set of anchors, i.e. a dict of the form:
		anchors = {
			'RefMaterial_1': {d18O_key_out: ..., D17O_key_out: ...},
			'RefMaterial_2': {d18O_key_out: ..., D17O_key_out: ...},
		}		
		"""

		D17O_key_out = d17O_key_out.replace('d17', 'D17')

		anchors = anchors.copy()
		for s in anchors:
			if d17O_key_out not in anchors[s]:
				anchors[s][d17O_key_out] = 1e3 * (
					numpy.exp(anchors[s][D17O_key_out] / 1e3)
					* (1 + anchors[s][d18O_key_out] / 1e3)**lambda_17
					- 1
				)
			if D17O_key_out not in anchors[s]:
				anchors[s][D17O_key_out] = 1e3 * (
					numpy.log(1+anchors[s][d17O_key_out] / 1e3)
					- lambda_17 * numpy.log(1+anchors[s][d18O_key_out] / 1e3)
				)
		
		# TODO: check consistency between d17O and D17O in anchors

		fitparams = lmfit.Parameters()

		for s in self.data.Session.unique():
			fitparams.add('d18O_scaling_' + sanitize(s), value = 1.0)
			fitparams.add(  'd18O_of_wg_' + sanitize(s), value = 0.0)
			fitparams.add('d17O_scaling_' + sanitize(s), value = 1.0)
			fitparams.add(  'd17O_of_wg_' + sanitize(s), value = 0.0)
		for s in self.data.Sample.unique():
			if s not in anchors:
				fitparams.add('d18O_' + sanitize(s), value = 0.0)
				fitparams.add('D17O_' + sanitize(s), value = 0.0)

		for p in fitparams:
			if p in constraints:
				fitparams[p].expr = constraints[p]

		def observations():
			return (
				self.data[d18O_key_in],
				self.data[d17O_key_in],
				1e3 * (
					numpy.log(1+self.data[d17O_key_in]/1e3)
					- lambda_17 * numpy.log(1+self.data[d18O_key_in]/1e3)
				),
			)

		def truevalues(p):

			d18true = self.data.Sample.map({
				s: anchors[s][d18O_key_out]
				if s in anchors
				else float(p[f"d18O_{sanitize(s)}"])
				for s in self.data.Sample.unique()
			})

			D17true = self.data.Sample.map({
				s: anchors[s][D17O_key_out]
				if s in anchors
				else float(p[f"D17O_{sanitize(s)}"])
				for s in self.data.Sample.unique()
			})
			
			d17true = 1e3 * (numpy.exp(D17true/1e3) * (1+d18true/1e3)**lambda_17 - 1)

			return (d18true, d17true, D17true)

		def correct_observations(p):
			d18O_scalinp_map = {
				s: float(p[f"d18O_scaling_{sanitize(s)}"])
				for s in self.data.Session.unique()
				}
			d18O_of_wg_map = {
				s: float(p[f"d18O_of_wg_{sanitize(s)}"])
				for s in self.data.Session.unique()
				}
			d18corrected = 1e3 * (
				(1 + self.data[d18O_key_in] / 1e3 / self.data.Session.map(d18O_scalinp_map))
				* (1 + self.data.Session.map(d18O_of_wg_map) / 1e3)
				- 1
			)

			d17O_scalinp_map = {
				s: float(p[f"d17O_scaling_{sanitize(s)}"])
				for s in self.data.Session.unique()
				}
			d17O_of_wg_map = {
				s: float(p[f"d17O_of_wg_{sanitize(s)}"])
				for s in self.data.Session.unique()
				}
			d17corrected = 1e3 * (
				(1 + self.data[d17O_key_in] / 1e3 / self.data.Session.map(d17O_scalinp_map))
				* (1 + self.data.Session.map(d17O_of_wg_map) / 1e3)
				- 1
			)

			D17corrected = 1e3 * (
				numpy.log(1+d17corrected/1e3) - lambda_17 * numpy.log(1+d18corrected/1e3)
			)

			return (d18corrected, d17corrected, D17corrected)

		def predict_observations(p):
			d18true, d17true, D17true = truevalues(p)

			d18O_scalinp_map = {
				s: float(p[f"d18O_scaling_{sanitize(s)}"])
				for s in self.data.Session.unique()
				}
			d18O_of_wg_map = {
				s: float(p[f"d18O_of_wg_{sanitize(s)}"])
				for s in self.data.Session.unique()
				}
			d18predicted = (
				(
					(1 + d18true / 1e3)
					/ (1 + self.data.Session.map(d18O_of_wg_map) / 1e3)
					- 1
				)
				* 1e3
				* self.data.Session.map(d18O_scalinp_map)
			)

			d17O_scalinp_map = {
				s: float(p[f"d17O_scaling_{sanitize(s)}"])
				for s in self.data.Session.unique()
				}
			d17O_of_wg_map = {
				s: float(p[f"d17O_of_wg_{sanitize(s)}"])
				for s in self.data.Session.unique()
				}
			d17predicted = (
				(
					(1 + d17true / 1e3)
					/ (1 + self.data.Session.map(d17O_of_wg_map) / 1e3)
					- 1
				)
				* 1e3
				* self.data.Session.map(d17O_scalinp_map)
			)

			D17predicted = 1e3 * (
				numpy.log(1+d17predicted/1e3) - lambda_17 * numpy.log(1+d18predicted/1e3)
			)

			return (d18predicted, d17predicted, D17predicted)

		def prediction_residuals(p, residuals_17 = residuals_17, ChM = None):

			d18predicted, d17predicted, D17predicted = predict_observations(p)
			d18obs, d17obs, D17obs = observations()

			if residuals_17 == 'D17':
				R = numpy.array([
					d18predicted - d18obs,
					D17predicted - D17obs,
				])

			elif residuals_17 == 'd17':
				R = numpy.array([
					d18predicted - d18obs,
					d17predicted - d17obs,
				])

			if ChM is None:
				return R.flatten()
			else:
				return (ChM @ R).flatten()

		def correction_residuals(p, residuals_17 = residuals_17, ChM = None):

			d18true, d17true, D17true = truevalues(p)
			d18corrected, d17corrected, D17corrected = correct_observations(p)

			if residuals_17 == 'D17':
				R = numpy.array([
					d18corrected - d18true,
					D17corrected - D17true,
				])

			elif residuals_17 == 'd17':
				R = numpy.array([
					d18corrected - d18true,
					d17corrected - d17true,
				])
# 
			if ChM is None:
				return R.flatten()
			else:
				return (ChM @ R).flatten()

		residuals = {
			'correct_observations': correction_residuals,
			'predict_observations': prediction_residuals,
		}[method]

		fitresult = lmfit.minimize(
			residuals, fitparams, method = 'least_squares', scale_covar = True
		)

		_s1, _s2, _r = 0, 0, 10

		for _ in range(max_iter):
			CM = estimate_covariance(
				residuals(fitresult.params).reshape((2, self.data.shape[0])).T,
				robust = robust_cov_estimator,
				)
			if (
				abs(_s1/CM[0,0]**.5 - 1) < relative_sigma_precision
				and abs(_s2/CM[1,1]**.5 - 1) < relative_sigma_precision
				and abs(_r - CM[0,1]/CM[0,0]**.5/CM[1,1]**.5) < relative_sigma_precision
			):
				break

			_s1, _s2, _r = CM[0,0]**.5, CM[1,1]**.5, CM[0,1]/CM[0,0]**.5/CM[1,1]**.5

			ChM = scipy.linalg.cholesky(scipy.linalg.inv(CM))
			fitresult = lmfit.minimize(
				residuals,
				fitparams,
				method = 'least_squares',
				scale_covar = True, # ensures that we are ionsensitive to CM scaling
				kws = {'ChM': ChM},
				)
		else:
			raise ValueError(f'D17O Standardization failed to converge after {_+1} iterations.')

		self.data[D17O_key_out+'_anchor'] = self.data['Sample'].isin(anchors)

		self.data[d18O_key_out+'_corrected'] = correct_observations(fitresult.params)[0]
		self.data[d18O_key_out+'_true'] = truevalues(fitresult.params)[0]
		self.data[d18O_key_out+'_residual'] = self.data[d18O_key_out+'_corrected'] - self.data[d18O_key_out+'_true']

		self.data[D17O_key_out+'_corrected'] = correct_observations(fitresult.params)[2]
		self.data[D17O_key_out+'_true'] = truevalues(fitresult.params)[2]
		self.data[D17O_key_out+'_residual'] = self.data[D17O_key_out+'_corrected'] - self.data[D17O_key_out+'_true']


		if not hasattr(self, 'standardization'):
			self.standardization = dict()

		self.standardization[D17O_key_out] = dict(
			fitresult = fitresult,
			report = lmfit.fit_report(fitresult),
			rchisq = fitresult.redchi,
			Nf = fitresult.nfree // 2,
			t95 = scipy.stats.t.ppf(1 - 0.05 / 2, fitresult.nfree // 2)
		)


		if not hasattr(self, 'sessions'):
			self.sessions = dict()

		self.sessions[D17O_key_out] = pandas.DataFrame()
		sessions = self.sessions[D17O_key_out]

		sessions['N'] = self.data.value_counts(subset='Session')
		NaNu = self.data.groupby(['Session', D17O_key_out+'_anchor']).size().to_dict()
		sessions['Na'] = sessions.index.map({
			s: NaNu[(s, True)] for s in sessions.index
		})
		sessions['Nu'] = sessions.index.map({
			s: NaNu[(s,  False)] for s in sessions.index
		})
		sessions[f'{d18O_key_in}_scaling'] = sessions.index.map({
			s: fitresult.params[f"d18O_scaling_{sanitize(s)}"].value
			for s in sessions.index
		})
		sessions[f'SE_{d18O_key_in}_scaling'] = sessions.index.map({
			s: fitresult.params[f"d18O_scaling_{sanitize(s)}"].stderr
			for s in sessions.index
		})
		sessions[f'{d18O_key_out}_of_wg'] = sessions.index.map({
			s: fitresult.params[f"d18O_of_wg_{sanitize(s)}"].value
			for s in sessions.index
		})
		sessions[f'SE_{d18O_key_out}_of_wg'] = sessions.index.map({
			s: fitresult.params[f"d18O_of_wg_{sanitize(s)}"].stderr
			for s in sessions.index
		})
		sessions[f'{d17O_key_in}_scaling'] = sessions.index.map({
			s: fitresult.params[f"d17O_scaling_{sanitize(s)}"].value
			for s in sessions.index
		})
		sessions[f'SE_{d17O_key_in}_scaling'] = sessions.index.map({
			s: fitresult.params[f"d17O_scaling_{sanitize(s)}"].stderr
			for s in sessions.index
		})
		sessions[f'{d17O_key_out}_of_wg'] = sessions.index.map({
			s: fitresult.params[f"d17O_of_wg_{sanitize(s)}"].value
			for s in sessions.index
		})
		sessions[f'SE_{d17O_key_out}_of_wg'] = sessions.index.map({
			s: fitresult.params[f"d17O_of_wg_{sanitize(s)}"].stderr
			for s in sessions.index
		})


		samples = pandas.DataFrame()
		samples['N'] = self.data.value_counts(subset='Sample')

		samples[d18O_key_in] = (
			self.data
			.groupby('Sample')
			.agg({d18O_key_in: 'mean'})
		)

		samples[d17O_key_in] = (
			self.data
			.groupby('Sample')
			.agg({d17O_key_in: 'mean'})
		)

		samples[d18O_key_out] = samples.index.map({
			s: anchors[s][d18O_key_out] if s in anchors else fitresult.params[f"d18O_{sanitize(s)}"].value
			for s in samples.index
		})
		samples['SE_'+d18O_key_out] = samples.index.map({
			s: None if s in anchors else fitresult.params[f"d18O_{sanitize(s)}"].stderr
			for s in samples.index
		})
		samples['95CL_'+d18O_key_out] = samples['SE_'+d18O_key_out] * self.standardization[D17O_key_out]['t95']
		samples['SD_'+d18O_key_out] = (
			self.data
			.groupby('Sample')
			.agg({d18O_key_out+'_corrected': 'std'})
		)

		samples[d17O_key_out] = 0.

		samples[D17O_key_out] = samples.index.map({
			s: anchors[s][D17O_key_out] if s in anchors else fitresult.params[f"D17O_{sanitize(s)}"].value
			for s in samples.index
		})
		samples['SE_'+D17O_key_out] = samples.index.map({
			s: None if s in anchors else fitresult.params[f"D17O_{sanitize(s)}"].stderr
			for s in samples.index
		})
		samples['95CL_'+D17O_key_out] = samples['SE_'+D17O_key_out] * self.standardization[D17O_key_out]['t95']
		samples['SD_'+D17O_key_out] = (
			self.data
			.groupby('Sample')
			.agg({D17O_key_out+'_corrected': 'std'})
		)

		samples[d17O_key_out] = 1e3 * (
				numpy.exp(samples[D17O_key_out] / 1e3)
				* (1 + samples[d18O_key_out] / 1e3)**lambda_17
				- 1
			)

		# TODO: propagate uncertainties to d17O_key_out using the code below
		# 
		# try:
		# 	i = fitresult.var_names.index('d18O_'+_s)
		# 	j = fitresult.var_names.index('D17O_'+_s)
		# 	CM = fitresult.covar[ix_([i,j],[i,j])]
		# 
		# 	J = array([
		# 		[
		# 			exp(out['samples'][s][D17O_key_out]/1e3)
		# 			* ISOPARAMETERS.LAMBDA_17
		# 			* (1+out['samples'][s][d18O_key_out]/1e3) ** (ISOPARAMETERS.LAMBDA_17 - 1)
		# 			],
		# 		[
		# 			exp(out['samples'][s][D17O_key_out]/1e3)
		# 			* (1+out['samples'][s][d18O_key_out]/1e3) ** ISOPARAMETERS.LAMBDA_17
		# 			],
		# 		])
		# 
		# 	out['samples'][s]['SE_'+d17O_key_out] = float(J.T @ CM @ J)**.5
		# 	out['samples'][s]['95CL_'+d17O_key_out] = out['samples'][s]['SE_'+d17O_key_out] * out['t95']
		# except ValueError:
		# 	pass

		if hasattr(self, 'samples'):
			columns = self.samples.columns.tolist()
			for c in samples.columns:
				if c not in columns:
					columns.append(c)
			self.samples = pandas.concat([self.samples, samples], axis = 'columns')
		else:
			self.samples = samples.copy()


		df_anchors = pandas.DataFrame([{
			d18O_key_out: anchors[s][d18O_key_out],
			D17O_key_out: anchors[s][D17O_key_out],
			} for s in anchors], index = anchors.keys()
		)

		if hasattr(self, 'anchors'):
# 			self.anchors = self.anchors.combine_first(df_anchors)
			self.anchors = pandas.concat([self.anchors, df_anchors], axis = 'columns')
		else:
			self.anchors = df_anchors.copy()
			self.anchors.index.name = 'Sample'
		
		# TODO: compute anchor mismatch ie mean and SE of residuals

	def plot_D17O_residuals(self,
		d18O_key_out = 'd18O_VSMOW',
		D17O_key_out = 'D17O_VSMOW',
		plot_margins = 0.25,
		cbins = [-1, 0, 0.25, 0.5, 0.75, 0.95],
		robust_cov_estimator = True,
		kw_plot = dict(ls = 'None', marker = '+', mec = 'k', alpha = 0.5, mew = 0.5, ms = 4),
		kw_ellipse = dict(ec = 'k', lw = 1, ls = (0, (6, 2, 2, 2))),
	):
		x = self.data[d18O_key_out+'_residual'].array
		y = self.data[D17O_key_out+'_residual'].array

		xmin, xmax = x.min(), x.max()
		xmin, xmax = (
			xmin - (xmax - xmin) * plot_margins,
			xmax + (xmax - xmin) * plot_margins,
		)

		ymin, ymax = y.min(), y.max()
		ymin, ymax = (
			ymin - (ymax - ymin) * plot_margins,
			ymax + (ymax - ymin) * plot_margins,
		)

		xx, yy = numpy.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
		positions = numpy.vstack([xx.ravel(), yy.ravel()])
		values = numpy.vstack([x, y])
		kernel = scipy.stats.gaussian_kde(values)

		f = numpy.reshape(kernel(positions).T, xx.shape)
		f /= f.max()
		f = 1 - f

		ax = pyplot.gca()

		ax.set_xlim(xmin, xmax)
		ax.set_ylim(ymin, ymax)

		cfset = ax.contourf(xx, yy, f, levels = cbins, cmap = 'Blues_r', alpha = 1)
		cset = ax.contour(xx, yy, f, levels = cbins, colors = 'k')
		ax.plot(x, y, **kw_plot)

		ax.clabel(cset, inline = 1, fontsize = 10)
		ax.set_xlabel('δ$^{18}$O (‰)')
		ax.set_ylabel('Δ$^{17}$O (ppm)')

		## Or kernel density estimate plot instead of the contourf plot
		# ax.imshow(numpy.rot90(f), cmap = 'Blues', extent = [xmin, xmax, ymin, ymax])		# Contour plot

		CM = (
			estimate_covariance(numpy.array([x, y]).T, robust = robust_cov_estimator)
			* self.standardization[D17O_key_out]['rchisq']
		)
		print(self.standardization[D17O_key_out]['rchisq'], 1000*CM[0,0]**.5, 1000*CM[1,1]**.5)
		w, h, r = cov_ellipse(CM)
		ax.add_patch(Ellipse(xy = (0, 0), width = w, height = h, angle = r, fc = 'None', **kw_ellipse))
		ax.plot(
			[], [],
			label = '95 % confidence from\ncovariance estimate',
			color = kw_ellipse['ec'],
			lw = kw_ellipse['lw'],
			ls = kw_ellipse['ls'],
		)

		ax.legend(fontsize = 7, handlelength = 2.5)

		return ax


if __name__ == '__main__':

	from random import gauss

	N = 100
	_df = pandas.DataFrame([
		{
			'UID': f'X{_+1:03.0f}',
			'Session': f'2024-{_//20+1:02.0f}',
			'Sample': ['IAEA603', 'IAEA612', 'NBS18', 'FOO', 'BAR'][_ % 5],
			'd636': [2.00, -35.00, -6., 35., -20.][_ % 5] + 0.1 * gauss(),
			'd628': [-2.00, -11.00, -23.0, 0., 20.][_ % 5] + 0.1 * gauss(),
			'D627': 0.01 * gauss(),
		}
		for _ in range(N)
	])

	_df['d627']	= (numpy.exp(_df['D627']/1e3) * (1+_df['d628']/1e3)**0.528 - 1) * 1e3

	df = Dataset(_df)

	df.standardize('d636', 'd13C_VPDB', dict(IAEA603 = 2.46, IAEA612 = -36.722))	
	df.standardize('d628', 'd18O_VPDB', dict(IAEA603 = -2.37, NBS18 = -23.01))	
	df.standardize_D17O(
		'd628', 'd627', 'd18O_VSMOW', 'd17O_VSMOW',
		dict(IAEA603 = dict(d18O_VSMOW = -2, D17O_VSMOW = 0.), NBS18 = dict(d18O_VSMOW = -23., D17O_VSMOW = 0.)),
	)	

	df.plot_D17O_residuals()
	
	df.to_csv('foo.csv', float_format = '%.6f')

	print(df.data)
	for k in df.sessions:
		print()
		print(f'[{k}]')
		print(df.sessions[k])
	for k in df.standardization:
		print()
		print(f'[{k}]')
		print({_: df.standardization[k][_] for _ in df.standardization[k] if _ != 'report'})
	print()
	print(df.samples)
	print()
	print(df.anchors)
