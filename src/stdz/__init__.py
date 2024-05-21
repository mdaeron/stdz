#! /usr/bin/env python3

import numpy
import pandas
import lmfit
import scipy.stats


def sanitize(x):
	return x.replace('-', '').replace('.', '')


class DataFrame:

	def __init__(self, data = [], index_field = 'UID'):
		"""
		Hold data to be standardized
		"""
		if isinstance(data, pandas.DataFrame):
			self.data = data
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


		self.fitresult = fitresult
		self.Nf = fitresult.nfree
		self.t95 = scipy.stats.t.ppf(1 - 0.05 / 2, self.Nf)
		

		self.data[k_in+'_predicted'] = predict_observations(fitresult.params)
		self.data[k_in+'_residual'] = self.data[k_in] - self.data[k_in+'_predicted']
		self.data[k_out+'_corrected'] = correct_observations(fitresult.params)
		self.data[k_out+'_true'] = truevalues(fitresult.params)
		self.data[k_out+'_residual'] = self.data[k_out+'_corrected'] - self.data[k_out+'_true']
		self.data['Unknown'] = ~self.data['Sample'].isin(anchors)
	
	
		self.sessions = pandas.DataFrame()
		self.sessions['N'] = self.data.value_counts(subset='Session')
		NaNu = self.data.groupby(['Session', 'Unknown']).size().to_dict()
		self.sessions['Na'] = self.sessions.index.map({
			s: NaNu[(s, False)] for s in self.sessions.index
		})
		self.sessions['Nu'] = self.sessions.index.map({
			s: NaNu[(s,  True)] for s in self.sessions.index
		})
		self.sessions[f'{k_in}_scaling'] = self.sessions.index.map({
			s: fitresult.params[f"delta_scaling_{sanitize(s)}"].value
			for s in self.sessions.index
		})
		self.sessions[f'SE_{k_in}_scaling'] = self.sessions.index.map({
			s: fitresult.params[f"delta_scaling_{sanitize(s)}"].stderr
			for s in self.sessions.index
		})
		self.sessions[f'{k_out}_of_wg'] = self.sessions.index.map({
			s: fitresult.params[f"delta_of_wg_{sanitize(s)}"].value
			for s in self.sessions.index
		})
		self.sessions[f'SE_{k_out}_of_wg'] = self.sessions.index.map({
			s: fitresult.params[f"delta_of_wg_{sanitize(s)}"].stderr
			for s in self.sessions.index
		})


		self.samples = pandas.DataFrame()
		self.samples['N'] = self.data.value_counts(subset='Sample')
		self.samples[k_in] = (
			self.data
			.groupby('Sample')
			.agg({k_in: 'mean'})
		)
		self.samples[k_out] = self.samples.index.map({
			s: anchors[s] if s in anchors else fitresult.params[f"x_{sanitize(s)}"].value
			for s in self.samples.index
		})
		self.samples['SE_'+k_out] = self.samples.index.map({
			s: None if s in anchors else fitresult.params[f"x_{sanitize(s)}"].stderr
			for s in self.samples.index
		})
		self.samples['95CL_'+k_out] = self.samples['SE_'+k_out] * self.t95
		self.samples['SD_'+k_out] = (
			self.data
			.groupby('Sample')
			.agg({k_out+'_corrected': 'std'})
		)


if __name__ == '__main__':

	from random import random

	N = 100
	df = DataFrame([
		{
			'UID': f'X{_+1:03.0f}',
			'Session': f'2024-{_//20+1:02.0f}',
			'Sample': ['IAEA603', 'IAEA612', 'FOO', 'BAR'][_ % 4],
			'd636': [2.00, -35.00, 35., -20.][_ % 4] + 0.1 * (2*random()-1),
		}
		for _ in range(N)
	])
	df.to_csv('foo.csv', float_format = '%.6f')

	df.standardize('d636', 'd13C_VPDB', dict(IAEA603 = 2.46, IAEA612 = -36.722))
	
	print(df.data)
	print()
	print(df.sessions)
	print()
	print(df.samples)
