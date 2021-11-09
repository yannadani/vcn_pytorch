import torch
import numpy as np

class LiftDim(object):
	"""Get high dimensional samples from lower dimensional observations
	Arguments:
	dim: Number of dimensions to be lifted. If dim is None, it automarically chooses D = 2*d 
	where d is the number of nodes.
	type: type of function used to project. Options: {"linear", "non-linear"}

	For the "non-linear" case, choose the dim to be an integer multiple of d.
	"""
	def __init__(self, dim = None, type="non-linear"):
		self.dim = dim
		self.type = type

	def __call__(self, data):
		d = data.shape[-1]
		if self.dim is None:
			self.dim = 2*d

		if self.type == "linear":
			return self.linear_project(data)
		elif self.type == "non-linear":
			return self.nonlinear_project(data)
		else:
			raise NotImplementedError

	def nonlinear_project(self, data):
		d = data.shape[-1]
		num_features = int(self.dim/d)
		projected_data = torch.zeros(data.shape[0], self.dim)
		ind = 0
		for i in range(1, num_features + 1):
			projected_data[:,ind:ind+d] = data**i
			ind += d
		return projected_data

	def linear_project(self, data):
		d = data.shape[-1]
		det_matrix = 0
		while not det_matrix == 0:
			random_project_matrix = torch.randn(d, self.dim)
			det_matrix =  torch.det(random_project_matrix)
		return torch.matmul(data, random_project_matrix)

	