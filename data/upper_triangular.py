import numpy as np
import torch
from .generator import Generator
import networkx as nx 
import graphical_models

class UT(Generator):
	"""Generate upper triangular random graphs
		Args: 
		num_nodes - Number of Nodes in the graph
		noise_type - Type of exogenous variables
		noise_sigma - Std of the noise type
		num_sampels - number of observations
		mu_prior - prior of weights mean(gaussian)
		sigma_prior - prior of weights sigma (gaussian)
		seed - random seed for data
	"""
	
	def __init__(self, num_nodes, noise_type='isotropic-gaussian', noise_sigma = 1.0, num_samples=1000, mu_prior = 2.0, sigma_prior = 1.0, seed = 10):
		self.noise_sigma = noise_sigma
		torch.manual_seed(seed)
		adj_mat_full = torch.bernoulli(torch.ones(num_nodes, num_nodes)*0.5)
		self.graph = nx.DiGraph(torch.triu(adj_mat_full, diagonal = 1).numpy())
		super().__init__(num_nodes, len(self.graph.edges), noise_type, num_samples, mu_prior = mu_prior , sigma_prior = sigma_prior, seed = seed)
		self.init_sampler()
		self.samples = self.sample(self.num_samples)

	def __getitem__(self, index):
		return self.samples[index]