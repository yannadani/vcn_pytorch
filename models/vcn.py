import torch
import torch.nn as nn
import math
import utils

class VCN(nn.Module):
	def __init__(self, *, num_nodes, graph_dist, sparsity_factor = 0.0, gibbs_temp_init = 10., gibbs_update = None):
		super().__init__()
		self.num_nodes = num_nodes
		self.graph_dist = graph_dist
		self.sparsity_factor = sparsity_factor
		self.gibbs_temp = gibbs_temp_init
		self.gibbs_update = gibbs_update

	def forward(self, n_samples, bge_model, e, interv_targets = None):
		samples = self.graph_dist.sample([n_samples])	
		log_probs = self.graph_dist.log_prob(samples).squeeze()

		G = utils.vec_to_adj_mat(samples, self.num_nodes) 
		likelihood = bge_model.log_marginal_likelihood_given_g(w = G, interv_targets=interv_targets)

		dagness = utils.expm(G, self.num_nodes)
		self.update_gibbs_temp(e)
		kl_graph = log_probs + self.gibbs_temp*dagness + self.sparsity_factor*torch.sum(G, axis = [-1, -2]) 
		return likelihood, kl_graph, log_probs
	

	def sample(self, num_samples = 10000):
		samples = self.graph_dist.sample([num_samples])
		G = utils.vec_to_adj_mat(ret, self.num_nodes) 
		return G

	def update_gibbs_temp(self, e):
		if self.gibbs_update is None:
			return 0
		else:
			self.gibbs_temp =  self.gibbs_update(self.gibbs_temp, e)