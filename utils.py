import os.path as osp
import shutil, os

import torch
import torch.nn as nn

import numpy as np 
import matplotlib.pyplot as plt
import pickle 
from scipy.special import comb as scomb
from scipy.spatial.distance import cdist
from itertools import product
import networkx as nx

def simulate_gaussian_sem(graphs, w_logvar):
	std = torch.exp(0.5*w_logvar)
	samples = torch.zeros(len(graphs), len(graphs[0].nodes)).to(w_logvar.device)
	for i in range(len(graphs)):
		graph = graphs[i]
		for j in nx.topological_sort(graph):
			noise = torch.normal(mean = 0., std = std[i,j])
			parents = list(graph.predecessors(j))
			if len(parents) == 0:
				samples[i,j] = noise
			else:
				curr = noise
				for k in parents:
					curr += graph.edges[k,j]['weight']*samples[i,k]
				samples[i, j] = curr
	return samples

def matrix_poly(matrix, d):
	x = torch.eye(d).to(matrix.device) + torch.div(matrix, d)
	return torch.matrix_power(x, d)

def expm(A, m):
	expm_A = matrix_poly(A, m)
	h_A = expm_A.diagonal(dim1=-2, dim2=-1).sum(-1) - m
	return h_A

def matrix_poly_np(matrix, d):
	x = np.eye(d) + np.divide(matrix, d)
	return np.linalg.matrix_power(x, d)

def expm_np(A, m):
	expm_A = matrix_poly_np(A, m)
	h_A = np.trace(expm_A) - m
	return h_A

def all_combinations(num_nodes, num_classes=2, return_adj = False):
	comb = list(product(list(range(num_classes)),repeat = num_nodes*(num_nodes-1)))
	comb = np.array(comb)
	if return_adj:
		comb = vec_to_adj_mat_np(comb, num_nodes)
	return comb

def full_prior(num_nodes, num_classes=2, data_size = None, gibbs_temp = 10., sparsity_factor=0.):
	if data_size is None:
		data_size = num_nodes*(num_nodes-1)
	comb = all_combinations(num_classes, data_size)

	ref_config = np.zeros(len(comb))
	comb_full = np.zeros((len(comb), num_nodes, num_nodes))
	for i in range(len(comb)):
		#print(i, len(comb))
		temp_ = np.concatenate((np.zeros((num_nodes,1)),(comb[i].reshape((num_nodes, num_nodes -1)))), axis=1)
		for j in range(num_nodes):
			comb_full[i,j] = np.roll(temp_[j],j)
		resid = expm_np(comb_full[i], num_nodes)
		ref_config[i] = np.exp(-gibbs_temp*resid-sparsity_factor*np.sum(temp_))  
	norm_factor = np.sum(ref_config)
	ref_config = ref_config/norm_factor
	return ref_config, comb, norm_factor, comb_full

def vec_to_adj_mat(matrix, num_nodes):
	matrix = matrix.view(-1, num_nodes, num_nodes-1)
	matrix_full = torch.cat((torch.zeros(matrix.shape[0], num_nodes,1).to(matrix.device), matrix), dim = -1)
	for xx in range(num_nodes):
		matrix_full[:,xx] = torch.roll(matrix_full[:,xx], xx, -1) 
	return matrix_full

def vec_to_adj_mat_np(matrix, num_nodes):
	matrix = np.reshape(matrix, (-1, num_nodes, num_nodes-1))
	matrix_full = np.concatenate((np.zeros((matrix.shape[0], num_nodes,1), dtype = matrix.dtype), matrix), axis = -1)
	for xx in range(num_nodes):
		matrix_full[:,xx] = np.roll(matrix_full[:,xx], xx, axis = -1) 
	return matrix_full

def adj_mat_to_vec(matrix_full, num_nodes):
	for xx in range(num_nodes):
		matrix_full[:,xx] = torch.roll(matrix_full[:,xx], -xx, -1) 
	matrix = matrix_full[..., 1:]
	return matrix.reshape(-1, num_nodes*(num_nodes-1))

def adj_mat_to_vec_np(matrix_full, num_nodes):
	for xx in range(num_nodes):
		matrix_full[:,xx] = np.roll(matrix_full[:,xx], -xx, axis = -1) 
	matrix = np.reshape(matrix_full[..., 1:], (matrix_full.shape[0], num_nodes*(num_nodes-1)))
	return matrix

def shd(B_est, B_true):
	"""Compute various accuracy metrics for B_est.

	true positive = predicted association exists in condition in correct direction
	reverse = predicted association exists in condition in opposite direction
	false positive = predicted association does not exist in condition

	Args:
		B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
		B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

	Returns:
		fdr: (reverse + false positive) / prediction positive
		tpr: (true positive) / condition positive
		fpr: (reverse + false positive) / condition negative
		shd: undirected extra + undirected missing + reverse
		nnz: prediction positive

		Taken from https://github.com/xunzheng/notears
	"""
	if (B_est == -1).any():  # cpdag
		if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
			raise ValueError('B_est should take value in {0,1,-1}')
		if ((B_est == -1) & (B_est.T == -1)).any():
			raise ValueError('undirected edge should only appear once')
	else:  # dag
		if not ((B_est == 0) | (B_est == 1)).all():
			raise ValueError('B_est should take value in {0,1}')
		#if not is_dag(B_est):
		#    raise ValueError('B_est should be a DAG')
	d = B_true.shape[0]
	# linear index of nonzeros
	pred_und = np.flatnonzero(B_est == -1)
	pred = np.flatnonzero(B_est == 1)
	cond = np.flatnonzero(B_true)
	cond_reversed = np.flatnonzero(B_true.T)
	cond_skeleton = np.concatenate([cond, cond_reversed])
	# true pos
	true_pos = np.intersect1d(pred, cond, assume_unique=True)
	# treat undirected edge favorably
	true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
	true_pos = np.concatenate([true_pos, true_pos_und])
	# false pos
	false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
	false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
	false_pos = np.concatenate([false_pos, false_pos_und])
	# reverse
	extra = np.setdiff1d(pred, cond, assume_unique=True)
	reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
	# compute ratio
	pred_size = len(pred) + len(pred_und)
	cond_neg_size = 0.5 * d * (d - 1) - len(cond)
	fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
	tpr = float(len(true_pos)) / max(len(cond), 1)
	fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
	# structural hamming distance
	pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
	cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
	extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
	missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
	shd = len(extra_lower) + len(missing_lower) + len(reverse)
	shd_wc = shd + len(pred_und)
	prc = float(len(true_pos)) / max(float(len(true_pos)+len(reverse) + len(false_pos)), 1.)
	rec = tpr
	return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'prc': prc, 'rec' : rec, 'shd': shd, 'shd_wc': shd_wc, 'nnz': pred_size}

def kl_mdag(log_probs, probs_gt):
	return torch.nn.functional.kl_div(log_probs, probs_gt)

def one_hot(inputs, vocab_size = None):
	"""Returns one hot of data over each element of the inputs"""
	if vocab_size is None:
		vocab_size = inputs.max() + 1
	input_shape = inputs.shape
	inputs = inputs.flatten().unsqueeze(1).long()
	z = torch.zeros(len(inputs), vocab_size).to(inputs.device)
	z.scatter_(1, inputs, 1.)
	return z.view(*input_shape, vocab_size)





