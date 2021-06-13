from anndata import read_h5ad
import sys
from time import time
from scipy import stats, sparse
import numpy as np
import collections
import pickle
from sklearn.preprocessing import normalize
import os
from collections import Counter
from scipy import spatial
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,precision_recall_fscore_support, cohen_kappa_score, auc, average_precision_score,f1_score,precision_recall_curve
import time
import copy
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
#from libs import *
from sklearn.utils.graph_shortest_path import graph_shortest_path
from scipy.sparse.linalg import svds, eigs

def get_ontology_parents(GO_net, g, dfs_depth):
	term_valid = set()
	ngh_GO = set()
	ngh_GO.add(g)
	depth = {}
	depth[g] = 0
	while len(ngh_GO) > 0:
		for GO in list(ngh_GO):
			for GO1 in GO_net[GO]:
				ngh_GO.add(GO1)
				depth[GO1] = depth[GO] + 1
			ngh_GO.remove(GO)
			if depth[GO] < dfs_depth:
				term_valid.add(GO)
	return term_valid

def creat_cell_ontology_matrix(train_Y, co2co_graph, cell_ontology_ids, dfs_depth):
	lset = set(cell_ontology_ids)

	seen_l = sorted(np.unique(train_Y))
	unseen_l = sorted(lset - set(train_Y))
	ys =  np.concatenate((seen_l, unseen_l))
	i2l = {}
	l2i = {}
	for l in ys:
		nl = len(i2l)
		l2i[l] = nl
		i2l[nl] = l
	nco = len(i2l)

	net_dict = collections.defaultdict(dict)
	net_mat = np.zeros((nco,nco))
	for co1 in co2co_graph:
		l1 = l2i[co1]
		for co2 in co2co_graph[co1]:
			l2 = l2i[co2]
			net_dict[l1][l2] = 1
			net_mat[l1][l2] = 1
	for n in range(nco):
		ngh = get_ontology_parents(net_dict, n, dfs_depth)
		net_dict[n][n] = 1
		for n1 in ngh:
			net_dict[n][n1] = 1
	return unseen_l, l2i, i2l, net_dict, net_mat


def create_propagate_networks_using_nlp(l2i, onto_net, cls2cls, co2vec_nlp, rsts = [0.5,0.6,0.7,0.8], diss=[2,3], thress=[1,0.8]):
	ncls = np.shape(cls2cls)[0]
	#onto_net_nlp, onto_net_bin, stack_net_nlp, stack_net_bin, onto_net_nlp_all_pairs = create_nlp_networks(l2i, onto_net, cls2cls, ontology_nlp_file)
	#ncls = np.shape(cls2cls)[0]
	onto_net_nlp_all_pairs = (cosine_similarity(co2vec_nlp) + 1 ) /2#1 - spatial.distance.cosine(onto_nlp_emb, onto_nlp_emb)
	onto_net_nlp = np.zeros((ncls, ncls))
	stack_net_nlp = np.zeros((ncls, ncls))

	for n1 in onto_net:
		for n2 in onto_net[n1]:
			if n1==n2:
				continue
			stack_net_nlp[n2,n1] = onto_net_nlp_all_pairs[n2, n1]
			stack_net_nlp[n1,n2] = onto_net_nlp_all_pairs[n1, n2]
	for n1 in range(ncls):
		for n2 in range(ncls):
			if cls2cls[n1,n2] == 1 or cls2cls[n2,n1] == 1:
				onto_net_nlp[n1,n2] = onto_net_nlp_all_pairs[n1, n2]
				onto_net_nlp[n2,n1] = onto_net_nlp_all_pairs[n2, n1]
	#network = create_consensus_networks(rsts, stack_net_nlp, onto_net_nlp_all_pairs, cls2cls)
	cls2cls_sp = graph_shortest_path(cls2cls,method='FW',directed =False)
	networks = []
	for rst in rsts:
		for dis in diss:
			for thres in thress:
				#use_net = np.copy(stack_net_nlp)
				use_net = np.copy(onto_net_nlp)
				use_net[(cls2cls_sp<=dis)&(onto_net_nlp_all_pairs > thres)] = onto_net_nlp_all_pairs[(cls2cls_sp<=dis)&(onto_net_nlp_all_pairs > thres)]
				onto_net_rwr = RandomWalkRestart(use_net, rst)
				networks.append(onto_net_rwr)
	return networks


def map_genes(test_X, test_genes, train_genes):
	ntest_cell = np.shape(test_X)[0]
	ntrain_gene = len(train_genes)
	new_test_x = np.zeros((ntest_cell, ntrain_gene))
	genes = set(test_genes) & set(train_genes)
	train_genes = list(train_genes)
	test_genes = list(test_genes)
	ind1 = []
	ind2 = []
	for i,g in enumerate(genes):
		ind1.append(train_genes.index(g))
		ind2.append(test_genes.index(g))
	ind1 = np.array(ind1)
	ind2 = np.array(ind2)
	new_test_x[:,ind1] = test_X[:,ind2]
	return new_test_x


def extend_prediction_2unseen(pred_Y_seen, networks, nseen, ratio=200, use_normalize=False):
	if not isinstance(networks, list):
		networks = [networks]
	pred_Y_all_totoal = 0.
	for onto_net_rwr in networks:
		if use_normalize:
			onto_net_rwr = onto_net_rwr - np.tile(np.mean(onto_net_rwr, axis = 1), (np.shape(onto_net_rwr)[0], 1))
		pred_Y_seen_norm = pred_Y_seen / pred_Y_seen.sum(axis=1)[:, np.newaxis]
		pred_Y_all = np.dot(pred_Y_seen_norm, onto_net_rwr[:nseen,:])
		pred_Y_all[:,:nseen] = normalize(pred_Y_all[:,:nseen],norm='l1',axis=1)
		pred_Y_all[:,nseen:] = normalize(pred_Y_all[:,nseen:],norm='l1',axis=1) * ratio
		pred_Y_all_totoal += pred_Y_all
	return pred_Y_all_totoal

def create_consensus_networks(rsts, onto_net_mat, onto_net_nlp_all_pairs, cls2cls, diss=[2,3], thress=[1,0.8]):
	cls2cls_sp = graph_shortest_path(cls2cls,method='FW',directed =False)
	ncls = np.shape(onto_net_mat)[0]
	networks = []
	for rst in rsts:
		for dis in diss:
			for thres in thress:
				use_net = np.copy(onto_net_mat)
				use_net[(cls2cls_sp<=dis)&(onto_net_nlp_all_pairs > thres)] = onto_net_nlp_all_pairs[(cls2cls_sp<=dis)&(onto_net_nlp_all_pairs > thres)]
				onto_net_rwr = RandomWalkRestart(use_net, rst)
				networks.append(onto_net_rwr)
	return networks


def fine_nearest_co_using_nlp(sentences,co2emb,cutoff=0.8):
	from sentence_transformers import SentenceTransformer
	model = SentenceTransformer('bert-base-nli-mean-tokens')
	sentence_embeddings = model.encode(sentences)
	co_embeddings = []
	cos = []
	for co in co2emb:
		co_embeddings.append(co2emb[co])
		cos.append(co)
	co_embeddings = np.array(co_embeddings)
	sent2co = {}
	for sentence, embedding, ind in zip(sentences, sentence_embeddings, range(len(sentences))):
		scs = cosine_similarity(co_embeddings, embedding.reshape(1,-1))

		co_id = np.argmax(scs)
		sc = scs[co_id]
		if sc>cutoff:
			sent2co[sentence] = cos[co_id]
	return sent2co


def read_cell_type_nlp_network(nlp_emb_file, cell_type_network_file):
	cell_ontology_ids = set()
	fin = open(cell_type_network_file)
	co2co_graph = {}
	for line in fin:
		w = line.strip().split('\t')
		if w[0] not in co2co_graph:
			co2co_graph[w[0]] = set()
		co2co_graph[w[0]].add(w[1])
		cell_ontology_ids.add(w[0])
		cell_ontology_ids.add(w[1])
	fin.close()
	if nlp_emb_file is not None:
		fin = open(nlp_emb_file)
		co2vec_nlp = {}
		for line in fin:
			w = line.strip().split('\t')
			vec = []
			for i in range(1,len(w)):
				vec.append(float(w[i]))
			co2vec_nlp[w[0]] = np.array(vec)
		fin.close()
		co2co_nlp = {}
		for id1 in co2co_graph:
			co2co_nlp[id1] = {}
			for id2 in co2co_graph[id1]:
				sc = 1 - spatial.distance.cosine(co2vec_nlp[id1], co2vec_nlp[id2])
				co2co_nlp[id1][id2] = sc
	else:
		co2co_nlp = {}
		for id1 in co2co_graph:
			co2co_nlp[id1] = {}
			for id2 in co2co_graph[id1]:
				co2co_nlp[id1][id2] = 1.
		co2vec_nlp = {}
		for c in cell_ontology_ids:
			co2vec_nlp[c] = np.ones((10))
	return co2co_graph, co2co_nlp, co2vec_nlp, cell_ontology_ids

def create_unseen_candidates(cell_type_network_file, co2i, i2co, nseen, use_unseen_distance, test_Y_pred_all):
	nct = len(co2i)
	A = np.zeros((nct, nct))
	fin = open(cell_type_network_file)
	for line in fin:
		w = line.strip().split('\t')
		A[co2i[w[0]], co2i[w[1]]] = 1.
		A[co2i[w[1]], co2i[w[0]]] = 1.
	fin.close()
	A_dis = graph_shortest_path(A,method='FW',directed =False)
	min_d = np.min(A_dis[:nseen, nseen:], axis = 0)
	assert(len(min_d) == nct - nseen)
	unseen_cand = np.where(min_d > use_unseen_distance)[0] + nseen
	test_Y_pred_all[:, unseen_cand] = 0
	assert(np.shape(test_Y_pred_all)[1] == nct)
	return test_Y_pred_all



def graph_embedding_dca(A, i2l, mi=0, dim=20,unseen_l=None):
	nl = np.shape(A)[0]
	seen_ind = []
	unseen_ind = []
	for i in range(nl):
		if i2l[i] in unseen_l:
			unseen_ind.append(i)
		else:
			seen_ind.append(i)
	seen_ind = np.array(seen_ind)
	unseen_ind = np.array(unseen_ind)

	#if len(seen_ind) * 0.8 < dim:
	#	dim = int(len(seen_ind) * 0.8)
	if mi==0 or mi == 1:
		sp = graph_shortest_path(A,method='FW',directed =False)
	else:
		sp = RandomWalkRestart(A, 0.8)

	sp = sp[seen_ind, :]
	sp = sp[:,seen_ind]
	X = np.zeros((np.shape(sp)[0],dim))
	svd_dim = min(dim, np.shape(sp)[0]-1)
	if mi==0 or mi == 2:
		X[:,:svd_dim] = svd_emb(sp, dim=svd_dim)
	else:
		X[:,:svd_dim] = DCA_vector(sp, dim=svd_dim)[0]
	X_ret = np.zeros((nl, dim))
	X_ret[seen_ind,:] = X
	if mi==2 or mi == 3:
		sp *= -1
	return sp, X_ret

def DCA_vector(Q, dim):
	nnode = Q.shape[0]
	alpha = 1. / (nnode **2)
	Q = np.log(Q + alpha) - np.log(alpha);

	#Q = Q * Q';
	[U, S, V] = svds(Q, dim);
	S = np.diag(S)
	X = np.dot(U, np.sqrt(S))
	Y = np.dot(np.sqrt(S), V)
	Y = np.transpose(Y)
	return X,U,S,V,Y

def RandomWalkRestart(A, rst_prob, delta = 1e-4, reset=None, max_iter=50,use_torch=False,return_torch=False):
	if use_torch:
		device = torch.device("cuda:0")
	nnode = A.shape[0]
	#print nnode
	if reset is None:
		reset = np.eye(nnode)
	nsample,nnode = reset.shape
	#print nsample,nnode
	P = renorm(A)
	P = P.T
	norm_reset = renorm(reset.T)
	norm_reset = norm_reset.T
	if use_torch:
		norm_reset = torch.from_numpy(norm_reset).float().to(device)
		P = torch.from_numpy(P).float().to(device)
	Q = norm_reset

	for i in range(1,max_iter):
		#Q = gnp.garray(Q)
		#P = gnp.garray(P)
		if use_torch:
			Q_new = rst_prob*norm_reset + (1-rst_prob) * torch.mm(Q, P)#.as_numpy_array()
			delta = torch.norm(Q-Q_new, 2)
		else:
			Q_new = rst_prob*norm_reset + (1-rst_prob) * np.dot(Q, P)#.as_numpy_array()
			delta = np.linalg.norm(Q-Q_new, 'fro')
		Q = Q_new
		#print (i,Q)
		sys.stdout.flush()
		if delta < 1e-4:
			break
	if use_torch and not return_torch:
		Q = Q.cpu().numpy()
	return Q


def renorm(X):
	Y = X.copy()
	Y = Y.astype(float)
	ngene,nsample = Y.shape
	s = np.sum(Y, axis=0)
	#print s.shape()
	for i in range(nsample):
		if s[i]==0:
			s[i] = 1
			if i < ngene:
				Y[i,i] = 1
			else:
				for j in range(ngene):
					Y[j,i] = 1. / ngene
		Y[:,i] = Y[:,i]/s[i]
	return Y

def mean_normalization(train_X_mean, test_X):
	test_X = np.log1p(test_X)
	test_X_mean = np.mean(test_X, axis = 0)
	test_X = test_X - test_X_mean + train_X_mean
	return test_X

def process_expression(train_X, test_X, train_genes, test_genes):
	#this data process function is adapted from ACTINN, please check ACTINN for more information.
	#test_X = map_genes(test_X, test_genes, train_genes)
	c2g = np.vstack([train_X, test_X])
	c2g = np.array(c2g,  dtype=np.float64)
	c2g = c2g.T
	index = np.sum(c2g, axis=1)>0
	c2g = c2g[index, :]
	train_genes = train_genes[index]
	c2g = np.divide(c2g, np.sum(c2g, axis=0, keepdims=True)) * 10000
	c2g = np.log2(c2g+1)
	expr = np.sum(c2g, axis=1)
	#total_set = total_set[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]
	index = np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99))
	c2g = c2g[index,]
	train_genes = train_genes[index]
	#print (np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)))

	cv = np.std(c2g, axis=1) / np.mean(c2g, axis=1)
	index = np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99))
	c2g = c2g[index,]
	train_genes = train_genes[index]
	c2g = c2g.T
	c2g_list_new = []
	index = 0
	for c in [train_X, test_X]:
		ncell = np.shape(c)[0]
		c2g_list_new.append(c2g[index:index+ncell,:])
		index = ncell
		assert (len(train_genes) == np.shape(c2g)[1])
	return c2g_list_new[0], c2g_list_new[1], train_genes

def emb_ontology(i2l, ontology_mat, co2co_nlp, dim=5, mi=0, unseen_l = None):
	nco = len(i2l)
	network = np.zeros((nco, nco))
	for i in range(nco):
		c1 = i2l[i]
		for j in range(nco):
			if ontology_mat[i,j] == 1:
				network[i,j] = co2co_nlp[c1][i2l[j]]
				network[j,i] = co2co_nlp[c1][i2l[j]]
	idd = 0
	sp, i2emb = graph_embedding_dca(network, i2l, mi=mi, dim=dim, unseen_l=unseen_l)
	return i2emb
