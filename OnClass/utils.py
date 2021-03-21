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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,precision_recall_fscore_support, cohen_kappa_score, auc, average_precision_score,f1_score,precision_recall_curve
import time
import umap
import copy
from sklearn import preprocessing
from fbpca import pca
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
#from libs import *
from sklearn.utils.graph_shortest_path import graph_shortest_path
from scipy.sparse.linalg import svds, eigs

nn_nhidden = [1000]
rsts = [0.5,0.6,0.7,0.8]
dfs_depth = 1
co_dim = 5
keep_prob = 1.0
use_diagonal = True
max_iter = 20
niter = 5
def translate_paramter(ps):
	s = []
	for p in ps:
		if isinstance(p, list):
			p = [str(i) for i in p]
			p = '.'.join(p)
			s.append(p)
		else:
			s.append(str(p))
	s = '_'.join(s)
	return s
pname = translate_paramter([max_iter])

def make_folder(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)
	return folder

def create_propagate_networks(dname, l2i, onto_net, cls2cls, ontology_nlp_file, rsts = [0.5,0.6,0.7,0.8], diss=[2,3], thress=[1,0.8]):
	ncls = np.shape(cls2cls)[0]
	if dname != 'allen':
		onto_net_nlp, onto_net_bin, stack_net_nlp, stack_net_bin, onto_net_nlp_all_pairs = create_nlp_networks(l2i, onto_net, cls2cls, ontology_nlp_file)
		#network = create_consensus_networks(rsts, stack_net_nlp, onto_net_nlp_all_pairs, cls2cls)
		network = create_consensus_networks(rsts, stack_net_nlp, onto_net_nlp_all_pairs, cls2cls, diss = diss, thress = thress)
	else:
		stack_net_bin = np.zeros((ncls,ncls))
		for n1 in onto_net:
			for n2 in onto_net[n1]:
				if n1==n2:
					continue
				stack_net_bin[n1,n2] = 1
				stack_net_bin[n2,n1] = 1
		network = [RandomWalkRestart(stack_net_bin, rst) for rst in rsts]
	return network


def fine_nearest_co_using_nlp(sentences,co2emb,obo_file,nlp_mapping_cutoff=0.8):
	co2name, name2co = get_ontology_name(obo_file = obo_file)
	from sentence_transformers import SentenceTransformer
	model = SentenceTransformer('bert-base-nli-mean-tokens')
	sentences = np.array([sentence.lower() for sentence in sentences])
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
		if sc>nlp_mapping_cutoff:
			sent2co[sentence.lower()] = cos[co_id]
			names = set()
			for name in name2co:
				if name2co[name].upper() == cos[co_id]:
					names.add(name)
			#print (sentence, cos[co_id], sc, co2name[cos[co_id]],names)
	return sent2co


def ImputeUnseenCls(y_vec, y_raw, cls2cls, nseen, knn=1):
	nclass = np.shape(cls2cls)[0]
	seen2unseen_sim = cls2cls[:nseen, nseen:]
	nngh = np.argsort(seen2unseen_sim*-1, axis = 0)[0,:]
	ncell = len(y_vec)
	y_mat = np.zeros((ncell, nclass))
	y_mat[:,:nseen] = y_raw[:, :nseen]
	for i in range(ncell):
		if y_vec[i] == -1:
			#kngh = np.argsort(y_raw[i,:nseen]*-1)[0:knn]
			#if len(kngh) == 0:
			#	continue
			y_mat[i,nseen:] = y_mat[i,nngh]
			y_mat[i,:nseen] -= 1000000
	return y_mat


def ImputeUnseenCls_Backup(y_vec, y_raw, cls2cls, nseen, knn=1):
	nclass = np.shape(cls2cls)[0]
	seen2unseen_sim = cls2cls[:nseen, nseen:]
	ncell = len(y_vec)
	y_mat = np.zeros((ncell, nclass))
	y_mat[:,:nseen] = y_raw[:, :nseen]
	for i in range(ncell):
		if y_vec[i] == -1:
			kngh = np.argsort(y_raw[i,:nseen]*-1)[0:knn]
			if len(kngh) == 0:
				continue
			y_mat[i,:nseen] -= 1000000
			y_mat[i,nseen:] = np.dot(y_raw[i,kngh], seen2unseen_sim[kngh,:])
	return y_mat

def find_gene_ind(genes, common_genes):
	gid = []
	for g in common_genes:
		gid.append(np.where(genes == g)[0][0])
	gid = np.array(gid)
	return gid

def RandomWalkOntology(onto_net, l2i, ontology_nlp_file, ontology_nlp_emb_file = '../OnClass_data/cell_ontology/cl.ontology.nlp.emb', rst = 0.7):
	ncls = len(l2i)
	onto_net_nlp, _, onto_nlp_emb = read_cell_ontology_nlp(l2i, ontology_nlp_file = ontology_nlp_file, ontology_nlp_emb_file = '../OnClass_data/cell_ontology/cl.ontology.nlp.emb')
	onto_net_nlp = (cosine_similarity(onto_nlp_emb) + 1 ) /2#1 - spatial.distance.cosine(onto_nlp_emb, onto_nlp_emb)
	onto_net_mat = np.zeros((ncls, ncls))
	for n1 in onto_net:
		for n2 in onto_net[n1]:
			if n1==n2:
				continue
			onto_net_mat[n1,n2] = onto_net_nlp[n1, n2]
			onto_net_mat[n2,n1] = onto_net_nlp[n2, n1]
	onto_net_rwr = RandomWalkRestart(onto_net_mat, rst)
	return onto_net_rwr

def process_expression(c2g_list):
	#this data process function is motivated by ACTINN, please check ACTINN for more information.
	c2g = np.vstack(c2g_list)
	c2g = c2g.T
	#print ('onclass d0',np.shape(c2g))
	c2g = c2g[np.sum(c2g, axis=1)>0, :]
	#print (c2g)
	#print ('onclass d1',np.shape(c2g))
	c2g = np.divide(c2g, np.sum(c2g, axis=0, keepdims=True)) * 10000
	c2g = np.log2(c2g+1)
	expr = np.sum(c2g, axis=1)
	#total_set = total_set[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]

	c2g = c2g[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]
	#print (c2g)
	#print ('onclass d2',np.shape(c2g))
	cv = np.std(c2g, axis=1) / np.mean(c2g, axis=1)
	c2g = c2g[np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99)),]
	#print (c2g)
	#print ('onclass d3',np.shape(c2g))
	c2g = c2g.T
	#print (c2g)
	#print ('onclass d4',np.shape(c2g))
	c2g_list_new = []
	index = 0
	for c in c2g_list:
		ncell = np.shape(c)[0]
		c2g_list_new.append(c2g[index:index+ncell,:])
		index = ncell
	return c2g_list_new

def read_ontology_file(dname, data_folder):
	if 'allen' in dname:
		cell_type_network_file = data_folder + '/cell_ontology/allen.ontology'
		cell_type_nlp_emb_file = None
		cl_obo_file = None
	else:
		cell_type_network_file = data_folder + '/cell_ontology/cl.ontology'
		cell_type_nlp_emb_file = data_folder + '/cell_ontology/cl.ontology.nlp.emb'
		cl_obo_file = data_folder + '/cell_ontology/cl.obo'
	return cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file

def read_data_file(dname, data_dir):

	if 'microcebus' in dname:
		tech = '10x'
		feature_file = data_dir + 'TMS_official_060520/' + dname +'.h5ad'
		filter_key={'method':tech }
		label_file = None
		gene_file = ''
		label_key = 'cell_ontology_class'
	elif 'muris' in dname:
		tech = dname.split('_')[1]
		feature_file = data_dir + 'TMS_official_060520/' + 'tabula-muris-senis-'+tech+'-official-raw-obj.h5ad'
		filter_key = {}
		label_file = None
		gene_file = ''
		batch_key = ''
		label_key = 'cell_ontology_class'
	elif 'sapiens' in dname:
		feature_file = data_dir + 'sapiens/' + 'Pilot1_Pilot2_decontX_Oct2020.h5ad'
		filter_key = {}
		label_file = None
		gene_file = ''
		batch_key = ''
		label_key = 'cell_ontology_type'
	elif 'allen' in dname:
		feature_file = data_dir + '/Allen/features.pkl'
		label_file = data_dir + '/Allen/labels.pkl'
		gene_file = data_dir + '/Allen/genes.pkl'
		label_key = ''
		filter_key = {}
	elif 'krasnow' in dname:
		tech = dname.split('_')[1]
		feature_file = data_dir + 'Krasnow/'+tech+'_features.pkl'
		label_file = data_dir + 'Krasnow/'+tech+'_labels.pkl'
		gene_file = data_dir + 'Krasnow/'+tech+'_genes.pkl'
		label_key = ''
		filter_key = {}
	else:
		sys.exit('wrong dname '+dname)
	if feature_file.endswith('.pkl'):
		return feature_file, filter_key, label_key, label_file, gene_file
	elif feature_file.endswith('.h5ad'):
		return feature_file, filter_key, label_key, label_file, gene_file
	sys.exit('wrong file suffix')

def read_singlecell_data(dname, data_dir, nsample = 500000000, read_tissue = False, exclude_non_leaf_ontology = True):
	if 'microcebus' in dname:
		tech = '10x'
		#file = data_dir + 'TMS_official_060520/' + 'tabula-microcebus_smartseq2-10x_combined_annotated_filtered_gene-labels-correct.h5ad'
		file = data_dir + 'TMS_official_060520/' + dname +'.h5ad'
		filter_key={'method':tech }
		batch_key = ''#original_channel
		ontology_nlp_file = '../OnClass_data/cell_ontology/cl.ontology.nlp'
		ontology_file = '../OnClass_data/cell_ontology/cl.ontology'
		if not read_tissue:
			feature, label, genes = parse_h5ad(file, nsample = nsample, read_tissue = read_tissue, label_key='cell_ontology_class', batch_key = batch_key, filter_key = filter_key, cell_ontology_file = ontology_file, exclude_non_leaf_ontology = exclude_non_leaf_ontology, exclude_non_ontology = True, DATA_DIR = '../OnClass_data/')
		else:
			feature, label, genes, tissues = parse_h5ad(file, nsample = nsample, read_tissue = read_tissue, label_key='cell_ontology_class', batch_key = batch_key, filter_key = filter_key, cell_ontology_file = ontology_file, exclude_non_leaf_ontology = exclude_non_leaf_ontology, exclude_non_ontology = True, DATA_DIR = '../OnClass_data/')
	elif 'muris' in dname:
		tech = dname.split('_')[1]
		file = data_dir + 'TMS_official_060520/' + 'tabula-muris-senis-'+tech+'-official-raw-obj.h5ad'
		filter_key = {}
		batch_key = ''
		ontology_nlp_file = '../OnClass_data/cell_ontology/cl.ontology.nlp'
		ontology_file = '../OnClass_data/cell_ontology/cl.ontology'
		if not read_tissue:
			feature, label, genes = parse_h5ad(file,  nsample = nsample, read_tissue = read_tissue, label_key='cell_ontology_class', batch_key = batch_key, cell_ontology_file = ontology_file, filter_key=filter_key, exclude_non_leaf_ontology = exclude_non_leaf_ontology, exclude_non_ontology = True, DATA_DIR = '../OnClass_data/')
		else:
			feature, label, genes, tissues = parse_h5ad(file, nsample = nsample, read_tissue = read_tissue, label_key='cell_ontology_class', batch_key = batch_key, cell_ontology_file = ontology_file, filter_key=filter_key, exclude_non_leaf_ontology = exclude_non_leaf_ontology, exclude_non_ontology = True, DATA_DIR = '../OnClass_data/')
	elif 'allen_part' in dname:
		feature_file = data_dir + 'Allen/matrix_part.csv'
		label_file = data_dir + 'Allen/metadata.csv'
		ontology_file = data_dir + 'Allen/cell_type_ontology'
		ontology_nlp_file = None
		feature, label, genes = parse_csv(feature_file, label_file, nsample = nsample, label_key='cell_type_accession_label', exclude_non_ontology = True, exclude_non_leaf_ontology = True, cell_ontology_file=ontology_file)
	elif 'allen' in dname:
		feature_file = data_dir + 'Allen/features.pkl'
		label_file = data_dir + 'Allen/labels.pkl'
		gene_file = data_dir + 'Allen/genes.pkl'
		ontology_file = data_dir + 'Allen/cell_type_ontology'
		ontology_nlp_file = None
		feature, label, genes = parse_pkl(feature_file, label_file, gene_file, nsample = nsample, exclude_non_leaf_ontology = True, cell_ontology_file=ontology_file)
	elif 'krasnow' in dname:
		tech = dname.split('_')[1]
		feature_file = data_dir + 'Krasnow/'+tech+'_features.pkl'
		label_file = data_dir + 'Krasnow/'+tech+'_labels.pkl'
		gene_file = data_dir + 'Krasnow/'+tech+'_genes.pkl'
		ontology_file = '../OnClass_data/cell_ontology/cl.ontology'
		ontology_nlp_file = '../OnClass_data/cell_ontology/cl.ontology.nlp'
		feature, label, genes = parse_pkl(feature_file, label_file, gene_file, nsample = nsample,  exclude_non_leaf_ontology = True, cell_ontology_file=ontology_file)
	else:
		sys.exit('wrong dname '+dname)
	if read_tissue:
		return feature, label, genes, tissues, ontology_nlp_file, ontology_file
	else:
		return feature, label, genes, ontology_nlp_file, ontology_file



def parse_krasnow(feature_file, label_file, gene_file, seed = 1, nsample = 1000,exclude_non_leaf_ontology = True, exclude_non_ontology = True, cell_ontology_file=None):
	np.random.seed(seed)

	if feature_file.endswith('.pkl'):
		features = pickle.load(open(feature_file, 'rb'))
		labels = pickle.load(open(label_file, 'rb'))
		genes = pickle.load(open(gene_file, 'rb'))
		ncell, ngene = np.shape(features)
		assert(ncell == len(labels))
		assert(ngene == len(genes))
		index = np.random.choice(ncell,min(nsample,ncell),replace=False)
		features = features[index, :]
		labels = labels[index]
	if exclude_non_leaf_ontology:
		new_ids, exclude_terms = exclude_parent_child_nodes(cell_ontology_file, labels)
		#print (len(exclude_terms),'non leaf terms are excluded')
		features = features[new_ids, :]
		labels = labels[new_ids]
	genes = [x.upper() for x in genes]
	genes = np.array(genes)
	return features, labels, genes

def parse_pkl(feature_file, label_file, gene_file, seed = 1, nsample = 10000000,exclude_non_leaf_ontology = True, cell_ontology_file=None):
	np.random.seed(seed)
	if feature_file.endswith('.pkl'):
		features = pickle.load(open(feature_file, 'rb'))
		labels = pickle.load(open(label_file, 'rb'))
		genes = pickle.load(open(gene_file, 'rb'))
		ncell, ngene = np.shape(features)
		assert(ncell == len(labels))
		assert(ngene == len(genes))
		index = np.random.choice(ncell,ncell,replace=False)
		features = features[index, :]
		labels = labels[index]
	if exclude_non_leaf_ontology:
		new_ids, exclude_terms = exclude_parent_child_nodes(cell_ontology_file, labels)
		#print (len(exclude_terms),'non leaf terms are excluded')
		features = features[new_ids, :]
		labels = labels[new_ids]
	genes = [x.upper() for x in genes]
	genes = np.array(genes)
	return features, labels, genes

def select_high_var_genes(train_X, test_X, ngene = 200):
	mat = np.vstack((train_X, test_X))
	#mat = mat.todense()
	gstd = np.std(mat, axis=0)
	best_genes = np.argsort(gstd*-1)
	best_genes = best_genes[:ngene]
	return train_X[:, best_genes], test_X[:, best_genes]

def emb_cells(train_X, test_X, dim=20):
	if dim==-1:
		return np.log1p(train_X.todense()), np.log1p(test_X.todense())
	train_X = np.log1p(train_X)
	test_X = np.log1p(test_X)
	train_X = preprocessing.normalize(train_X, axis=1)
	test_X = preprocessing.normalize(test_X, axis=1)
	ntrain = np.shape(train_X)[0]
	mat = sparse.vstack((train_X, test_X))
	U, s, Vt = pca(mat, k=dim) # Automatically centers.
	X = U[:, range(dim)] * s[range(dim)]
	return X[:ntrain,:], X[ntrain:,:]

def write_markers(fname, markers):
	## Write marker genes to file
	fmarker_genes = open(fname,'w')
	for t in markers:
		fmarker_genes.write(t+'\t')
		g2pv = sorted(markers[t].items(), key=lambda item: item[1])
		for g,pv in g2pv:
			fmarker_genes.write(g+'(pv:'+'{:.2e}'.format(pv)+')\t')
		fmarker_genes.write('\n')
	fmarker_genes.close()


def calculate_markers(cell2term, cell2gene, genes, terms, topk_cells=500, only_over_expressed = True, return_k_genes = 100):
	ncell, nterm = np.shape(cell2term)
	ngene = np.shape(cell2gene)[1]
	assert(ncell == np.shape(cell2gene)[0])
	markers = collections.defaultdict(dict)
	for t in range(nterm):
		scs = np.argsort(cell2term[:,t])
		k_bot_cells = scs[:topk_cells]
		k_top_cells = scs[ncell-topk_cells:]
		pv = scipy.stats.ttest_ind(cell2gene[k_top_cells,:], cell2gene[k_bot_cells,:], axis=0)[1] #* ngene
		top_mean = np.mean(cell2gene[k_top_cells,:],axis=0)
		bot_mean = np.mean(cell2gene[k_bot_cells,:],axis=0)
		if only_over_expressed:
			for g in range(ngene):
				if top_mean[g] < bot_mean[g]:
					pv[g] = 1.
		pv_sort = list(np.argsort(pv))
		#for i in range(return_k_genes):
		#markers[terms[t]][genes[pv_sort[i]]] = pv[pv_sort[i]]
		markers[terms[t]] = pv
		for i,p in enumerate(pv):
			if np.isnan(p):
				pv[i] = 1.
			#markers[terms[t]][str(pv_sort[i])] = pv[pv_sort[i]]
	return markers

def peak_h5ad(file):
	'''
	peak the number of cells, classes, genes in h5ad file
	'''
	x = read_h5ad(file)
	#print (np.shape(x.X))
	#print (x.X[:10][:10])
	#print (x.obs.keys())
	ncell, ngene = np.shape(x.X)
	nclass = len(np.unique(x.obs['free_annotation']))
	#print (np.unique(x.obs['free_annotation']))
	f2name = {}
	sel_cell = 0.
	for i in range(ncell):
		if x.obs['method'][i]!='10x':
			continue

		free = x.obs['free_annotation'][i]
		name = x.obs['cell_ontology_class'][i]
		f2name[free] = name
		sel_cell += 1
	#return f2name
	#for key in x.obs.keys():
	#	print (key, np.unique(x.obs[key]))
	return sel_cell, ngene, nclass
	#for i in range(10):
	#	print (x.obs['method'][i], x.obs['channel_no_10x'][i])
	#for key in x.obs.keys():
	#	print (key, np.unique(x.obs[key]))
	#return index


def get_onotlogy_parents(GO_net, g):
	term_valid = set()
	ngh_GO = set()
	ngh_GO.add(g)
	while len(ngh_GO) > 0:
		for GO in list(ngh_GO):
			for GO1 in GO_net[GO]:
				ngh_GO.add(GO1)
			ngh_GO.remove(GO)
			term_valid.add(GO)
	return term_valid

def exclude_non_ontology_term(labels, label_key, DATA_DIR = '../../OnClass_data/'):
	co2name, name2co = get_ontology_name()
	new_labs = []
	new_ids = []
	if label_key!='cell_ontology_class' and label_key!='cell_ontology_id':
		use_co = False
		for kk in np.unique(labels):
			if kk.lower().startswith('cl:'):
				use_co = True
				break
	else:
		if label_key == 'cell_ontology_class':
			use_co = False
		else:
			use_co = True
	for i in range(len(labels)):
		l = labels[i]
		if not use_co:
			if l.lower() in name2co.keys():
				new_labs.append(name2co[l.lower()])
				new_ids.append(i)
		else:
			if l.lower() in co2name.keys():
				new_labs.append(l.lower())
				new_ids.append(i)
	new_labs = np.array(new_labs)
	new_ids = np.array(new_ids)
	return new_ids, new_labs


def parse_raw_h5ad(file,seed=1,nsample=1e10,tissue_key='tissue',label_key='cell_ontology_class', read_tissue = True, batch_key = '', filter_key={}, cell_ontology_file = None, exclude_non_leaf_ontology = True, exclude_non_ontology=True, DATA_DIR = '../../OnClass_data/'):
	np.random.seed(seed)
	x = read_h5ad(file)

	ncell = np.shape(x.raw.X)[0]
	select_cells = set(range(ncell))
	for key in filter_key:
		value = filter_key[key]
		select_cells = select_cells & set(np.where(np.array(x.obs[key])==value)[0])
	select_cells = sorted(select_cells)
	feature = x.raw.X[select_cells, :]
	labels = np.array(x.obs[label_key].tolist())[select_cells]
	if read_tissue:
		tissues = np.array(x.obs[tissue_key].tolist())[select_cells]
	if batch_key=='' or batch_key not in x.obs.keys():
		batch_labels = np.ones(len(labels))
	else:
		batch_labels = np.array(x.obs[batch_key].tolist())[select_cells]
	genes = x.var.index
	ncell = len(select_cells)
	if exclude_non_ontology:
		new_ids, labels = exclude_non_ontology_term(labels, label_key, DATA_DIR = DATA_DIR)
		feature = feature[new_ids, :]
		batch_labels = batch_labels[new_ids]
	if exclude_non_leaf_ontology:
		new_ids, exclude_terms = exclude_parent_child_nodes(cell_ontology_file, labels)
		#print (len(exclude_terms),'non leaf terms are excluded')
		feature = feature[new_ids, :]
		batch_labels = batch_labels[new_ids]
		labels = labels[new_ids]
		if read_tissue:
			tissues = tissues[new_ids]
	ncell = len(labels)
	index = np.random.choice(ncell,min(nsample,ncell),replace=False)
	batch_labels = batch_labels[index]
	feature = feature[index, :] # cell by gene matrix
	labels = labels[index]
	if read_tissue:
		tissues = tissues[index]
	genes = x.var.index
	corrected_feature = run_scanorama_same_genes(feature, batch_labels)
	corrected_feature = corrected_feature.toarray()
	genes = [x.upper() for x in genes]
	genes = np.array(genes)
	if read_tissue:
		assert(len(tissues) == len(labels))
		return corrected_feature, labels, genes, tissues
	else:
		return corrected_feature, labels, genes

def select_cells_based_on_keys(x, features, tissues = None, labels = None, filter_key = None):
	ncell = np.shape(x.X)[0]
	select_cells = set(range(ncell))
	for key in filter_key:
		value = filter_key[key]
		select_cells = select_cells & set(np.where(np.array(x.obs[key])==value)[0])
	select_cells = sorted(select_cells)
	features = features[select_cells,: ]
	if labels is not None:
		labels = labels[select_cells]
	if tissues is not None:
		tissues = tissues[select_cells]
	x = x[select_cells,:]
	return features, labels, tissues, x

def find_marker_genes(train_X, pred_Y_all, genes, i2l, topk = 50):
	cor = corr2_coeff(pred_Y_all[:,:].T, train_X[:,:].T)
	cor = np.nan_to_num(cor) # cell type to gene
	nl = len(i2l)
	c2g = {}
	for i in range(nl):
		gl = np.argsort(cor[i,:]*-1)
		c2g[i2l[i]] = {}
		for j in range(topk):
			c2g[i2l[i]][genes[gl[j]]] = cor[i, gl[j]]
	return c2g, cor


def use_pretrained_model(OnClass, genes, test_X, models = []):
	last_l2i = {}
	last_i2l = {}

	pred_Y_all_models = 0.
	ngene = len(genes)
	for model in models:
		OnClass.BuildModel(OnClass.co2emb, ngene = ngene, use_pretrain = model)
		print ('Build model finished for ',model)
		pred_Y_seen, pred_Y_all, pred_label = OnClass.Predict(test_X, test_genes = genes)
		print ('Predict for ',model)
		pred_Y_all = pred_Y_all.T / (pred_Y_all.T.sum(axis=1)[:, np.newaxis] + 1)
		pred_Y_all = pred_Y_all.T
		if len(last_l2i)>0:
			new_ct_ind = []
			for i in range(len(last_i2l)):
				l = last_i2l[i]
				new_ct_ind.append(OnClass.co2i[l])
			pred_Y_all = pred_Y_all[:, np.array(new_ct_ind)]
			pred_Y_all_models += pred_Y_all
		else:
			last_l2i = OnClass.co2i
			last_i2l = OnClass.i2co
			pred_Y_all_models = pred_Y_all
	return pred_Y_all_models


def read_data(feature_file, cell_ontology_ids, exclude_non_leaf_ontology = False, ct_mapping_key = {}, tissue_key = None, seed = 1, filter_key = None, AnnData_label_key=None, nlp_mapping = True, nlp_mapping_cutoff = 0.8, co2emb = None, label_file=None, cl_obo_file = None, cell_ontology_file = None):
	np.random.seed(seed)
	x = read_h5ad(feature_file)
	ncell = np.shape(x.X)[0]
	dataset = x.X.toarray()
	genes = np.array([x.upper() for x in x.var.index])

	if tissue_key is not None:
		tissues = np.array(x.obs['tissue'].tolist())
	else:
		tissues = None
	if AnnData_label_key is None and label_file is None:
		print ('no label file is provided')
		labels = None
		dataset, labels, tissues, x = select_cells_based_on_keys(x, dataset, labels = labels, tissues = tissues, filter_key = filter_key)
		return dataset, genes, labels, tissues, x
	if AnnData_label_key is not None:
		labels = x.obs[AnnData_label_key].tolist()
	else:
		fin = open(label_file)
		labels = []
		for line in fin:
			labels.append(line.strip())
		fin.close()
	labels = np.array(labels)
	dataset, labels, tissues, x = select_cells_based_on_keys(x, dataset, labels = labels, tissues = tissues, filter_key = filter_key)
	ind, labels, unfound_labs = map_and_select_labels(labels, cell_ontology_ids, cl_obo_file, ct_mapping_key = ct_mapping_key, nlp_mapping = nlp_mapping, co2emb = co2emb, nlp_mapping_cutoff = nlp_mapping_cutoff, cl_obo_file = cl_obo_file)
	if tissue_key is not None:
		tissues = tissues[ind]
	dataset = dataset[ind, :]
	x = x[ind, :]
	if exclude_non_leaf_ontology:
		new_ids, exclude_terms = exclude_parent_child_nodes(cell_ontology_file, labels)
		tissues = tissues[new_ids]
		dataset = dataset[new_ids, :]
		labels = labels[new_ids]
		x = x[new_ids, :]

	ncell = np.shape(dataset)[0]
	index = np.random.choice(ncell,ncell,replace=False)
	dataset = dataset[index, :] # cell by gene matrix
	labels = labels[index]
	if tissue_key is not None:
		tissues = tissues[index]
	return dataset, genes, labels, tissues, x

def exact_match_co_name_2_co_id(labels, lab2co, cl_obo_file = None):
	if cl_obo_file is None:
		return lab2co
	co2name, name2co = get_ontology_name(obo_file = cl_obo_file)
	for label in labels:
		if label.lower() in name2co:
			lab2co[label.lower()] = name2co[label.lower()]
	for name in name2co:
		lab2co[name.lower()] = name2co[name]
	return lab2co


def map_and_select_labels(labels, cell_ontology_ids, obo_file, ct_mapping_key = {}, nlp_mapping = True, nlp_mapping_cutoff = 0.8, co2emb = None, cl_obo_file = None):
	lab2co = {}
	if nlp_mapping:
		if co2emb is None:
			sys.exit('Please provide cell type embedding to do NLP-based mapping.')
		lab2co = fine_nearest_co_using_nlp(np.unique(labels), co2emb, obo_file,nlp_mapping_cutoff = nlp_mapping_cutoff)
	lab2co = exact_match_co_name_2_co_id(np.unique(labels), lab2co, cl_obo_file = cl_obo_file)
	for ct in ct_mapping_key:
		lab2co[ct_mapping_key[ct]] = lab2co[ct]
	ind = []
	lab_id = []
	unfound_labs = set()
	for i,l in enumerate(labels):
		if l in cell_ontology_ids:
			ind.append(i)
			lab_id.append(l)
		elif l.lower() in lab2co:
			ind.append(i)
			lab_id.append(lab2co[l.lower()])
		else:
			unfound_labs.add(l)
	frac = len(ind) * 1. / len(labels)
	ind = np.array(ind)
	labels = np.array(lab_id)
	unfound_labs = set(unfound_labs)
	warn_message = 'Warning: Only: %f precentage of labels are in the Cell Ontology. The remaining cells are excluded! Consider using NLP mapping and choose a small mapping cutoff (nlp_mapping_cutoff)' % (frac * 100)

	print (warn_message)
	print ('Here are unfound labels:',unfound_labs)
	return ind, labels, unfound_labs

def parse_h5ad(file,seed=1,nsample=1e10,label_key='cell_ontology_class', read_tissue = False, batch_key = '', filter_key={}, cell_ontology_file = None, exclude_non_leaf_ontology = True, exclude_non_ontology=True, DATA_DIR = '../../OnClass_data/'):
	'''
	read h5ad file
	feature: cell by gene expression
	label: cell ontology class
	genes: gene names HGNC
	'''
	np.random.seed(seed)
	x = read_h5ad(file)
	ncell = np.shape(x.X)[0]
	select_cells = set(range(ncell))
	for key in filter_key:
		value = filter_key[key]
		select_cells = select_cells & set(np.where(np.array(x.obs[key])==value)[0])
	select_cells = sorted(select_cells)
	feature = x.X[select_cells, :]
	labels = np.array(x.obs[label_key].tolist())[select_cells]
	if read_tissue:
		tissues = np.array(x.obs['tissue'].tolist())[select_cells]
	if batch_key=='' or batch_key not in x.obs.keys():
		batch_labels = np.ones(len(labels))
	else:
		batch_labels = np.array(x.obs[batch_key].tolist())[select_cells]
	genes = x.var.index
	ncell = len(select_cells)

	if exclude_non_ontology:
		new_ids, labels = exclude_non_ontology_term(labels, label_key, DATA_DIR = DATA_DIR)
		feature = feature[new_ids, :]
		batch_labels = batch_labels[new_ids]
	if exclude_non_leaf_ontology:
		new_ids, exclude_terms = exclude_parent_child_nodes(cell_ontology_file, labels)
		#print (len(exclude_terms),'non leaf terms are excluded')
		feature = feature[new_ids, :]
		batch_labels = batch_labels[new_ids]
		labels = labels[new_ids]
		if read_tissue:
			tissues = tissues[new_ids]
	ncell = len(labels)
	index = np.random.choice(ncell,min(nsample,ncell),replace=False)
	batch_labels = batch_labels[index]
	feature = feature[index, :] # cell by gene matrix
	labels = labels[index]
	if read_tissue:
		tissues = tissues[index]
	genes = x.var.index
	corrected_feature = run_scanorama_same_genes(feature, batch_labels)
	corrected_feature = corrected_feature.toarray()
	genes = [x.upper() for x in genes]
	genes = np.array(genes)
	if read_tissue:
		assert(len(tissues) == len(labels))
		return corrected_feature, labels, genes, tissues
	else:
		return corrected_feature, labels, genes


def exclude_parent_child_nodes(cell_ontology_file,labels):
	uniq_labels = np.unique(labels)
	excludes = set()
	net = collections.defaultdict(dict)
	fin = open(cell_ontology_file)
	for line in fin:
		s,p = line.strip().split('\t')
		net[s][p] = 1 #p is parent
	fin.close()
	for n in list(net.keys()):
		ngh = get_ontology_parents(net, n)
		for n1 in ngh:
			net[n][n1] = 1
	for l1 in uniq_labels:
		for l2 in uniq_labels:
			if l1 in net[l2] and l1!=l2: #l1 is l2 parent
				excludes.add(l1)
	#print (excludes)
	new_ids = []
	for i in range(len(labels)):
		if labels[i] not in excludes:
			new_ids.append(i)
	new_ids = np.array(new_ids)
	return new_ids, excludes

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def extract_data_based_on_class(feats, labels, sel_labels):
	ind = []
	for l in sel_labels:
		id = np.where(labels == l)[0]
		ind.extend(id)
	np.random.shuffle(ind)
	X = feats[ind,:]
	Y = labels[ind]
	print (np.shape(X), np.shape(Y))
	return X, Y, ind

def SplitTrainTest(all_X, all_Y, all_tissues = None, random_state=10, nfold_cls = 0.3, nfold_sample = 0.2, nmin_size=10):
	np.random.seed(random_state)

	cls = np.unique(all_Y)
	cls2ct = Counter(all_Y)
	ncls = len(cls)
	test_cls = list(np.random.choice(cls, int(ncls * nfold_cls), replace=False))
	for c in cls2ct:
		if cls2ct[c] < nmin_size:
			test_cls.append(c)
	test_cls = np.unique(test_cls)
	#add rare class to test, since they cannot be split into train and test by using train_test_split(stratify=True)
	train_cls =  [x for x in cls if x not in test_cls]
	train_cls = np.array(train_cls)
	train_X, train_Y, train_ind = extract_data_based_on_class(all_X, all_Y, train_cls)
	test_X, test_Y, test_ind = extract_data_based_on_class(all_X, all_Y, test_cls)
	if all_tissues is not None:
		train_tissues = all_tissues[train_ind]
		test_tissues = all_tissues[test_ind]
		train_X_train, train_X_test, train_Y_train, train_Y_test, train_tissues_train, train_tissues_test = train_test_split(
	 	train_X, train_Y, train_tissues, test_size=nfold_sample, stratify = train_Y,random_state=random_state)
		test_tissues = np.concatenate((test_tissues, train_tissues_test))
		train_tissues = train_tissues_train
	else:
		train_X_train, train_X_test, train_Y_train, train_Y_test = train_test_split(
	 	train_X, train_Y, test_size=nfold_sample, stratify = train_Y,random_state=random_state)
	test_X = np.vstack((test_X, train_X_test))
	test_Y = np.concatenate((test_Y, train_Y_test))
	train_X = train_X_train
	train_Y = train_Y_train
	if all_tissues is not None:
		return train_X, train_Y, train_tissues, test_X, test_Y, test_tissues
	else:
		return train_X, train_Y, test_X, test_Y

'''
def SplitTrainTest(all_X, all_Y, all_tissues = None, random_state=10, nfold_cls = 0.3, nfold_sample = 0.2, nmin_size=10):
	np.random.seed(random_state)

	cls = np.unique(all_Y)
	cls2ct = Counter(all_Y)
	ncls = len(cls)
	rare_cls = []
	not_rare_cls = []
	for c in cls2ct:
		if cls2ct[c] < 2:
			continue
		elif cls2ct[c] < nmin_size:
			rare_cls.append(c)
		else:
			not_rare_cls.append(c)
	cls = np.concatenate((rare_cls, not_rare_cls))
	ncls = len(cls)
	rare_cls = np.array(rare_cls)
	not_rare_cls = np.array(not_rare_cls)
	train_non_rare_cls = list(np.random.choice(not_rare_cls, int(len(not_rare_cls) * (1 - nfold_cls)), replace=False))
	train_cls = np.concatenate((train_non_rare_cls, rare_cls))
	test_cls =  [x for x in cls if x not in train_cls]
	test_cls = np.array(test_cls)
	assert(len(test_cls) + len(train_cls) == ncls)
	assert(len(set(test_cls) & set(train_cls)) == 0)
	#add rare class to test, since they cannot be split into train and test by using train_test_split(stratify=True)
	train_X, train_Y, train_ind = extract_data_based_on_class(all_X, all_Y, train_cls)
	test_X, test_Y, test_ind = extract_data_based_on_class(all_X, all_Y, test_cls)

	if all_tissues is not None:
		train_tissues = all_tissues[train_ind]
		test_tissues = all_tissues[test_ind]
		train_X_train, train_X_test, train_Y_train, train_Y_test, train_tissues_train, train_tissues_test = train_test_split(
	 	train_X, train_Y, train_tissues, test_size=nfold_sample, stratify = train_Y,random_state=random_state)
		test_tissues = np.concatenate((test_tissues, train_tissues_test))
		train_tissues = train_tissues_train
	else:
		train_X_train, train_X_test, train_Y_train, train_Y_test = train_test_split(
	 	train_X, train_Y, test_size=nfold_sample, stratify = train_Y,random_state=random_state)
	test_X = np.vstack((test_X, train_X_test))
	test_Y = np.concatenate((test_Y, train_Y_test))
	train_X = train_X_train
	train_Y = train_Y_train
	if all_tissues is not None:
		return train_X, train_Y, train_tissues, test_X, test_Y, test_tissues
	else:
		return train_X, train_Y, test_X, test_Y
'''

def LeaveOneOutTrainTest(all_X, all_Y, test_Y, all_tissues = None, random_state=10, nfold_sample = 0.2, nmin_size=10):
	np.random.seed(random_state)

	cls = np.unique(all_Y)
	cls2ct = Counter(all_Y)
	ncls = len(cls)
	test_cls = [test_Y]
	test_cls = np.unique(test_cls)
	#add rare class to test, since they cannot be split into train and test by using train_test_split(stratify=True)
	train_cls =  [x for x in cls if x not in test_cls]
	train_cls = np.array(train_cls)
	train_X, train_Y, train_ind = extract_data_based_on_class(all_X, all_Y, train_cls)
	test_X, test_Y, test_ind = extract_data_based_on_class(all_X, all_Y, test_cls)
	if all_tissues is not None:
		train_tissues = all_tissues[train_ind]
		test_tissues = all_tissues[test_ind]
		train_X_train, train_X_test, train_Y_train, train_Y_test, train_tissues_train, train_tissues_test = train_test_split(
	 	train_X, train_Y, train_tissues, test_size=nfold_sample, stratify = train_Y,random_state=random_state)
		test_tissues = np.concatenate((test_tissues, train_tissues_test))
		train_tissues = train_tissues_train
	else:
		train_X_train, train_X_test, train_Y_train, train_Y_test = train_test_split(
	 	train_X, train_Y, test_size=nfold_sample, stratify = train_Y,random_state=random_state)
	test_X = np.vstack((test_X, train_X_test))
	test_Y = np.concatenate((test_Y, train_Y_test))
	train_X = train_X_train
	train_Y = train_Y_train
	if all_tissues is not None:
		return train_X, train_Y, train_tissues, test_X, test_Y, test_tissues
	else:
		return train_X, train_Y, test_X, test_Y

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

def read_cell_ontology_nlp(l2i, ontology_nlp_file = '../../OnClass_data/cell_ontology/cl.ontology.nlp', ontology_nlp_emb_file = '../../OnClass_data/cell_ontology/cl.ontology.nlp.emb'):
	ncls = len(l2i)
	net = np.zeros((ncls, ncls))
	bin_net = np.zeros((ncls, ncls))
	fin = open(ontology_nlp_file)
	for line in fin:
		s,p,wt = line.upper().strip().split('\t')
		wt = float(wt)
		net[l2i[s], l2i[p]] = np.exp(wt)
		net[l2i[p], l2i[s]] = np.exp(wt)
		bin_net[l2i[s], l2i[p]] = 1
		bin_net[l2i[p], l2i[s]] = 1
	fin.close()

	l2vec = {}
	fin = open(ontology_nlp_emb_file)
	for line in fin:
		w = line.upper().strip().split('\t')
		l2vec[w[0]] = []
		dim = len(w)-1
		for i in range(1,len(w)):
			l2vec[w[0]].append(float(w[i]))
	fin.close()

	l2vec_mat = np.zeros((ncls, dim))
	for l in l2vec:
		if l.upper() not in l2i:
			continue
		l2vec_mat[l2i[l.upper()],:] = l2vec[l]

	'''
	net_sum = np.sum(net,axis=0)
	for i in range(ncls):
		if net_sum[i] == 0:
			net[i,i] = 1.
		net[:,i] /= np.sum(net[:,i])
	#net = net / net.sum(axis=1)[:, np.newaxis]
	'''
	return net, bin_net, l2vec_mat


def GetReverseNet(onto_net):
	onto_net_rev = collections.defaultdict(dict)
	for a in onto_net:
		for b in onto_net[a]:
			onto_net_rev[b][a] = 1
	return onto_net_rev


def ParseCLOnto(train_Y, co_dim=5, co_mi=3, dfs_depth = 1, combine_unseen = False,  add_emb_diagonal = True, use_pretrain = None, ontology_nlp_file = '../../OnClass_data/cell_ontology/cl.ontology.nlp', ontology_file = '../../OnClass_data/cell_ontology/cl.ontology', use_seen_only = True):#
	unseen_l, l2i, i2l, train_X2Y, onto_net, onto_net_mat = create_labels(train_Y, dfs_depth = dfs_depth, combine_unseen = combine_unseen, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
	Y_emb = emb_ontology(i2l, dim = co_dim, mi=co_mi,  ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file, use_pretrain = use_pretrain, use_seen_only = True, unseen_l = unseen_l)
	if add_emb_diagonal:
		Y_emb = np.column_stack((np.eye(len(i2l)), Y_emb))
	return unseen_l, l2i, i2l, onto_net, Y_emb, onto_net_mat



def graph_embedding(A, i2l, mi=0, dim=20,use_seen_only=True,unseen_l=None):
	nl = np.shape(A)[0]
	if use_seen_only:
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
	if use_seen_only:
		sp = sp[seen_ind, :]
		sp = sp[:,seen_ind]
	X = np.zeros((np.shape(sp)[0],dim))
	svd_dim = min(dim, np.shape(sp)[0]-1)
	if mi==0 or mi == 2:
		X[:,:svd_dim] = svd_emb(sp, dim=svd_dim)
	else:
		X[:,:svd_dim] = DCA_vector(sp, dim=svd_dim)[0]
	if use_seen_only:
		X_ret = np.zeros((nl, dim))
		X_ret[seen_ind,:] = X
	else:
		X_ret = X
	if mi==2 or mi == 3:
		sp *= -1
	return sp, X_ret

def cal_ontology_emb(dim=20, mi=3,  use_pretrain = None, ontology_nlp_file = '../../OnClass_data/cell_ontology/cl.ontology.nlp', ontology_file = '../../OnClass_data/cell_ontology/cl.ontology', use_seen_only = True, unseen_l = None):
	if use_pretrain is None or not os.path.isfile(use_pretrain+'X.npy') or not os.path.isfile(use_pretrain+'sp.npy'):
		cl_nlp = collections.defaultdict(dict)
		if ontology_nlp_file is not None:
			fin = open(ontology_nlp_file)
			for line in fin:
				s,p,wt = line.upper().strip().split('\t')
				cl_nlp[s][p] = float(wt)
				cl_nlp[p][s] = float(wt)
			fin.close()

		fin = open(ontology_file)
		lset = set()
		s2p = {}
		for line in fin:
			w = line.strip().split('\t')
			s = w[0]
			p = w[1]
			if len(w)==2:
				if p in cl_nlp and s in cl_nlp[p]:
					wt = cl_nlp[p][s]
				else:
					wt = 1.
			else:
				wt = float(w[2])
			if s not in s2p:
				s2p[s] = {}
			s2p[s][p] = wt
			lset.add(s)
			lset.add(p)
		fin.close()
		lset = np.sort(list(lset))
		nl = len(lset)
		l2i = dict(zip(lset, range(nl)))
		i2l = dict(zip(range(nl), lset))
		A = np.zeros((nl, nl))
		for s in s2p:
			for p in s2p[s]:
				A[l2i[s], l2i[p]] = s2p[s][p]
				A[l2i[p], l2i[s]] = s2p[s][p]
		sp, X =  graph_embedding(A, i2l, mi=mi, dim=dim, use_seen_only=use_seen_only, unseen_l=unseen_l)
		if use_pretrain is not None:
			i2l_file = use_pretrain+'i2l.npy'
			l2i_file = use_pretrain+'l2i.npy'
			X_file = use_pretrain+'X.npy'
			sp_file = use_pretrain+'sp.npy'
			np.save(X_file, X)
			np.save(i2l_file, i2l)
			np.save(l2i_file, l2i)
			np.save(sp_file, sp)
	else:
		i2l_file = use_pretrain+'i2l.npy'
		l2i_file = use_pretrain+'l2i.npy'
		X_file = use_pretrain+'X.npy'
		sp_file = use_pretrain+'sp.npy'
		X = np.load(X_file)
		i2l = np.load(i2l_file,allow_pickle=True).item()
		l2i = np.load(l2i_file,allow_pickle=True).item()
		sp = np.load(sp_file,allow_pickle=True)
	return X, l2i, i2l, sp

def merge_26_datasets(datanames_26datasets, scan_dim = 50):
	datasets, genes_list, n_cells = load_names(datanames_26datasets,verbose=False,log1p=True)
	datasets, genes = merge_datasets(datasets, genes_list)
	datasets_dimred, genes = process_data(datasets, genes, dimred=scan_dim)
	datasets_dimred, expr_datasets = my_assemble(datasets_dimred, ds_names=datanames_26datasets, expr_datasets = datasets, sigma=150)
	datasets_dimred = sparse.vstack(expr_datasets).toarray()
	return datasets_dimred, genes

def emb_ontology(i2l, dim=20, mi=0,  ontology_nlp_file = '../../OnClass_data/cell_ontology/cl.ontology.nlp', ontology_file = '../../OnClass_data/cell_ontology/cl.ontology', use_pretrain = None, use_seen_only = True, unseen_l = None):
	X, ont_l2i, ont_i2l, A = cal_ontology_emb(dim=dim, mi=mi,  ontology_nlp_file =ontology_nlp_file, ontology_file = ontology_file, use_pretrain = use_pretrain, use_seen_only = True, unseen_l = unseen_l)

	i2emb = np.zeros((len(i2l),dim))
	nl = len(i2l)
	for i in range(nl):
		ant = i2l[i]
		if ant not in ont_l2i:
			print (ant, ont_l2i)
			assert('xxx' in ant.lower() or 'nan' in ant.lower())
			continue
		i2emb[i,:] = X[ont_l2i[ant],:]
	'''
	AA = np.zeros((nl, nl))
	for i in range(nl):
		for j in range(nl):
			anti, antj = i2l[i], i2l[j]
			if anti in ont_l2i and antj in ont_l2i:
				AA[i,j] = A[ont_l2i[anti],ont_l2i[antj]]
	'''
	return i2emb
'''
def get_ontology_parents(GO_net, g):
	term_valid = set()
	ngh_GO = set()
	ngh_GO.add(g)
	while len(ngh_GO) > 0:
		for GO in list(ngh_GO):
			for GO1 in GO_net[GO]:
				ngh_GO.add(GO1)
			ngh_GO.remove(GO)
			term_valid.add(GO)
	return term_valid
'''

def get_ontology_parents(GO_net, g, dfs_depth=100):
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

def create_labels(train_Y, combine_unseen = False, dfs_depth = 1000, ontology_nlp_file = '../../OnClass_data/cell_ontology/cl.ontology.nlp', ontology_file = '../../OnClass_data/cell_ontology/cl.ontology'):

	fin = open(ontology_file)
	lset = set()
	for line in fin:
		s,p = line.strip().split('\t')
		lset.add(s)
		lset.add(p)
	fin.close()

	seen_l = sorted(np.unique(train_Y))
	unseen_l = sorted(lset - set(train_Y))
	ys =  np.concatenate((seen_l, unseen_l))

	i2l = {}
	l2i = {}
	for l in ys:
		nl = len(i2l)
		col = l
		if combine_unseen and l in unseen_l:
			nl = len(seen_l)
			l2i[col] = nl
			i2l[nl] = col
			continue
		l2i[col] = nl
		i2l[nl] = col
	train_Y = [l2i[y] for y in train_Y]
	train_X2Y = ConvertLabels(train_Y, ncls = len(i2l))
	onto_net, onto_net_mat = read_ontology(l2i, dfs_depth = dfs_depth, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
	return unseen_l, l2i, i2l, train_X2Y, onto_net, onto_net_mat

def query_depth_ontology(net, node, root='cl:0000000'):
	depth = 0
	while node != root:
		if len(net[node]) == 0:
			print (node)
		node = sorted(list(net[node].keys()))[0]
		depth += 1
		if depth>100:
			sys.error('root not found')
	return depth


def read_ontology(l2i, dfs_depth = 1000, ontology_nlp_file = '../../OnClass_data/cell_ontology/cl.ontology.nlp', ontology_file = '../../OnClass_data/cell_ontology/cl.ontology'):
	nl = len(l2i)
	net = collections.defaultdict(dict)
	net_mat = np.zeros((nl,nl))
	fin = open(ontology_file)
	for line in fin:
		s,p = line.strip().split('\t')
		si = l2i[s]
		pi = l2i[p]
		net[si][pi] = 1
		net_mat[si][pi] = 1
	fin.close()
	for n in range(nl):
		ngh = get_ontology_parents(net, n, dfs_depth = dfs_depth)
		net[n][n] = 1
		for n1 in ngh:
			net[n][n1] = 1
	return net, net_mat

def extract_label_propagate_tree(onto_net, ncls):
	tree = np.zeros((ncls,ncls))
	for n1 in onto_net:
		for n2 in onto_net[n1]:
			tree[n1,n2] = 1
	return tree

def ConvertLabels(labels, ncls=-1):
	ncell = np.shape(labels)[0]
	if len(np.shape(labels)) ==1 :
		#bin to mat
		if ncls == -1:
			ncls = np.max(labels)
		mat = np.zeros((ncell, ncls))
		for i in range(ncell):
			mat[i, labels[i]] = 1
		return mat
	else:
		if ncls == -1:
			ncls = np.shape(labels)[1]
		vec = np.zeros(ncell)
		for i in range(ncell):
			ind = np.where(labels[i,:]!=0)[0]
			assert(len(ind)<=1) # not multlabel classification
			if len(ind)==0:
				vec[i] = -1
			else:
				vec[i] = ind[0]
		return vec

def MapLabel2CL(test_Y, l2i):
	test_Y_new = np.array([l2i[y] for y in test_Y])
	return test_Y_new

def get_ontology_name(obo_file = 'software/OnClass_data/cell_ontology/cl.obo', lower=True):
	fin = open(obo_file)
	co2name = {}
	name2co = {}
	tag_is_syn = {}
	for line in fin:
		if line.startswith('id: '):
			co = line.strip().split('id: ')[1]
		if line.startswith('name: '):
			if lower:
				name = line.strip().lower().split('name: ')[1]
			else:
				name = line.strip().split('name: ')[1]
			co2name[co] = name
			name2co[name] = co
		if line.startswith('synonym: '):
			if lower:
				syn = line.strip().lower().split('synonym: "')[1].split('" ')[0]
			else:
				syn = line.strip().split('synonym: "')[1].split('" ')[0]
			if syn in name2co:
				continue
			name2co[syn] = co
	fin.close()
	return co2name, name2co

def knn_ngh(Y2Y):
	ind = np.argsort(Y2Y*-1, axis=1)
	return ind

def extend_prediction_2unseen_normalize(pred_Y_seen, onto_net_rwr, nseen, ratio=200):
	sys.exit(-1)#NOT USED
	ncls = np.shape(onto_net_rwr)[0]
	onto_net_rwr = onto_net_rwr - np.tile(np.mean(onto_net_rwr, axis = 1), (ncls, 1))
	pred_Y_seen_norm = pred_Y_seen / pred_Y_seen.sum(axis=1)[:, np.newaxis]
	pred_Y_all = np.dot(pred_Y_seen_norm, onto_net_rwr[:nseen,:])
	pred_Y_all[:,:nseen] = normalize(pred_Y_all[:,:nseen],norm='l1',axis=1)
	pred_Y_all[:,nseen:] = normalize(pred_Y_all[:,nseen:],norm='l1',axis=1) * ratio
	return pred_Y_all

def create_nlp_networks(l2i, onto_net, cls2cls,  ontology_nlp_file):
	ncls = np.shape(cls2cls)[0]
	_, _, onto_nlp_emb = read_cell_ontology_nlp(l2i, ontology_nlp_file = ontology_nlp_file, ontology_nlp_emb_file =  '../OnClass_data/cell_ontology/cl.ontology.nlp.emb')
	onto_net_nlp_all_pairs = (cosine_similarity(onto_nlp_emb) + 1 ) /2#1 - spatial.distance.cosine(onto_nlp_emb, onto_nlp_emb)
	onto_net_nlp = np.zeros((ncls, ncls))
	onto_net_bin = np.zeros((ncls, ncls))
	stack_net_bin = np.zeros((ncls, ncls))
	stack_net_nlp = np.zeros((ncls, ncls))

	for n1 in onto_net:
		for n2 in onto_net[n1]:
			if n1==n2:
				continue
			stack_net_nlp[n2,n1] = onto_net_nlp_all_pairs[n2, n1]
			stack_net_nlp[n1,n2] = onto_net_nlp_all_pairs[n1, n2]
			stack_net_bin[n1,n2] = 1
			stack_net_bin[n2,n1] = 1
	for n1 in range(ncls):
		for n2 in range(ncls):
			if cls2cls[n1,n2] == 1 or cls2cls[n2,n1] == 1:
				onto_net_nlp[n1,n2] = onto_net_nlp_all_pairs[n1, n2]
				onto_net_nlp[n2,n1] = onto_net_nlp_all_pairs[n2, n1]
				onto_net_bin[n1,n2] = 1
				onto_net_bin[n2,n1] = 1
	return onto_net_nlp, onto_net_bin, stack_net_nlp, stack_net_bin, onto_net_nlp_all_pairs


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

def my_auprc(y_true, y_pred):
	precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
	area = auc(recall, precision)
	return area

def sampled_auprc(truths,preds):
	pos = np.where(truths == 1)[0]
	neg = np.where(truths == 0)[0]
	assert(len(pos) + len(neg) == len(truths))
	nneg = len(neg)
	npos = len(pos)
	select_neg = np.random.choice(nneg, npos*3, replace = True)
	select_ind = np.concatenate((pos, select_neg))
	return average_precision_score(truths[select_ind], preds[select_ind])

def evaluate(Y_pred_mat, Y_truth_vec, unseen_l, nseen, Y_truth_bin_mat = None, Y_pred_vec = None, Y_ind=None, Y_net = None, Y_net_mat = None, write_screen = True, write_to_file = None, combine_unseen = False, prefix='', metrics = ['AUROC(seen)','AUPRC(seen)','AUROC','AUPRC','AUROC(unseen)', 'AUPRC(unseen)','Accuracy@3','Accuracy@5']):
	#preprocess scores
	unseen_l = np.array(list(unseen_l))
	ncell,nclass = np.shape(Y_pred_mat)
	nseen = nclass - len(unseen_l)
	if Y_ind is not None:
		non_Y_ind = np.array(list(set(range(nclass)) - set(Y_ind)))
		if len(non_Y_ind)>0:
			Y_pred_mat[:,non_Y_ind] = -1 * np.inf
	if Y_pred_vec is None:
		Y_pred_vec = np.argmax(Y_pred_mat, axis=1)
	if Y_truth_bin_mat is  None:
		Y_truth_bin_mat = ConvertLabels(Y_truth_vec, nclass)

	Y_pred_bin_mat = ConvertLabels(Y_pred_vec, nclass)
	#class-based metrics
	class_auc_macro = np.full(nclass, np.nan)
	class_auprc_macro = np.full(nclass, np.nan)
	class_f1 = np.full(nclass, np.nan)
	for i in range(nclass):
		if len(np.unique(Y_truth_bin_mat[:,i]))==2 and np.sum(Y_truth_bin_mat[:,i])>=10:
			class_auc_macro[i] = roc_auc_score(Y_truth_bin_mat[:,i], Y_pred_mat[:,i])
			class_auprc_macro[i] = sampled_auprc(Y_truth_bin_mat[:,i], Y_pred_mat[:,i])
			class_f1[i] = f1_score(Y_truth_bin_mat[:,i], Y_pred_bin_mat[:,i])


	#sample-based metrics
	extend_acc, extend_Y = extend_accuracy(Y_truth_vec, Y_pred_vec, Y_net, unseen_l)
	kappa = cohen_kappa_score(Y_pred_vec, Y_truth_vec)
	extend_kappa = cohen_kappa_score(extend_Y, Y_truth_vec)
	accuracy = accuracy_score(Y_truth_vec, Y_pred_vec)
	prec_at_k_3 = precision_at_k(Y_pred_mat, Y_truth_vec, 3)
	prec_at_k_5 = precision_at_k(Y_pred_mat, Y_truth_vec, 5)

	#print ([(x,np.sum(Y_truth_bin_mat[:,unseen_l[i]])) for i,x in enumerate(class_auprc_macro[unseen_l]) if not np.isnan(x)])
	seen_auc_macro = np.nanmean(class_auc_macro[:nseen])
	seen_auprc_macro = np.nanmean(class_auprc_macro[:nseen])
	seen_f1 = np.nanmean(class_f1[:nseen])
	if len(unseen_l) == 0:
		unseen_auc_macro = 0
		unseen_auprc_macro = 0
		unseen_f1 = 0
	else:
		unseen_auc_macro = np.nanmean(class_auc_macro[unseen_l])
		#unseen_auprc_macro = np.nanmean([x for i,x in enumerate(class_auprc_macro[unseen_l]) if np.sum(Y_truth_bin_mat[:,unseen_l[i]])>100])#
		unseen_auprc_macro = np.nanmean(class_auprc_macro[unseen_l])
		unseen_f1 = np.nanmean(class_f1[unseen_l])

	#metrics = ['AUROC','AUPRC','unseen_AUROC', 'unseen_AUPRC','Cohens Kappa','Accuracy@3','Accuracy@5']
	#res_v = [seen_auc_macro, seen_auprc_macro, np.nanmean(class_auc_macro), np.nanmean(class_auprc_macro), extend_kappa, prec_at_k_3, prec_at_k_5, unseen_auc_macro, unseen_auprc_macro]
	all_v = {'AUROC':np.nanmean(class_auc_macro), 'AUPRC': np.nanmean(class_auprc_macro), 'AUROC(seen)':seen_auc_macro, 'AUPRC(seen)': seen_auprc_macro, 'AUROC(unseen)':unseen_auc_macro, 'AUPRC(unseen)': unseen_auprc_macro, 'Cohens Kappa':extend_kappa, 'Accuracy@3':prec_at_k_3, 'Accuracy@5':prec_at_k_5}
	res_v = {}
	for metric in metrics:
		res_v[metric] = all_v[metric]
	#res_v = [seen_auc_macro, seen_auprc_macro, seen_f1, np.nanmean(class_auc_macro), np.nanmean(class_auprc_macro), np.nanmean(class_f1), unseen_auc_macro, unseen_auprc_macro, unseen_f1]
	if write_screen:
		print (prefix, end='\t')
		for v in metrics:
			print ('%.4f'%res_v[v], end='\t')
		print ('')
		sys.stdout.flush()
	if write_to_file is not None:
		write_to_file.write(prefix+'\t')
		for v in metrics:
			write_to_file.write('%.2f\t'%res_v[v])
		write_to_file.write('\n')
		write_to_file.flush()
	return res_v

def precision_at_k(pred,truth,k):
	ncell, nclass = np.shape(pred)
	hit = 0.
	for i in range(ncell):
		x = np.argsort(pred[i,:]*-1)
		rank = np.where(x==truth[i])[0][0]
		if rank < k:
			hit += 1.
	prec = hit / ncell
	return prec


def read_type2genes(g2i, marker_gene = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/marker_genes/gene_marker_expert_curated.txt'):
	co2name, name2co = get_ontology_name()

	c2cnew = {}
	c2cnew['cd4+ t cell'] = 'CD4-positive, CXCR3-negative, CCR6-negative, alpha-beta T cell'.lower()
	c2cnew['chromaffin cells (enterendocrine)'] = 'chromaffin cell'.lower()


	c2cnew['mature NK T cell'] = 'mature NK T cell'.lower()
	c2cnew['cd8+ t cell'] = 'CD8-positive, alpha-beta cytotoxic T cell'.lower()
	fin = open(marker_gene)
	fin.readline()
	tp2genes = {}
	unfound = set()
	for line in fin:
		w = line.strip().split('\t')
		c1 = w[1].lower()
		c2 = w[2].lower()
		genes = []
		for ww in w[8:]:
			if ww.upper() in g2i:
				genes.append(ww.upper())
		if len(genes)==0:
			continue
		if c1.endswith('s') and c1[:-1] in name2co:
			c1 = c1[:-1]
		if c2.endswith('s') and c2[:-1] in name2co:
			c2 = c2[:-1]
		if c1 + ' cell' in name2co:
			c1 +=' cell'
		if c2 + ' cell' in name2co:
			c2 +=' cell'
		if c1 in c2cnew:
			c1 = c2cnew[c1]
		if c2 in c2cnew:
			c2 = c2cnew[c2]
		if c1 in name2co:
			tp2genes[name2co[c1]] = genes
		else:
			unfound.add(c1)
		if c2 in name2co:
			tp2genes[name2co[c2]] = genes
		else:
			unfound.add(c2)
	fin.close()

	return tp2genes




def extend_accuracy(test_Y, test_Y_pred_vec, Y_net, unseen_l):
	unseen_l = set(unseen_l)
	n = len(test_Y)
	acc = 0.
	ntmp = 0.
	new_pred = []
	for i in range(n):
		if test_Y[i] in unseen_l and test_Y_pred_vec[i] in unseen_l:
			if test_Y_pred_vec[i] in Y_net[test_Y[i]] and Y_net[test_Y[i]][test_Y_pred_vec[i]] == 1:
				acc += 1
				ntmp += 1
				new_pred.append(test_Y[i])
			else:
				new_pred.append(test_Y_pred_vec[i])
		else:
			if test_Y[i] == test_Y_pred_vec[i]:
				acc += 1
			new_pred.append(test_Y_pred_vec[i])
	new_pred = np.array(new_pred)
	return acc/n, new_pred
