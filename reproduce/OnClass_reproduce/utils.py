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
from sklearn import preprocessing
from fbpca import pca
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from libs import *
from sklearn.utils.graph_shortest_path import graph_shortest_path
from scipy.sparse.linalg import svds, eigs
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
sys.path.append(repo_dir)
os.chdir(repo_dir)

'''
OUTPUT_DIR : path to the output of OnClass
DATA_DIR : path to the data you get from figshare
REPO_DIR : path to your code repo
'''
OUTPUT_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/'
DATA_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/'
REPO_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/OnClass/'
ontology_nlp_emb_file = DATA_DIR + '/cell_ontology/cl.ontology.nlp.emb'

def ImputeUnseenCls(y_vec, y_raw, cls2cls, nseen, knn=1):
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

def RandomWalkOntology(onto_net, l2i, ontology_nlp_file, ontology_nlp_emb_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology.nlp.emb', rst = 0.7):
	ncls = len(l2i)
	onto_net_nlp, _, onto_nlp_emb = read_cell_ontology_nlp(l2i, ontology_nlp_file = ontology_nlp_file, ontology_nlp_emb_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology.nlp.emb')
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

def read_singlecell_data(dname, data_dir, nsample = 500, read_tissue = False):
	if 'microcebus' in dname:
		tech = '10x'
		#file = data_dir + 'TMS_official_060520/' + 'tabula-microcebus_smartseq2-10x_combined_annotated_filtered_gene-labels-correct.h5ad'
		file = data_dir + 'TMS_official_060520/' + dname +'.h5ad'
		filter_key={'method':tech }
		batch_key = ''#original_channel
		ontology_nlp_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology.nlp'
		ontology_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology'
		if not read_tissue:
			feature, label, genes = parse_h5ad(file, nsample = nsample, read_tissue = read_tissue, label_key='cell_ontology_class', batch_key = batch_key, filter_key = filter_key, cell_ontology_file = ontology_file, exclude_non_leaf_ontology = True, exclude_non_ontology = True, DATA_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/')
		else:
			feature, label, genes, tissues = parse_h5ad(file, nsample = nsample, read_tissue = read_tissue, label_key='cell_ontology_class', batch_key = batch_key, filter_key = filter_key, cell_ontology_file = ontology_file, exclude_non_leaf_ontology = True, exclude_non_ontology = True, DATA_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/')
	elif 'muris' in dname:
		tech = dname.split('_')[1]
		file = data_dir + 'TMS_official_060520/' + 'tabula-muris-senis-'+tech+'-official-raw-obj.h5ad'
		filter_key = {}
		batch_key = ''
		ontology_nlp_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology.nlp'
		ontology_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology'
		if not read_tissue:
			feature, label, genes = parse_h5ad(file, nsample = nsample, read_tissue = read_tissue, label_key='cell_ontology_class', batch_key = batch_key, cell_ontology_file = ontology_file, filter_key=filter_key, exclude_non_leaf_ontology = True, exclude_non_ontology = True, DATA_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/')
		else:
			feature, label, genes, tissues = parse_h5ad(file, nsample = nsample, read_tissue = read_tissue, label_key='cell_ontology_class', batch_key = batch_key, cell_ontology_file = ontology_file, filter_key=filter_key, exclude_non_leaf_ontology = True, exclude_non_ontology = True, DATA_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/')
	elif 'allen_part' in dname:
		feature_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Allen/matrix_part.csv'
		label_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Allen/metadata.csv'
		ontology_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Allen/cell_type_ontology'
		ontology_nlp_file = None
		feature, label, genes = parse_csv(feature_file, label_file, nsample = nsample, label_key='cell_type_accession_label', exclude_non_ontology = True, exclude_non_leaf_ontology = True, cell_ontology_file=ontology_file)
	elif 'allen' in dname:
		feature_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Allen/features.pkl'
		label_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Allen/labels.pkl'
		gene_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Allen/genes.pkl'
		ontology_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Allen/cell_type_ontology'
		ontology_nlp_file = None
		feature, label, genes = parse_csv(feature_file, label_file, nsample = nsample, gene_file = gene_file, label_key='cell_type_accession_label', exclude_non_leaf_ontology = True, exclude_non_ontology = True, cell_ontology_file=ontology_file)

	else:
		sys.exit('wrong dname '+dname)
	if read_tissue:
		return feature, label, genes, tissues, ontology_nlp_file, ontology_file
	else:
		return feature, label, genes, ontology_nlp_file, ontology_file

def parse_csv(feature_file, label_file, seed = 1, gene_file = None, nsample = 1000, exclude_non_leaf_ontology = True, label_key='cell_type_accession_label',sample_key='sample_name', exclude_non_ontology = True, cell_ontology_file=None):
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
	else:
		labels = pd.read_csv(label_file,header=0,index_col=None, sep=',')
		sample2label = dict(zip(labels[sample_key], labels[label_key]))
		features = pd.read_csv(feature_file,header=0,index_col=None, sep=',')
		nsample = len(features[sample_key])

		labels = []
		for i in range(nsample):
			labels.append(sample2label[features[sample_key][i]])
		genes = np.array(list(features.keys())[1:])
		features = features.values[:,1:]
		labels = np.array(labels)
		#features = sparse.csr_matrix(features.astype(int))

		if exclude_non_ontology:
			ctype = set()
			fin = open(cell_ontology_file)
			for line in fin:
				ctype.add(line.strip().split('\t')[0])
				ctype.add(line.strip().split('\t')[1])
			new_ids = []
			for i in range(len(labels)):
				if labels[i] in ctype:
					new_ids.append(i)
			features = features[new_ids, :]
			labels = labels[new_ids]
		features = np.array(features.astype(int))
	if exclude_non_leaf_ontology:
		new_ids, exclude_terms = exclude_parent_child_nodes(cell_ontology_file, labels)
		#print (len(exclude_terms),'non leaf terms are excluded')
		features = features[new_ids, :]
		labels = labels[new_ids]
	genes = [x.upper() for x in genes]
	genes = np.array(genes)
	return features, labels, genes


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
	print (np.shape(x.X))
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
	co2name, name2co = get_ontology_name(DATA_DIR = DATA_DIR)
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
		#print 'random walk iter',i, delta
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

def cal_ontology_emb(dim=20, mi=0,  use_pretrain = None, ontology_nlp_file = '../../OnClass_data/cell_ontology/cl.ontology.nlp', ontology_file = '../../OnClass_data/cell_ontology/cl.ontology'):
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
		if mi==0:
			sp = graph_shortest_path(A,method='FW',directed =False)
			X = svd_emb(sp, dim=dim)
			sp *= -1.
		elif mi==1:
			sp = graph_shortest_path(A,method='FW',directed =False)
			X = DCA_vector(sp, dim=dim)[0]
			sp *= -1.
		elif mi==2:
			sp = RandomWalkRestart(A, 0.8)
			X = svd_emb(sp, dim=dim)
		elif mi==3:
			sp = RandomWalkRestart(A, 0.8)
			X = DCA_vector(sp, dim=dim)[0]
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

def ParseCLOnto(train_Y, co_dim=1000, co_mi=3, combine_unseen = False,  use_pretrain = None, ontology_nlp_file = '../../OnClass_data/cell_ontology/cl.ontology.nlp', ontology_file = '../../OnClass_data/cell_ontology/cl.ontology'):#
	unseen_l, l2i, i2l, train_X2Y, onto_net = create_labels(train_Y, combine_unseen = combine_unseen, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
	Y_emb, cls2cls = emb_ontology(i2l, dim = co_dim, mi=co_mi,  ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file, use_pretrain = use_pretrain)
	return unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls

def emb_ontology(i2l, dim=20, mi=0,  ontology_nlp_file = '../../OnClass_data/cell_ontology/cl.ontology.nlp', ontology_file = '../../OnClass_data/cell_ontology/cl.ontology', use_pretrain = None):
	X, ont_l2i, ont_i2l, A = cal_ontology_emb(dim=dim, mi=mi,  ontology_nlp_file =ontology_nlp_file, ontology_file = ontology_file, use_pretrain = use_pretrain)
	i2emb = np.zeros((len(i2l),dim))
	nl = len(i2l)
	for i in range(nl):
		ant = i2l[i]
		if ant not in ont_l2i:
			print (ant, ont_l2i)
			assert('xxx' in ant.lower() or 'nan' in ant.lower())
			continue
		i2emb[i,:] = X[ont_l2i[ant],:]
	AA = np.zeros((nl, nl))
	for i in range(nl):
		for j in range(nl):
			anti, antj = i2l[i], i2l[j]
			if anti in ont_l2i and antj in ont_l2i:
				AA[i,j] = A[ont_l2i[anti],ont_l2i[antj]]
	return i2emb, AA

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

def create_labels(train_Y, combine_unseen = False, ontology_nlp_file = '../../OnClass_data/cell_ontology/cl.ontology.nlp', ontology_file = '../../OnClass_data/cell_ontology/cl.ontology'):

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
	onto_net = read_ontology(l2i, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
	return unseen_l, l2i, i2l, train_X2Y, onto_net

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

def read_ontology(l2i, ontology_nlp_file = '../../OnClass_data/cell_ontology/cl.ontology.nlp', ontology_file = '../../OnClass_data/cell_ontology/cl.ontology'):
	nl = len(l2i)
	net = collections.defaultdict(dict)
	fin = open(ontology_file)
	for line in fin:
		s,p = line.strip().split('\t')
		si = l2i[s]
		pi = l2i[p]
		net[si][pi] = 1
	fin.close()
	for n in range(nl):
		ngh = get_ontology_parents(net, n)
		net[n][n] = 1
		for n1 in ngh:
			net[n][n1] = 1
	return net

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
	test_Y = np.array([l2i[y] for y in test_Y])
	return test_Y

def get_ontology_name(DATA_DIR = '', lower=True):
	fin = open(DATA_DIR + '/cell_ontology/cl.obo')
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

def extend_prediction_2unseen(pred_Y_seen, onto_net_rwr, nseen, ratio=200):
	pred_Y_seen_norm = pred_Y_seen / pred_Y_seen.sum(axis=1)[:, np.newaxis]
	pred_Y_all = np.dot(pred_Y_seen_norm, onto_net_rwr[:nseen,:])
	pred_Y_all[:,:nseen] = normalize(pred_Y_all[:,:nseen],norm='l1',axis=1)
	pred_Y_all[:,nseen:] = normalize(pred_Y_all[:,nseen:],norm='l1',axis=1) * ratio
	return pred_Y_all

def my_auprc(y_true, y_pred):
	precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
	area = auc(recall, precision)
	return area

def evaluate(Y_pred_mat, Y_truth_vec, unseen_l, nseen, Y_pred_vec = None, Y_ind=None, Y_net = None, write_screen = True, write_to_file = None, combine_unseen = False, prefix='', metrics = ['AUROC(seen)','AUPRC(seen)','AUROC','AUPRC','AUROC(unseen)', 'AUPRC(unseen)','Accuracy@3','Accuracy@5']):
	#preprocess scores
	unseen_l = np.array(list(unseen_l))
	ncell,nclass = np.shape(Y_pred_mat)
	nseen = nclass - len(unseen_l)
	if Y_ind is not None:
		non_Y_ind = np.array(list(set(range(nclass)) - set(Y_ind)))
		Y_pred_mat[:,non_Y_ind] = -1 * np.inf
	if Y_pred_vec is None:
		Y_pred_vec = np.argmax(Y_pred_mat, axis=1)
	Y_truth_bin_mat = ConvertLabels(Y_truth_vec, nclass)
	Y_pred_bin_mat = ConvertLabels(Y_pred_vec, nclass)
	#class-based metrics
	class_auc_macro = np.full(nclass, np.nan)
	class_auprc_macro = np.full(nclass, np.nan)
	class_f1 = np.full(nclass, np.nan)
	for i in range(nclass):
		if len(np.unique(Y_truth_bin_mat[:,i]))==2:
			class_auc_macro[i] = roc_auc_score(Y_truth_bin_mat[:,i], Y_pred_mat[:,i])
			class_auprc_macro[i] = average_precision_score(Y_truth_bin_mat[:,i], Y_pred_mat[:,i])
			class_f1[i] = f1_score(Y_truth_bin_mat[:,i], Y_pred_bin_mat[:,i])


	#sample-based metrics
	extend_acc, extend_Y = extend_accuracy(Y_truth_vec, Y_pred_vec, Y_net, unseen_l)
	kappa = cohen_kappa_score(Y_pred_vec, Y_truth_vec)
	extend_kappa = cohen_kappa_score(extend_Y, Y_truth_vec)
	accuracy = accuracy_score(Y_truth_vec, Y_pred_vec)
	prec_at_k_3 = precision_at_k(Y_pred_mat, Y_truth_vec, 3)
	prec_at_k_5 = precision_at_k(Y_pred_mat, Y_truth_vec, 5)

	if len(unseen_l) == 0:
		unseen_auc_macro = 0
		unseen_auprc_macro = 0
		unseen_f1 = 0
	else:
		unseen_auc_macro = np.nanmean(class_auc_macro[unseen_l])
		unseen_auprc_macro = np.nanmean(class_auprc_macro[unseen_l])
		unseen_f1 = np.nanmean(class_f1[unseen_l])
		seen_auc_macro = np.nanmean(class_auc_macro[:nseen])
		seen_auprc_macro = np.nanmean(class_auprc_macro[:nseen])
		seen_f1 = np.nanmean(class_f1[:nseen])
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
			print ('%.2f'%res_v[v], end='\t')
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
	co2name, name2co = get_ontology_name(DATA_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/')

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
