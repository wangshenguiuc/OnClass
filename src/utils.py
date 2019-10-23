from anndata import read_h5ad
import sys
from time import time
from scipy import stats, sparse
import numpy as np
import collections
import os
import time
from fbpca import pca
from collections import Counter
from scipy.sparse.linalg import svds, eigs
from sklearn.metrics import roc_auc_score,accuracy_score,precision_recall_fscore_support, cohen_kappa_score
from sklearn import preprocessing
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.decomposition import PCA
from sklearn import preprocessing
import umap
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
sys.path.append(repo_dir)
os.chdir(repo_dir)
from src.models.random_walk_with_restart.RandomWalkRestart import RandomWalkRestart, DCA_vector
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
MEDIUM_SIZE = 30
BIGGER_SIZE = 50

plt.rc('font', size=MEDIUM_SIZE)		  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)	 # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)	# fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)	# fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)	# fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)	# legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

'''
'/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Scanorama/data/pbmc/10x/68k_pbmc',
'/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Scanorama/data/pbmc/10x/b_cells',
'/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Scanorama/data/pbmc/10x/cd14_monocytes',
'/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Scanorama/data/pbmc/10x/cd4_t_helper',
'/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Scanorama/data/pbmc/10x/cd56_nk',
'/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Scanorama/data/pbmc/10x/cytotoxic_t',
'/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Scanorama/data/pbmc/10x/memory_t',
'/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Scanorama/data/pbmc/10x/regulatory_t',
'/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Scanorama/data/pbmc/pbmc_kang',
'/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/Scanorama/data/pbmc/pbmc_10X']
'''
def map_26_datasets_cell2cid(use_detailed=False):
	if not use_detailed:
		onto_ids = ['CL:0000236','CL:0000235','CL:0000037','CL:0002338','CL:0000492','CL:0000815','CL:0000910','CL:0001054','CL:2000001','CL:0000813']#CL:0000910,0000815
	else:
		onto_ids = ['CL:0000236','CL:0000235','CL:0000037','CL:0002338','CL:0000492','CL:0000792','CL:0000794','CL:0001054','CL:2000001','CL:0000897']#CL:0000910,0000815
	keywords = ['b_cells','infected','hsc','cd56_nk','cd4_t_helper','regulatory_t','cytotoxic_t','cd14_monocytes','pbmc','memory_t']
	keyword2cname = {}
	keyword2cname['b_cells'] = 'B cell'
	keyword2cname['infected'] = 'Macrophage'
	keyword2cname['hsc'] = 'HSC'
	keyword2cname['cd56_nk'] = 'CD56+ NK'
	keyword2cname['cd4_t_helper'] = 'CD4+ helper T'
	keyword2cname['regulatory_t'] = 'Regulatory T'
	keyword2cname['cytotoxic_t'] = 'Cytotoxic T'
	keyword2cname['cd14_monocytes'] = 'CD14+ monocyte'
	keyword2cname['pbmc'] = 'PBMC'
	keyword2cname['memory_t'] = 'Memory T'
	return onto_ids, keywords, keyword2cname

def read_type2genes(g2i, file='/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/tabula_muris_senis/gene_marker/gene_marker_tms.txt'):
	co2name, name2co = get_ontology_name()

	c2cnew = {}
	c2cnew['cd4+ t cell'] = 'CD4-positive, CXCR3-negative, CCR6-negative, alpha-beta T cell'.lower()
	c2cnew['chromaffin cells (enterendocrine)'] = 'chromaffin cell'.lower()


	c2cnew['mature NK T cell'] = 'mature NK T cell'.lower()
	c2cnew['cd8+ t cell'] = 'CD8-positive, alpha-beta cytotoxic T cell'.lower()
	fin = open('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/tabula_muris_senis/gene_marker/gene_marker_tms.txt')
	fin.readline()
	tp2genes = {}
	unfound = set()
	for line in fin:
		w = line.strip().split('\t')
		c1 = w[1].lower()
		c2 = w[2].lower()
		genes = []
		for ww in w[8:]:
			if ww.lower() in g2i:
				genes.append(ww.lower())
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


def get_rgb(n):
	r = n/256/256
	r = int(r)
	g = n - r*256*256
	g = g/256
	g = int(g)
	b = n%256
	b = int(b)
	return r,g,b

def get_man_colors():
	import matplotlib.colors as pltcolors

	cmap = [plt.cm.get_cmap("tab20b")(0)] # Aorta
	for i in range(3,5): # BAT, Bladder
		cmap.append(plt.cm.get_cmap("tab20b")(i))
	for i in range(6,9): # Brain_Myeloid, Brain_Non_Myeloid, Diaphgram
		cmap.append(plt.cm.get_cmap("tab20b")(i))
	for i in range(9,13): # GAT, Heart, Kidney, Large_Intestine
		cmap.append(plt.cm.get_cmap("tab20b")(i))
	print (cmap)
	for i in range(14,20): # Limb_Muscle, Liver, Lung, MAT, Mammary_Gland, Marrow
		cmap.append(plt.cm.get_cmap("tab20b")(i))
	for i in range(0,20): # Pancreas, SCAT
		cmap.append(plt.cm.get_cmap("tab20c")(i))
	print (cmap)
	manual_colors = []
	for c in cmap:
		manual_colors.append(pltcolors.to_hex(c))
	print (manual_colors)
	print (len(manual_colors))
	return manual_colors

def generate_colors(labels):
	labels = np.unique(labels)
	n = len(labels)

	'''
	man_colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999', '#621e15', '#e59076', '#128dcd', '#083c52', '#64c5f2', '#61afaf', '#0f7369', '#9c9da1', '#365e96', '#983334', '#77973d', '#5d437c', '#36869f', '#d1702f', '#8197c5', '#c47f80', '#acc484', '#9887b0', '#2d588a', '#58954c', '#e9a044', '#c12f32', '#723e77', '#7d807f', '#9c9ede', '#7375b5', '#4a5584', '#cedb9c', '#b5cf6b', '#8ca252', '#637939', '#e7cb94', '#e7ba52', '#bd9e39', '#8c6d31', '#e7969c', '#d6616b', '#ad494a', '#843c39', '#de9ed6', '#ce6dbd', '#a55194', '#7b4173', '#000000', '#0000FF']
	man_colors = np.sort(man_colors)
	'''
	man_colors = get_man_colors()
	nman_colors = len(man_colors)
	man_step = int(np.floor(nman_colors*1./n))
	#print (man_step)
	#print (labels)
	color_map = plt.cm.get_cmap('hsv', n)
	marker = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x"]
	lab2marker = {}
	lab2col = {}
	for i in range(n):
		if n > len(man_colors):
			lab2col[labels[i]] = color_map(i)
		else:
			lab2col[labels[i]] = man_colors[i*man_step]
		lab2marker[labels[i]] = marker[i % len(marker)]
	return lab2col, lab2marker


def plot_umap(embedding, lab, lab2col, file,  lab2marker = None, legend=True, size=1,title='',legendmarker=10):

	mpl.rcParams['pdf.fonttype'] = 42
	SMALL_SIZE = 15
	MEDIUM_SIZE = 15
	BIGGER_SIZE = 30

	plt.rc('font', size=SMALL_SIZE)		  # controls default text sizes
	plt.rc('axes', titlesize=BIGGER_SIZE)	 # fontsize of the axes title
	plt.rc('axes', labelsize=BIGGER_SIZE)	# fontsize of the x and y labels
	plt.rc('xtick', labelsize=MEDIUM_SIZE)	# fontsize of the tick labels
	plt.rc('ytick', labelsize=MEDIUM_SIZE)	# fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)	# legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


	if np.shape(embedding)[1]!=2:
		embedding = umap.UMAP(random_state = 1).fit_transform(embedding)
	assert(np.shape(embedding)[1]==2)

	plt.clf()

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	for l in lab2col:
		ind = np.where(lab==l)[0]
		if len(ind)==0:
			continue
		if lab2marker is None:
			plt.scatter(embedding[ind, 0], embedding[ind, 1],c=lab2col[l], label=l, s=size)
		else:
			plt.scatter(embedding[ind, 0], embedding[ind, 1],c=lab2col[l], label=l, s=size, marker=lab2marker[l])
	#plt.title()
	if legend:
		plt.legend(loc='lower left',ncol=6,fontsize=6)
	plt.suptitle(title)
	plt.tick_params(
	axis='x',		  # changes apply to the x-axis
	which='both',	  # both major and minor ticks are affected
	bottom=False,	  # ticks along the bottom edge are off
	top=False,		 # ticks along the top edge are off
	labelbottom=False) # labels along the bottom edge are off
	plt.tick_params(
	axis='y',		  # changes apply to the x-axis
	which='both',	  # both major and minor ticks are affected
	left=False,	  # ticks along the bottom edge are off
	right=False,		 # ticks along the top edge are off
	labelleft=False) # labels along the bottom edge are off
	#plt.axis('off')
	plt.xlabel('UMAP 1')
	plt.ylabel('UMAP 2')
	plt.tight_layout()
	plt.savefig(file,dpi=100)
	plt.savefig(file+'.png',dpi=100)
	if legend:
		return

	handles,labels = ax.get_legend_handles_labels()

	fig_legend = plt.figure(figsize=(20,20))
	axi = fig_legend.add_subplot(111)
	fig_legend.legend(handles, labels, loc='center', scatterpoints = 1, ncol=1, frameon=False,markerscale=legendmarker)
	axi.xaxis.set_visible(False)
	axi.yaxis.set_visible(False)
	plt.savefig(file +'_legend.pdf',dpi=100)
	plt.gcf().clear()

def check_unseen_num(train_Y, test_Y):
	train_Y = set(train_Y)
	test_Y = set(test_Y)
	inter = set(train_Y) & set(test_Y)
	union = set(train_Y) | set(test_Y)
	print ('%d %d %d %d' % (len(inter), len(union), len(test_Y) - len(inter), len(train_Y) - len(inter)))

def parse_para(para_set):
	method_name,split_method,combine_unseen,cell_dim,co_dim,premi,cellmi,comi=para_set.split('#')
	combine_unseen = int(combine_unseen)
	cell_dim = int(cell_dim)
	co_dim = int(co_dim)
	premi = int(premi)
	cellmi = int(cellmi)
	comi = int(comi)
	return method_name,split_method,combine_unseen,cell_dim,co_dim,premi,cellmi,comi

def read_data(months=['21m','3m'],techs=['droplet'],seed=1,nsample=1000000,dlevel='cell_ontology_id',
dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/tabula_muris_senis/2_raw_data_processed/age_method/'):
	np.random.seed(seed)
	datanames = []
	genes_list = {}
	labels = {}
	datasets = {}
	types = {}
	month_labels = {}
	for m in months:
		for tech in techs:
			dataname = tech+m
			fname = dir + dataname + '.h5ad'
			if not os.path.isfile(fname):
				continue
			x = read_h5ad(fname)
			ncell = np.shape(x.X)[0]
			index = np.random.choice(ncell,min(nsample,ncell),replace=False)
			datasets[dataname] = x.X[index,:]
			datanames.append(dataname)
			genes_list[dataname] = x.var.index
			labels[dataname] = np.array(x.obs[dlevel].tolist())[index]
			month_labels[dataname] = np.full(len(x.obs[dlevel].tolist()), len(index))
			types[dataname] = Counter(np.array(x.obs[dlevel].tolist())[index])
	return datanames, genes_list, labels, datasets, types, month_labels


def read_data(filename,seed=1,nsample=3000000,dlevel='cell_ontology_class_reannotated',exclude_tissues=['marrow']):
	name2co = get_ontology_name()[1]
	np.random.seed(seed)
	if 'facs' in filename:
		tech = 'facs'
	elif 'droplet' in filename:
		tech = 'droplet'
	else:
		tech = ''
	if not os.path.isfile(filename):
		sys.exit('%s not exists' % filename)
	x = read_h5ad(filename)
	ncell = np.shape(x.X)[0]
	dataset = x.X
	months = np.array(x.obs['age'].tolist())
	labels = np.array(x.obs[dlevel].tolist())
	tissues = np.array(x.obs['tissue'].tolist())

	ind = []
	for i in range(ncell):
		tis = tissues[i]
		lab = labels[i]
		if tis.lower() in exclude_tissues or lab.lower() not in name2co:
			#print ('%s %s' % (tis, lab))
			continue
		ind.append(i)
	ind = np.array(ind)
	dataset = dataset[ind,:]
	months = months[ind]
	labels = labels[ind]
	tissues = tissues[ind]
	annot = [name2co[y.lower()] for y in labels]
	annot = np.array(annot)

	datanames = []
	genes_list = {}
	labels = {}
	datasets = {}
	types = {}
	month_labels = {}
	uniq_age = np.unique(months)
	for m in uniq_age:
		dataname = tech+m
		datanames.append(dataname)
		index = np.array(months == m)
		datasets[dataname] = dataset[index,:]
		genes_list[dataname] = x.var.index
		labels[dataname] = annot[index]
		month_labels[dataname] = np.full(len(annot), len(index))
		types[dataname] = Counter(np.array(annot)[index])
	all_X, all_Y = extract_data(datanames, datasets, labels)
	return all_X, all_Y


def extract_data(datanames, datasets, labels):
	#datanames = np.sort(datanames)
	mat = []
	Y = []
	for di,dataname in enumerate(datanames):
		Y.append(labels[dataname])
		mat.append(datasets[dataname])
	Y = np.concatenate(Y)
	mat = sparse.vstack(mat)
	return mat, Y

def emb_cells_scan(train_X, test_X, dim=20):
	train_X = np.log1p(train_X)
	test_X = np.log1p(test_X)
	train_X = preprocessing.normalize(train_X, axis=1)
	test_X = preprocessing.normalize(test_X, axis=1)
	ntrain = np.shape(train_X)[0]
	mat = sparse.vstack((train_X, test_X))
	U, s, Vt = pca(mat, k=dim) # Automatically centers.
	X = U[:, range(dim)] * s[range(dim)]
	return X[:ntrain,:], X[ntrain:,:]

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
	'''
	ntrain = np.shape(train_X)[0]
	if log1p:
		train_X = np.log1p(train_X)
		test_X = np.log1p(test_X)
	if norm:
		train_X = preprocessing.normalize(train_X, axis=1)
		test_X = preprocessing.normalize(test_X, axis=1)
	mat = sparse.vstack((train_X, test_X))
	if mi == 0:
		X = svd_emb(mat, dim=dim)
	else:
		sys.exit('wrong emb method')
	return X[:ntrain,:], X[ntrain:,:]
	'''

def svd_emb(mat, dim=20):
	U, S, V = svds(mat, k=dim)
	X = np.dot(U, np.sqrt(np.diag(S)))
	return X

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


def evaluate(test_Y_pred, train_Y, test_Y, unseen_l, test_Y_pred_vec = None, combine_unseen = False):

	Y_ind = np.array(list(set(test_Y) |  set(train_Y)))
	Y_ind = np.sort(Y_ind)
	ncell,nclass = np.shape(test_Y_pred)
	nseen = nclass - len(unseen_l)

	if Y_ind is not None:
		non_Y_ind = np.array(list(set(range(nclass)) - set(Y_ind)))
		#print (non_Y_ind)
		test_Y_pred[:,non_Y_ind] = -1 * np.inf

	unseen_l = np.array(list(unseen_l))
	test_Y = np.array(test_Y)
	if combine_unseen:
		unseen_l = [nseen]
		test_Y[test_Y>=nseen] = nseen
		test_Y_pred_new = np.zeros((ncell, nseen+1))
		test_Y_pred_new[:,:nseen] = test_Y_pred[:, :nseen]
		test_Y_pred_new[:,nseen] = np.max(test_Y_pred[:, nseen:],axis=1)
		test_Y_pred = test_Y_pred_new

	test_Y_truth = np.zeros((ncell, nclass))
	for i in range(ncell):
		test_Y_truth[i,test_Y[i]] = 1

	class_auc_macro = np.full(nclass, np.nan)
	for i in range(nclass):
		if len(np.unique(test_Y_truth[:,i]))==2:
			class_auc_macro[i] = roc_auc_score(test_Y_truth[:,i], test_Y_pred[:,i])
	if test_Y_pred_vec is None:
		test_Y_pred_vec = np.argmax(test_Y_pred, axis=1)
	kappa = cohen_kappa_score(test_Y_pred_vec, test_Y)
	prec_at_k_3 = precision_at_k(test_Y_pred, test_Y, 3)
	prec_at_k_5 = precision_at_k(test_Y_pred, test_Y, 5)

	sel_c = []
	for i in unseen_l:
		sel_c.extend(np.where(test_Y == i)[0])
	sel_c = np.array(sel_c)

	if len(sel_c)<=1:
		print ('Warning, very few unseen points:%d',len(sel_c))
	if len(unseen_l) == 0:
		unseen_auc_macro = 0
	else:
		unseen_auc_macro = np.nanmean(class_auc_macro[unseen_l])
	res_v = {}
	res_v['prec_at_k_3'] = prec_at_k_3
	res_v['prec_at_k_5'] = prec_at_k_5
	res_v['kappa'] = kappa
	res_v['unseen_auc_macro'] = unseen_auc_macro
	res_v['class_auc_macro'] = np.nanmean(class_auc_macro)
	return res_v


def get_ontology_name():
	fin = open('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/cl.obo')
	co2name = {}
	name2co = {}
	tag_is_syn = {}
	for line in fin:
		if line.startswith('id: '):
			co = line.strip().split('id: ')[1]
		if line.startswith('name: '):
			name = line.strip().lower().split('name: ')[1]
			co2name[co] = name
			name2co[name] = co
		if line.startswith('synonym: '):
			syn = line.strip().lower().split('synonym: "')[1].split('" ')[0]
			if syn in name2co:
				continue
			name2co[syn] = co
	fin.close()
	return co2name, name2co

def cal_ontology_emb(dim=20, mi=0):
	fin = open('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/cl.ontology')
	lset = set()
	s2p = {}
	for line in fin:
		s,p = line.strip().split('\t')
		if s not in s2p:
			s2p[s] = set()
		s2p[s].add(p)
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
			A[l2i[s], l2i[p]] = 1
			A[l2i[p], l2i[s]] = 1
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
	return X, l2i, i2l, sp

def postprocess(mat, mi=0):
	if mi==0:
		return preprocessing.scale(mat,axis=0)
	elif mi==1:
		return preprocessing.scale(mat,axis=1)
	elif mi==2:
		return preprocessing.normalize(mat,axis=0)
	elif mi==3:
		return preprocessing.normalize(mat,axis=1)
	elif mi==4:
		return mat

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

def read_ontology(l2i):
	nl = len(l2i)
	net = collections.defaultdict(dict)
	fin = open('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/cl.ontology')
	for line in fin:
		s,p = line.strip().split('\t')
		si = l2i[s]
		pi = l2i[p]
		net[pi][si] = 1
	fin.close()
	for n in range(nl):
		ngh = get_onotlogy_parents(net, n)
		net[n][n] = 1
		for n1 in ngh:
			net[n][n1] = 1
	return net

def emb_onotlogy(i2l, dim=20, mi=0):
	X, ont_l2i, ont_i2l, A = cal_ontology_emb(dim=dim, mi=mi)
	i2emb = np.zeros((len(i2l),dim))
	nl = len(i2l)
	for i in range(nl):
		ant = i2l[i]
		if ant not in ont_l2i:
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


def create_labels(train_Y, combine_unseen = False):
	fin = open('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/cl.ontology')
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
	onto_net = read_ontology(l2i)
	return unseen_l, l2i, i2l, train_X2Y, onto_net

def extract_data_based_on_class(feats, labels, sel_labels):
	ind = []
	for l in sel_labels:
		id = np.where(labels == l)[0]
		ind.extend(id)
	np.random.shuffle(ind)
	X = feats[ind,:]
	Y = labels[ind]
	return X, Y


def ImputeUnseenCls(y_vec, y_raw, cls2cls_sim, nunseen, knn=1, combine_unseen=False):
	if combine_unseen:
		return ConvertLabels(y_vec, ncls=np.shape(cls2cls_sim)[0])
	nclass = np.shape(cls2cls_sim)[0]
	nseen =nclass - nunseen
	seen2unseen_sim = cls2cls_sim[:nseen, nseen:]
	ncell = len(y_vec)
	y_mat = np.zeros((ncell, nclass))
	y_mat[:,:nseen] = y_raw[:, :nseen]
	for i in range(ncell):
		if y_vec[i] == nseen:
			kngh = np.argsort(y_raw[i,:]*-1)[0:knn]
			kngh_new = []
			for k in kngh:
				if y_raw[i,k]!=0:
					kngh_new.append(k)
			kngh_new = np.array(kngh_new)
			if len(kngh_new) == 0:
				continue
			y_mat[i,:nseen] -= 10000
			y_mat[i,nseen:] = np.dot(y_raw[i,kngh_new], seen2unseen_sim[kngh_new,:])
			#unseen_c = np.argsort(wt_sc * -1)[0]
			#y_vec[i] = unseen_c + nseen
			#y_mat[i,y_vec[i]] = 1.
	#y_mat = ConvertLabels(y_vec, ncls=np.shape(cls2cls_sim)[0])
	return y_mat

def filter_no_label_cells(X, Y):
	if np.unique(Y)[0].startswith('CL'):
		return X, Y
	fin = open('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/tabula_muris_senis/2_raw_data_processed/age_method/cell_ontology_id.txt')
	annot2id = {}
	remove_ind = []
	for line in fin:
		v,i=line.rstrip().split('\t')
		i = int(i)
		annot2id[i] = v.replace('_',':').replace('CL:000115','CL:0000115')
		if 'xxx' in annot2id[int(i)] or 'nan' in annot2id[int(i)]:
			remove_ind.extend(np.where(Y==i)[0])
	fin.close()
	remove_ind = np.array(remove_ind)
	N = len(Y)
	keep_ind = [x for x in range(N) if x not in remove_ind]
	keep_ind = np.array(keep_ind)
	print ('%d cells have no valid annotations'%(len(remove_ind)))
	return X[keep_ind,:], Y[keep_ind]


def SplitTrainTest(all_X, all_Y, iter=10, nfold_cls = 0.3, nfold_sample = 0.2, nmin_size=10):
	np.random.seed(iter)

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
	train_X, train_Y = extract_data_based_on_class(all_X, all_Y, train_cls)
	test_X, test_Y = extract_data_based_on_class(all_X, all_Y, test_cls)
	train_X_train, train_X_test, train_Y_train, train_Y_test = train_test_split(
 	train_X, train_Y, test_size=nfold_sample, stratify = train_Y,random_state=iter)
	test_X = sparse.vstack((test_X, train_X_test))
	test_Y = np.concatenate((test_Y, train_Y_test))
	train_X = train_X_train
	train_Y = train_Y_train

	train_X, train_Y = filter_no_label_cells(train_X, train_Y)
	test_X, test_Y = filter_no_label_cells(test_X, test_Y)

	return train_X, train_Y, test_X, test_Y

def ParseCLOnto(train_Y, co_dim=1000, co_mi=3, combine_unseen = False):#
	unseen_l, l2i, i2l, train_X2Y, onto_net = create_labels(train_Y, combine_unseen = combine_unseen)
	Y_emb, cls2cls = emb_onotlogy(i2l, dim = co_dim, mi=co_mi)
	return unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls

def MapLabel2CL(test_Y, l2i):
	test_Y = np.array([l2i[y] for y in test_Y])
	return test_Y
