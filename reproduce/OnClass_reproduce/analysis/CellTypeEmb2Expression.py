import sys
from time import time
import timeit
from datetime import datetime
from scipy import stats, sparse
from scipy.sparse.linalg import svds, eigs
from scipy.special import expit
import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn import metrics
from collections import Counter
from sklearn.preprocessing import normalize
from scipy.stats import norm as dist_model
from scipy.sparse import csr_matrix
from sklearn import preprocessing
import umap
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from scipy.sparse.linalg import svds, eigs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from model.BilinearNNDebug import BilinearNN
from plots import *
from libs import my_assemble, datanames_26datasets



if len(sys.argv) <= 2:
	pid = 0
	total_pid = -1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

output_dir = OUTPUT_DIR + '/26datasets/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

fig_dir = OUTPUT_DIR + '/Figures/EmbeddingAnalyze/'
if not os.path.exists(fig_dir):
	os.makedirs(fig_dir)

ontology_dir =  OUTPUT_DIR + '/Ontology/CellOntology/'
#read TMS


file = DATA_DIR + 'TMS_official_060520/tabula-muris-senis-facs-official-raw-obj.h5ad'
dim = 10
feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data('muris_facs', DATA_DIR,nsample=50000000)
co_dim = 5
ontology_emb_file = ontology_dir + str(co_dim)
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(label, co_dim = co_dim, use_pretrain = ontology_emb_file, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
onto_net_rwr = RandomWalkOntology(onto_net, l2i, rst=0.7, ontology_nlp_file = ontology_nlp_file, ontology_nlp_emb_file = ontology_nlp_emb_file)
#y2y = metrics.pairwise.pairwise_distances(onto_net_rwr)
u, s, vt = svds(np.log(onto_net_rwr + 1e-5) - np.log(1e-5), k=dim)
y2y = metrics.pairwise.cosine_similarity(np.dot(u, np.diag(np.sqrt(s))))

np.random.seed(0)
x = read_h5ad(file)
ncell = np.shape(x.X)[0]
tissues = x.obs['tissue'].tolist()
exp = x.X.toarray()
labels = x.obs['cell_ontology_id'].tolist()
tissues = np.array(tissues)
labels = np.array(labels)
ind = []
for i in range(ncell):
	if labels[i] in l2i:
		ind.append(i)
ind = np.array(ind)
print (np.shape(ind), np.shape(y2y))
labels = labels[ind]
tissues = tissues[ind]
exp = exp[ind, :]
labels = [l2i[l] for l in labels]
labels = np.array(labels)
unq_labels = np.unique(labels)
exp = np.log2(exp + 1)
exp_u, exp_s, exp_vt = svds(exp, k=dim)
exp_imputed = np.dot(exp_u, np.diag(np.sqrt(exp_s)))
print (np.shape(exp_imputed), np.shape(exp))
unq_tissues = np.unique(tissues)
print (unq_tissues)
fig_ct = 2
for tis in unq_tissues:
	y2y_vec = []
	exp_vec = []
	for l1 in unq_labels:
		y1_ind = np.where((labels == l1) & (tissues == tis))[0]
		for l2 in unq_labels:
			if l1>=l2:
				continue
			y2_ind = np.where((labels == l2 )& (tissues == tis))[0]
			if len(y1_ind)==0 or len(y2_ind)==0:
				continue
			exp_sim = np.mean(metrics.pairwise.cosine_similarity(exp_imputed[y1_ind,:], exp_imputed[y2_ind,:]))
			exp_vec.append(exp_sim)
			y2y_vec.append(y2y[l1, l2])
	if len(y2y_vec) < 10:
		continue
	exp_vec = np.array(exp_vec)
	y2y_vec = np.array(y2y_vec)
	pear,pv = stats.pearsonr(exp_vec,y2y_vec)
	if tis == 'Lung' or tis =='Pancreas':
		plot_expression_embedding_spearman(exp_vec,y2y_vec,tis.replace('_',' '),fig_file = fig_dir +' '+tis )
		continue
	plot_expression_embedding_spearman(exp_vec,y2y_vec,tis.replace('_',' '),fig_file = fig_dir + 'Suppl. '+str(fig_ct)+' '+tis)
	fig_ct+=1
