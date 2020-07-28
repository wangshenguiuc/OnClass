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
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn import metrics
from collections import Counter
from sklearn.preprocessing import normalize
from scipy.stats import norm as dist_model
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/OnClass/'
data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/'
sys.path.append(repo_dir)
sys.path.append(repo_dir+'src/task/OnClass/')
sys.path.append(repo_dir+'src/task/OnClass/model/')
os.chdir(repo_dir)
from model.OnClassPred import OnClassPred
from model.BilinearNNDebug import BilinearNN
from utils import *
from libs import my_assemble, datanames_26datasets
from plots import *


supple_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/SupplementaryTable/'

if len(sys.argv) <= 2:
	pid = 0
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

fig_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/Figures/CrossDataset/'
if not os.path.exists(fig_dir):
	os.makedirs(fig_dir)
output_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/Crossdatasets/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
ontology_dir =  '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/Ontology/CellOntology/'
#read TMS
nn_nhidden = [100,50,25]
keep_prob = 0.7
dname = 'muris_facs'
unseen_ratio = 0.5
dnames = ['muris_facs','muris_droplet','microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy']
co2name, name2co = get_ontology_name(DATA_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/', lower=False)
dname1 = 'muris_facs'
dname2 = 'muris_droplet'
feature1, label1, genes1, ontology_nlp_file, ontology_file = read_singlecell_data(dname1, data_dir,nsample=50000000)
co_dim = 5
ontology_emb_file = ontology_dir + str(co_dim)
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(label1, co_dim = co_dim, use_pretrain = ontology_emb_file, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
print (np.sum(cls2cls))
ncls = np.shape(cls2cls)[0]
label1 = MapLabel2CL(label1, l2i)
unseen_l = MapLabel2CL(unseen_l, l2i)
nunseen = len(unseen_l)
nseen = ncls - nunseen

feature2, label2, genes2, ontology_nlp_file, ontology_file = read_singlecell_data(dname2, data_dir,nsample=50000000)
label2 = MapLabel2CL(label2, l2i)

common_genes = np.array(list(set(genes1) & set(genes2)))
gid1 = find_gene_ind(genes1, common_genes)
gid2 = find_gene_ind(genes2, common_genes)

feature1 = feature1[:, gid1]
feature2 = feature2[:, gid2]

cls2cls = np.zeros((ncls, ncls))
fin = open(ontology_file)
for line in fin:
	w = line.strip().split('\t')
	cls2cls[l2i[w[0]], l2i[w[1]]] = 1
	cls2cls[l2i[w[1]], l2i[w[0]]] = 1
fin.close()
pred_Y_all = np.load(output_dir+dname1+'.'+dname2+'pred_Y_all.npy')
ncell,ngene = np.shape(pred_Y_all)
pred_Y_non_max_id = np.argsort(pred_Y_all*-1, axis=0)[2:]
Y2Y_sc = metrics.pairwise_distances(pred_Y_all.T)
print (np.shape(Y2Y_sc))
#np.fill_diagonal(Y2Y_sc, 10000)
diss = []
pairs = []
for i in range(ncls):
	for j in range(ncls):
		if cls2cls[i,j]==0 and cls2cls[j,i]==0:
			continue
		if i not in np.unique(label1) and j not in np.unique(label1):
			continue
		if i>=j:
			continue
		diss.append(Y2Y_sc[i,j])
		pairs.append((i,j))
close_term = np.argsort(diss)
topk = 30
fout = open(supple_dir+'new_cell_populations.txt','w')
for i in range(topk):
	ind = close_term[i]
	id0 = pairs[ind][0]
	id1 = pairs[ind][1]
	print (close_term[i], diss[ind], i2l[id0], i2l[id1], co2name[i2l[id1]], co2name[i2l[id0]], cls2cls[id0,id1], cls2cls[id1,id0])
	fout.write(str(diss[ind])+'\t'+i2l[id0]+'\t'+str(co2name[i2l[id0]])+'\t'+i2l[id1]+'\t'+str(co2name[i2l[id1]])+'\n')
	sc = np.mean(pred_Y_all[:,[id0, id1]], axis=1)
	mat = np.zeros((3, ncell))
	mat[0,:] = sc
	mat[1:,:] = pred_Y_all[:,[id0, id1]].T
	assert(len(sc) == ncell)
	cor = corr2_coeff(mat, feature2.T)
	cor = np.nan_to_num(cor)
	print (np.shape(cor))
	topk = 50
	thres = 0.4
	sub_cor = np.where(np.abs(cor)>thres, 1, 0)
	#nmarkers = []
	#for i in range(3):
	#	markers = np.argsort(cor[i,:])
	#	print (markers)
	#	#markers = np.where(sub_cor[i,:]==1)[0]
	#	#print (i, markers)
fout.close()
