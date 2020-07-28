import sys
from time import time
from scipy import stats, sparse
from scipy.sparse.linalg import svds, eigs
from scipy.special import expit
import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,precision_score,precision_recall_fscore_support, cohen_kappa_score, average_precision_score,f1_score,average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn import metrics
from collections import Counter
from scipy.stats import norm as dist_model
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA
import collections
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/OnClass/'
data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/'
sys.path.append(repo_dir)
sys.path.append(repo_dir+'src/task/OnClass/')
sys.path.append(repo_dir+'src/task/OnClass/model/')
os.chdir(repo_dir)
from model.OnClassPred import OnClassPred
from model.BilinearNNDebug import BilinearNN
from utils import *
from plots import *
from libs import my_assemble, datanames_26datasets

if len(sys.argv) <= 2:
	pid = 0
	total_pid = -1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])


supple_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/SupplementaryTable/'
output_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/MarkerGenes/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
ontology_dir =  '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/Ontology/CellOntology/'

co2name, name2co = get_ontology_name(DATA_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/' ,lower=False)
fig_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/Figures/MarkerGenes/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

nn_nhidden = [100,50,25]
keep_prob = 0.7
unseen_ratio = 0.5
dnames = ['muris_facs']#,'muris_droplet','microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy'
#dnames = ['muris_facs','muris_droplet']
for dnamei, dname1 in enumerate(dnames):
	feature, label, genes1, ontology_nlp_file, ontology_file = read_singlecell_data(dname1, data_dir,nsample=50000000)
	train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = 0)
	train_X = np.vstack((train_X, test_X))
	train_Y_str = np.concatenate((train_Y_str, test_Y_str))
	co_dim = 5
	ontology_emb_file = ontology_dir + str(co_dim)
	unseen_l, l2i1, i2l1, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, co_dim = co_dim, use_pretrain = ontology_emb_file, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
	terms1 = [i2l1[i] for i in range(len(i2l1))]
	cor = np.load(output_dir + dname1 + '.npy')
	ncls = np.shape(Y_emb)[0]
	assert(np.shape(cor)[0] == len(terms1))
	assert(np.shape(cor)[1] == len(genes1))
	g2i1 = {}
	i2g1 = {}
	for i,g in enumerate(genes1):
		g2i1[g] = i
		i2g1[i] = g

	topk = 50
	thres = 0.4
	sub_cor = np.where(np.abs(cor)>thres, 1, 0)
	nmarkers = []
	fout = open(supple_dir + dname1+'_marker_genes.txt', 'w')
	for i in range(ncls):
		our_marker = np.where(sub_cor[i,:]==1)[0]#np.argsort(cor[l2i1[tp],:]*-1)[:topk]
		if len(our_marker) ==0:
			our_marker = np.argsort(sub_cor[i,:]*-1)[:topk]
		fout.write(co2name[i2l1[i]]+'\t'+i2l1[i]+'\t')
		for gi in our_marker:
			fout.write(i2g1[gi]+'\t')
		fout.write('\n')
	fout.close()
