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
from scipy.sparse import csr_matrix
from sklearn import preprocessing
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

if len(sys.argv) <= 2:
	pid = 0
	total_pid = -1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

output_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/26datasets/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
ontology_dir =  '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/Ontology/CellOntology/'
#read TMS
pct = -1
#dnames = ['muris_droplet','microcebusAntoine','microcebusBernard','microcebusStumpy','microcebusMartine']
dnames = ['microcebusAntoine','microcebusStumpy']
scan_dim = 50
nn_nhidden = [100,50,25]
keep_prob = 0.7
unseen_ratio = 0.5
for dname in dnames:
	pct += 1
	if total_pid>1 and pct%total_pid != pid:
		continue
	feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data(dname, data_dir,nsample=50000000)
	genes = [x.upper() for x in genes]
	train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = 0)
	train_X = np.vstack((train_X, test_X))
	train_Y_str = np.concatenate((train_Y_str, test_Y_str))

	co_dim = 5
	ontology_emb_file = ontology_dir + str(co_dim)
	unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, co_dim = co_dim, use_pretrain = ontology_emb_file, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
	ncls = np.shape(cls2cls)[0]
	train_Y = MapLabel2CL(train_Y_str, l2i)
	unseen_l = MapLabel2CL(unseen_l, l2i)
	nunseen = len(unseen_l)
	nseen = ncls - nunseen
	ntrain = np.shape(train_X)[0]
	print (np.shape(train_X), np.shape(train_Y))

	onto_net_rwr = RandomWalkOntology(onto_net, l2i, rst=0.7, ontology_nlp_file = ontology_nlp_file, ontology_nlp_emb_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology.nlp.emb')
	#read 26 datasets

	datasets, genes_list, n_cells = load_names(datanames_26datasets,verbose=False,log1p=True)
	for data in datasets:
		print (data)
	datasets.append(csr_matrix(np.log1p(train_X)))
	genes_list.append(genes)
	datanames_26datasets.append('TMS')
	datasets, genes = merge_datasets(datasets, genes_list)
	datasets_dimred, genes = process_data(datasets, genes, dimred=scan_dim)
	datasets_dimred, expr_datasets = my_assemble(datasets_dimred, ds_names=datanames_26datasets, expr_datasets = datasets, sigma=150)
	datasets_dimred = sparse.vstack(expr_datasets).toarray()
	print (datasets_dimred)
	#datasets_dimred = np.log1p(datasets_dimred)
	#train on TMS
	print (np.shape(datasets_dimred))
	nsample = np.shape(datasets_dimred)[0]
	train_X = datasets_dimred[nsample-ntrain:,:]
	test_X = datasets_dimred[:nsample-ntrain,:]
	print (np.shape(train_X),np.shape(test_X), nseen)
	#train_X, test_X = process_expression([train_X, test_X])


	BN = BilinearNN()
	BN.train(train_X, train_Y, Y_emb, nseen, use_y_emb = False,
	nhidden = nn_nhidden, max_iter = 50, minibatch_size = 128, lr = 0.0001, l2=0.005, keep_prob = 1.0, early_stopping_step = 5)
	pred_Y_seen = BN.predict(test_X)
	np.save(output_dir+dname+'.pred_Y_seen.npy',pred_Y_seen)
	print (np.shape(pred_Y_seen))
	pred_Y_all = extend_prediction_2unseen(pred_Y_seen, onto_net_rwr, nseen, ratio = (ncls*1./nseen)**2)
	print (np.shape(pred_Y_all))
	np.save(output_dir+dname+'.pred_Y_all.npy',pred_Y_all)
