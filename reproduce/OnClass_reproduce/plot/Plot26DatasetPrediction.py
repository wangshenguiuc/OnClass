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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from libs import my_assemble, datanames_26datasets
from plots import *

output_dir = OUTPUT_DIR + '/26datasets/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
fig_dir = OUTPUT_DIR + '/Figures/26datasets/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
ontology_dir = OUTPUT_DIR + '/Ontology/CellOntology/'
dnames = ['muris_facs','microcebusBernard','microcebusMartine','muris_droplet','microcebusAntoine','microcebusStumpy']

scan_dim = 50
nn_nhidden = [100,50,25]
keep_prob = 0.7
if not os.path.isfile(output_dir+ 'integrated'+'.pretrained.pred_Y_all.npy'):
	last_l2i = {}
	last_i2l = {}
	for dname in dnames:
		unseen_ratio = 0.5
		feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data(dname, DATA_DIR,nsample=50000000)
		genes = [x.upper() for x in genes]
		train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = 0)
		train_X = np.vstack((train_X, test_X))
		train_Y_str = np.concatenate((train_Y_str, test_Y_str))
		exist_y = np.unique(train_Y_str)
		co_dim = 5
		ontology_emb_file = ontology_dir + str(co_dim)
		unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, co_dim = co_dim, use_pretrain = ontology_emb_file, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
		ncls = np.shape(cls2cls)[0]
		train_Y = MapLabel2CL(train_Y_str, l2i)
		unseen_l = MapLabel2CL(unseen_l, l2i)
		nunseen = len(unseen_l)
		nseen = ncls - nunseen
		ntrain = np.shape(train_X)[0]
		#print (np.shape(train_X), np.shape(train_Y))

		test_Y_pred = np.load(output_dir+dname+'.pretrained.pred_Y_all.npy')
		test_Y_pred = test_Y_pred.T / test_Y_pred.T.sum(axis=1)[:, np.newaxis]
		test_Y_pred = test_Y_pred.T

		#print (np.shape(test_Y_pred))
		if len(last_l2i)>0:
			new_ct_ind = []
			for i in range(len(last_i2l)):
				l = last_i2l[i]
				new_ct_ind.append(l2i[l])
			test_Y_pred = test_Y_pred[:,np.array(new_ct_ind)]
			all_test_Y_pred += test_Y_pred
		else:
			last_l2i = l2i
			last_i2l = i2l
			all_test_Y_pred = test_Y_pred

	l2i = last_l2i
	test_Y_pred = all_test_Y_pred
	np.save(output_dir+dname+'.pretrained.pred_Y_all.npy',test_Y_pred)
	np.save(output_dir+dname+'.pretrained.last_l2i.npy',last_l2i)
	np.save(output_dir+dname+'.pretrained.last_i2l.npy',last_i2l)
else:
	test_Y_pred = np.load(output_dir+'integrated'+'.pretrained.pred_Y_all.npy', allow_pickle=True)
	l2i = np.load(output_dir+'integrated'+'.pretrained.last_l2i.npy', allow_pickle=True).item()
	i2l = np.load(output_dir+'integrated'+'.pretrained.last_i2l.npy', allow_pickle=True).item()

exist_y = []
for dname in dnames:
	feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data(dname, DATA_DIR,nsample=50000000)
	exist_y.extend(np.unique(label))
exist_y = set(exist_y)
dname = 'integrated'
datasets, genes_list, n_cells = load_names(datanames_26datasets,verbose=False,log1p=True)
onto_ids, keywords, keyword2cname = map_26_datasets_cell2cid(use_detailed=False)

k_ind = np.array( [l2i[onto_ids[i]] for i,k in enumerate(keywords)] )
k2i = {}
for i,k in enumerate(keywords):
	k2i[k] = i

is_seen = []
aucs = []
ctypes = []
for onto_id, keyword in zip(onto_ids,keywords):
	if keyword2cname[keyword]=='PBMC': #PBMC is too general, so we exclude it.
		continue
	st_ind = 0
	labels = []
	pred = test_Y_pred[:,l2i[onto_id]]
	for i,dnames_raw in enumerate(datanames_26datasets):
		dnames = dnames_raw.split('/')[-1]
		ed_ind = st_ind + np.shape(datasets[i])[0]
		if dnames.startswith(keyword):
			labels.extend(np.ones(ed_ind - st_ind))
		else:
			labels.extend(np.zeros(ed_ind - st_ind))
		#pred.extend(test_Y_pred[st_ind:ed_ind,l2i[onto_id]])
		st_ind = ed_ind
	labels = np.array(labels)
	auc = roc_auc_score(labels, pred)
	print (keyword,auc,len(labels), onto_id in exist_y)
	aucs.append(auc)
	is_seen.append(int(onto_id in exist_y))
	ctypes.append(keyword2cname[keyword])
	#plot_26dataset_auroc(labels, pred, keyword2cname[keyword], known_y = onto_id in exist_y, fig_file = fig_dir + dname + '.' + keyword + '.pretrained.pdf')

is_seen = np.array(is_seen)
aucs = np.array(aucs)
ctypes = np.array(ctypes)
plot_26dataset_bar(ctypes, aucs, is_seen, output_file = fig_dir + dname + '.pretrained_test.rename.bar.pdf')
