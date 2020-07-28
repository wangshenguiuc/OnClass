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
from plots import *
from libs import my_assemble, datanames_26datasets

def calculate_aucs(l2i, test_Y_pred):
	k_ind = np.array( [l2i[onto_ids[i]] for i,k in enumerate(keywords)] )
	k2i = {}
	for i,k in enumerate(keywords):
		k2i[k] = i
	aucs = []
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
		#print (keyword,auc,len(labels), onto_id in exist_y)
		aucs.append(auc)
	#auc = np.mean(aucs)
	return aucs

output_dir = OUTPUT_DIR + '/26datasets/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
fig_dir = OUTPUT_DIR + '/Figures/26datasets/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
ontology_dir =   OUTPUT_DIR+'/Ontology/CellOntology/'

datasets, genes_list, n_cells = load_names(datanames_26datasets,verbose=False,log1p=True)
onto_ids, keywords, keyword2cname = map_26_datasets_cell2cid(use_detailed=False)
scan_dim = 50
nn_nhidden = [100,50,25]
keep_prob = 0.7

npz_file = output_dir + 'scores_rename.npz'
if  not os.path.isfile(npz_file):
	test_Y_pred = np.load(output_dir+'integrated'+'.pretrained.pred_Y_all.npy', allow_pickle=True)
	l2i = np.load(output_dir+'integrated'+'.pretrained.last_l2i.npy', allow_pickle=True).item()
	i2l = np.load(output_dir+'integrated'+'.pretrained.last_i2l.npy', allow_pickle=True).item()
	all_auc = []
	methods = []
	auc = calculate_aucs(l2i, test_Y_pred)
	all_auc.append(auc)
	methods.append('All')
	last_l2i = {}
	last_i2l = {}
	for dname in dnames:
		unseen_ratio = 0.5
		feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data(dname, data_dir,nsample=50000000)
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

		test_Y_pred = np.load(output_dir+dname+'.pretrained.pred_Y_all.npy', allow_pickle=True)
		test_Y_pred = test_Y_pred.T / test_Y_pred.T.sum(axis=1)[:, np.newaxis]
		test_Y_pred = test_Y_pred.T

		#print (np.shape(test_Y_pred))
		auc = calculate_aucs(l2i, test_Y_pred)
		methods.append(dname2keyword[dname])
		all_auc.append(auc)
	np.savez(npz_file, all_auc=all_auc, methods=methods)
else:
	npzfile = np.load(npz_file,allow_pickle=True)
	all_auc = npzfile['all_auc']
	methods = npzfile['methods']
print (methods)
new_dnames = ['All','Muris FACS','Muris droplet','Lemur 1','Lemur 2','Lemur 3','Lemur 4']
new_aucs = []
for dname in new_dnames:
	print (dname, methods)
	print (np.where(methods == dname))
	print (np.where(methods == dname)[0])
	id = np.where(methods == dname)[0][0]
	print (id, methods, dname)
	new_aucs.append(all_auc[id])
errors = []
aucs = []
for sc in new_aucs:
	print (sc)
	errors.append(np.std(sc) / np.sqrt(len(sc)))
	aucs.append(np.mean(sc))
print (dnames,aucs,errors)
plot_26dataset_more_data_bar(new_dnames, aucs, errors, output_file = fig_dir + 'all.pretrained.bar.pdf')
