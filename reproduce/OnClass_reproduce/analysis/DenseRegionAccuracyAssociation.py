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
import string
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn import metrics
from collections import Counter
from scipy.special import softmax
from scipy.stats import norm as dist_model
from sklearn import preprocessing
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.preprocessing import normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from plots import *

if len(sys.argv) <= 2:
	pid = 0
	total_pid = -1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

ontology_dir =   OUTPUT_DIR+'/Ontology/CellOntology/'
result_dir = OUTPUT_DIR + '/DenseEffect/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


output_dir =  OUTPUT_DIR + '/CompareBaselines/baseline_scores/'
our_output_dir = OUTPUT_DIR + '/CompareBaselines/TuneOnclass/'
dnames = ['microcebusBernard','microcebusStumpy','microcebusMartine','muris_droplet','microcebusAntoine','muris_facs']#'microcebusAntoine',

metrics = ['AUROC', 'AUPRC', 'Accuracy@3', 'Accuracy@5', 'AUPRC(unseen)', 'AUROC(unseen)']
unseen_ratios = [0.1,0.3,0.5,0.7,0.9]
niter = 5
nn_nhidden = [100,50,25]
keep_prob = 0.7
KNN = 0.7
cutoffs = [2,3,4,5]
pct = -1

for dname in dnames:
	pct += 1
	if total_pid>1 and pct%total_pid != pid:
		continue
	aucs = {}
	sps = {}
	for cutoff in cutoffs:
		sps[cutoff] = []
		aucs[cutoff] = []
	fig_dir = OUTPUT_DIR + '/Figures/BarPlotsAll/' + dname + '/'
	figure_para = str(KNN)
	print (dname, figure_para)
	if not os.path.exists(fig_dir):
		os.makedirs(fig_dir)
	for iter in range(niter):
		if dname == 'microcebusMartine' and iter==1: # microcebusMartine only has 1 class in the training data when unseen_ratio=0.9, so we only average 1 iterations here.
			continue
		for unseen_ratio in unseen_ratios:
			output_prefix = output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio)+'_'
			print (output_prefix)
			npzfile = np.load(output_prefix + 'truth_vec.npz',allow_pickle=True)
			test_Y_ind = npzfile['test_Y_ind']
			onto_net = npzfile['onto_net'].item()
			nseen = npzfile['nseen']
			unseen_l = npzfile['unseen_l']
			l2i = npzfile['l2i'].item()
			i2l = npzfile['i2l'].item()
			cls2cls = npzfile['cls2cls']
			test_Y = npzfile['test_Y']
			ntest = len(test_Y)
			ncls = nseen + len(unseen_l)
			seen_l = np.array(range(nseen))

			cls2cls = np.zeros((ncls, ncls))
			fin = open(DATA_DIR+'/cell_ontology/cl.ontology')
			for line in fin:
				w = line.strip().split('\t') #w[1] is parent of w[0]
				cls2cls[int(l2i[w[0]]), int(l2i[w[1]])] = 1
				cls2cls[int(l2i[w[1]]), int(l2i[w[0]])] = 1
			fin.close()
			sp = graph_shortest_path(cls2cls,method='FW',directed =False)

			pname = translate_paramter([nn_nhidden,keep_prob,KNN])
			pred_Y_all = np.load(our_output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio) + '/'+ pname + 'pred_Y_all.npy')
			#res = evaluate(pred_Y_all, test_Y, unseen_l, nseen, Y_ind = test_Y_ind, Y_net = onto_net, write_screen = False, metrics = metrics, prefix = str(KNN))
			Y_truth_bin_mat = ConvertLabels(test_Y, ncls)
			class_auc_macro = np.full(ncls, np.nan)
			class_auprc_macro = np.full(ncls, np.nan)

			for i in unseen_l:
				if len(np.unique(Y_truth_bin_mat[:,i]))==2:
					class_auc_macro[i] = roc_auc_score(Y_truth_bin_mat[:,i], pred_Y_all[:,i])
					for cutoff in cutoffs:
						select_ctps = len(np.where(sp[i,seen_l]<=cutoff)[0])
						aucs[cutoff].append(class_auc_macro[i])
						sps[cutoff].append(select_ctps)
	np.save(result_dir+dname+'region_sps.npy', sps)
	np.save(result_dir+dname+'region_aucs.npy', aucs)
