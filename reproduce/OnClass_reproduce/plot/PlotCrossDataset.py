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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from plots import *



if len(sys.argv) <= 2:
	pid = 0
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

fig_dir = OUTPUT_DIR + '/Figures/CrossDataset/'
if not os.path.exists(fig_dir):
	os.makedirs(fig_dir)
output_dir =  OUTPUT_DIR + '/Crossdatasets/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ontology_dir = OUTPUT_DIR+'/Ontology/CellOntology/'

nn_nhidden = [100,50,25]
keep_prob = 0.7
unseen_ratio = 0.5
metrics = ['AUROC(seen)','AUPRC(seen)','AUROC','AUPRC','AUROC(unseen)', 'AUPRC(unseen)','Accuracy@3','Accuracy@5']
result_file = output_dir + 'auprc.result.txt'
if not os.path.isfile(result_file):
	fout = open(result_file,'w')
	pct = -1
	for dname1 in dnames:
		for dname2 in dnames:
			if dname1 == dname2:
				continue
			pct += 1
			if total_pid>1 and pct%total_pid != pid:
				continue

			feature1, label1, genes1, ontology_nlp_file, ontology_file = read_singlecell_data(dname1, DATA_DIR,nsample=50000000)
			print (np.unique(label1))

			co_dim = 5
			ontology_emb_file = ontology_dir + str(co_dim)
			unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(label1, co_dim = co_dim, use_pretrain = ontology_emb_file, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
			ncls = np.shape(cls2cls)[0]
			label1 = MapLabel2CL(label1, l2i)
			unseen_l = MapLabel2CL(unseen_l, l2i)
			nunseen = len(unseen_l)
			nseen = ncls - nunseen

			feature2, label2, genes2, ontology_nlp_file, ontology_file = read_singlecell_data(dname2, DATA_DIR,nsample=50000000)
			print (np.unique(label2))
			label2 = MapLabel2CL(label2, l2i)
			print (set(label1) - set(label2))
			print (len(set(label1) - set(label2)), len(set(label2) - set(label1)))


			common_genes = np.array(list(set(genes1) & set(genes2)))
			train_Y = label1
			test_Y = label2
			test_Y_ind = np.sort(np.array(list(set(test_Y) |  set(train_Y))))
			print ('ngenes:%d. seen: %d, ntrainY: %d, ntestY: %d, unseen: %d' % (len(common_genes), nseen, len(np.unique(train_Y)), len(np.unique(test_Y)), len(set(test_Y) - set(train_Y))))
			pred_Y_all = np.load(output_dir+dname1+'.'+dname2+'pred_Y_all.npy')
			res = evaluate(pred_Y_all, test_Y, unseen_l, nseen, metrics = metrics
			, Y_ind = test_Y_ind, write_to_file = fout, Y_net = onto_net, write_screen = True, prefix = dname1+'.'+dname2+'.'+str(len(set(test_Y) - set(train_Y)))+'.'+str(len(set(test_Y))))
	fout.close()

ndame = len(dnames)
mat2i = {}
for metric in metrics:
	mat2i[metric] = np.empty((ndame, ndame))
	mat2i[metric][:] = np.NaN
mat2i['Ratio of unseen cell types'] = np.empty((ndame, ndame))
mat2i['Ratio of unseen cell types'][:] = np.NaN
dname2i = {}
i2dname = {}
for i in range(ndame):
	dname2i[dnames[i]] = i
	i2dname[i] = dnames[i]
print (dname2i)
fin = open(result_file)
for line in fin:
	w = line.strip().split('\t')
	d1,d2,nunseen,ntest = w[0].split('.')
	nunseen_ratio = int(nunseen) * 1. / int(ntest)
	d1i = dname2i[d1]
	d2i = dname2i[d2]
	mat2i['Ratio of unseen cell types'][d1i,d2i] = nunseen_ratio
	for i, metric in enumerate(metrics):
		mat2i[metric][d1i,d2i] = float(w[i+1])

methods = []
for dname in dnames:
	methods.append(dname2keyword[dname])
metrics = list(mat2i.keys())
metrics.reverse()
for metric in metrics:
	heat_mat = mat2i[metric]
	print (fig_dir + metric + '.pdf')
	plot_heatmap_cross_dataset(heat_mat, methods=methods, file_name =fig_dir + metric.replace('\n',' ') + '_resize0.pdf', title=metric, ylabel=metric)
