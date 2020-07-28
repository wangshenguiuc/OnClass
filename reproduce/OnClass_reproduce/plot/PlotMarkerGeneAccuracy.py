import sys
from time import time
from scipy import stats, sparse
from scipy.sparse.linalg import svds, eigs
from scipy.special import expit
import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,precision_score,precision_recall_fscore_support, cohen_kappa_score, average_precision_score,f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn import metrics
from collections import Counter
from scipy.stats import norm as dist_model
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from model.BilinearNNDebug import BilinearNN
from plots import *

if len(sys.argv) <= 2:
	pid = 0
	total_pid = -1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])



output_dir = OUTPUT_DIR + '/MarkerGenes/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fig_dir = OUTPUT_DIR + '/Figures/MarkerGenes/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
ontology_dir =   OUTPUT_DIR+'/Ontology/CellOntology/'
#read TMS
nn_nhidden = [100,50,25]
keep_prob = 0.7
unseen_ratio = 0.5

score = {}
for dname in dnames:
	score[dname] = {}
	for method in ['Seen cell types', 'Unseen cell types']:
		score[dname][method] = []

npz_file = output_dir + 'marker_gene.npz'
if not os.path.isfile(npz_file):
	for dname in dnames:
		print (dname)
		feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data(dname, DATA_DIR,nsample=50000000)
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

		#train_X = process_expression([train_X])
		train_X = np.log1p(train_X)

		g2i = {}
		for i,g in enumerate(genes):
			g2i[g] = i
		tp2genes_base = read_type2genes(g2i)
		co2name, name2co = get_ontology_name(DATA_DIR = DATA_DIR)
		terms = [i2l[i] for i in range(len(i2l))]
		if os.path.isfile(output_dir + dname + '.npy'):
			cor = np.load(output_dir + dname + '.npy')
		else:
			pred_Y_all = np.load(output_dir+dname+'_pred_Y_all.npy')
			cor = corr2_coeff(pred_Y_all[:,:].T, train_X[:,:].T)
			cor = np.nan_to_num(cor)
			np.save(output_dir + dname + '.npy', cor)

		thres = 0.4
		topk = 50
		sub_cor = np.where(np.abs(cor)>thres, 1, 0)
		seen_aucs = []
		unseen_aucs = []
		recall = []
		precision = []
		for tp in tp2genes_base:
			label = []
			pred = cor[l2i[tp],:]
			for g in genes:
				label.append(g in tp2genes_base[tp])
			auc = roc_auc_score(label, pred)
			if l2i[tp]<nseen:
				seen_aucs.append(auc)
			else:
				unseen_aucs.append(auc)
		score[dname]['Seen cell types'] = seen_aucs
		score[dname]['Unseen cell types'] = unseen_aucs
	np.savez(npz_file, score = score)
else:
	npzfile = np.load(npz_file,allow_pickle=True)
	score = npzfile['score'].item()

metrics = ['Unseen cell types','Seen cell types']
ndname = len(dnames)
nmetric = len(metrics)
mean = np.zeros((ndname, nmetric))
error = np.zeros((ndname, nmetric))
group_l = []
for i in range(ndname):
	group_l.append(dname2keyword[dnames[i]])
	for j in range(nmetric):
		print (i,j)
		mean[i,j] = np.mean(score[dnames[i]][metrics[j]])
		error[i,j] = np.std(score[dnames[i]][metrics[j]]) / np.sqrt(len(score[dnames[i]][metrics[j]]))
print (group_l)
plot_marker_comparison_baselines_bar(mean, error, group_l = group_l, method_l = metrics,  output_file = fig_dir + 'barplot_rename.pdf', ylabel='AUROC',lab2col={'Seen cell types':'g','Unseen cell types':'y'})
