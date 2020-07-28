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

fig_dir = OUTPUT_DIR + '/Figures/MarkerGenes/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

output_dir = OUTPUT_DIR + '/MarkerGenes/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
ontology_dir =   OUTPUT_DIR+'/Ontology/CellOntology/'



nn_nhidden = [100,50,25]
keep_prob = 0.7
unseen_ratio = 0.5
#dnames = ['muris_facs','muris_droplet']
npz_file = output_dir + 'heat_mat.npz'
if not os.path.isfile(npz_file):
	has_truth_our_auc_mat = collections.defaultdict(dict)
	has_truth_base_auc_mat = collections.defaultdict(dict)
	no_truth_our_auc_mat = collections.defaultdict(dict)
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
		assert(np.shape(cor)[0] == len(terms1))
		assert(np.shape(cor)[1] == len(genes1))
		g2i1 = {}
		i2g1 = {}
		for i,g in enumerate(genes1):
			g2i1[g] = i
			i2g1[i] = g
		tp2genes_base = read_type2genes(g2i1)
		exist_y = np.unique(train_Y_str)
		for dnamej, dname2 in enumerate(dnames):
			if  dname1 == dname2:
				continue
			feature, label, genes2, ontology_nlp_file, ontology_file = read_singlecell_data(dname2, data_dir,nsample=50000000)
			train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = 0)
			train_X = np.vstack((train_X, test_X))
			train_Y_str = np.concatenate((train_Y_str, test_Y_str))
			co_dim = 5
			ontology_emb_file = ontology_dir + str(co_dim)
			unseen_l, l2i2, i2l2, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, co_dim = co_dim, use_pretrain = ontology_emb_file, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
			ncls = np.shape(cls2cls)[0]
			train_Y = MapLabel2CL(train_Y_str, l2i2)
			train_X2Y = ConvertLabels(np.array(train_Y,dtype=int), ncls)
			unseen_l = MapLabel2CL(unseen_l, l2i2)
			nunseen = len(unseen_l)
			nseen = ncls - nunseen
			terms2 = [i2l2[i] for i in range(len(i2l2))]
			g2i2 = {}
			for i,g in enumerate(genes2):
				g2i2[g] = i
			common_genes = list(set(genes1) & set(genes2))

			has_truth_our_auc = []
			has_truth_tp_base_auc = []
			no_truth_our_auc = []
			topk = 50
			thres = 0.4
			sub_cor = np.where(np.abs(cor)>thres, 1, 0)
			nmarkers = []
			for i in range(nseen):
				tp = i2l2[i]
				our_marker = np.where(sub_cor[l2i1[tp],:]==1)[0]#np.argsort(cor[l2i1[tp],:]*-1)[:topk]
				our_marker_id = [g2i2[i2g1[gi]] for gi in our_marker if i2g1[gi] in g2i2]
				nmarkers.append(len(our_marker_id))
				if len(our_marker_id) ==0:
					our_marker = np.argsort(cor[l2i1[tp],:]*-1)[:topk]
					our_marker_id = [g2i2[i2g1[gi]] for gi in our_marker if i2g1[gi] in g2i2]
				Y_true = train_X2Y[:,i]
				Y_pred_our = np.sum(train_X[:, our_marker_id], axis=1)
				our_auc = roc_auc_score(Y_true, Y_pred_our)
				if tp in tp2genes_base and len([g2i2[g] for g in tp2genes_base[tp] if g in g2i2])!=0:
					base_marker = tp2genes_base[tp]
					base_marker_id = [g2i2[g] for g in base_marker if g in g2i2]
					if len(base_marker_id) == 0:
						continue
					Y_pred_base = np.sum(train_X[:, base_marker_id], axis=1)
					base_auc = roc_auc_score(Y_true, Y_pred_base)#roc_auc_score(Y_true, Y_pred_base)
					has_truth_our_auc.append(our_auc)
					has_truth_tp_base_auc.append(base_auc)
				else:
					no_truth_our_auc.append(our_auc)
			pv = stats.ttest_rel(has_truth_our_auc, has_truth_tp_base_auc)[1] / 2.
			print ('%f %s %s %f seen(our,base,length,pv):%f %f %d %d %e' % (thres, dname1,dname2,np.mean(no_truth_our_auc),
			np.mean(has_truth_our_auc), np.mean(has_truth_tp_base_auc), len(has_truth_tp_base_auc), np.median(nmarkers), pv))
			has_truth_our_auc_mat[dname1][dname2] = has_truth_our_auc
			has_truth_base_auc_mat[dname1][dname2] = has_truth_tp_base_auc
			no_truth_our_auc_mat[dname1][dname2] = no_truth_our_auc
	np.savez(npz_file, has_truth_our_auc_mat = has_truth_our_auc_mat, has_truth_base_auc_mat = has_truth_base_auc_mat, no_truth_our_auc_mat = no_truth_our_auc_mat)
else:
	npzfile = np.load(npz_file,allow_pickle=True)
	has_truth_our_auc_mat = npzfile['has_truth_our_auc_mat'].item()
	has_truth_base_auc_mat = npzfile['has_truth_base_auc_mat'].item()
	no_truth_our_auc_mat = npzfile['no_truth_our_auc_mat'].item()


ndname = len(dnames)
heat_mat = np.zeros((ndname,ndname))
group_l = []
for i,dname1 in enumerate(dnames):
	group_l.append(dname2keyword[dname1])
	for j,dname2 in enumerate(dnames):
		if dname1 == dname2:
			continue
		heat_mat[i,j] = np.mean(no_truth_our_auc_mat[dname1][dname2])
plot_heatmap_cross_dataset(heat_mat, group_l, file_name = fig_dir + 'no_truth.pdf', ylabel = 'AUROC', title='AUROC')


metrics = ['OnClass referred marker gene','Curated marker gene']
ndname = len(dnames)-1
nmetric = len(metrics)

for dname1 in dnames:
	group_l = []
	mean = np.zeros((ndname, nmetric))
	error = np.zeros((ndname, nmetric))
	for dname2 in dnames:
		if dname2 == dname1:
			continue
		j = len(group_l)
		mean[j,0] = np.mean(has_truth_our_auc_mat[dname1][dname2])
		error[j,0] = np.std(has_truth_our_auc_mat[dname1][dname2]) / np.sqrt(len(has_truth_our_auc_mat[dname1][dname2]))
		mean[j,1] = np.mean(has_truth_base_auc_mat[dname1][dname2])
		error[j,1] = np.std(has_truth_base_auc_mat[dname1][dname2]) / np.sqrt(len(has_truth_base_auc_mat[dname1][dname2]))
		group_l.append(dname2keyword[dname2])
	plot_marker_comparison_prediction_accuracy_bar(mean, error, group_l = group_l, method_l = metrics,  output_file = fig_dir + dname1+'.barplot.resize.pdf', ylabel='AUROC',lab2col={'OnClass referred marker gene':'#D7191C','Curated marker gene':'#2C7BB6'})
