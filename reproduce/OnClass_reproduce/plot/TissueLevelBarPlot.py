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
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA
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



output_dir = OUTPUT_DIR + '/CompareBaselines/baseline_scores/'
our_output_dir = OUTPUT_DIR + '/CompareBaselines/TuneOnclass/'

#dnames = ['muris_facs','muris_droplet',]
dnames = ['muris_facs','muris_droplet','microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy']#
#dnames = ['allen']
method2name = {'lr':'LR','actinn':'ACTINN','svm':'SVM','lr_reject':'LR(reject)','doc':'DOC(reject)','svm_reject':'SVM(reject)','singlecellnet':'sCN(reject)'}
metrics = ['AUROC', 'AUPRC', 'Accuracy@3', 'Accuracy@5', 'AUROC(unseen)', 'AUPRC(unseen)']
methods = ['LR','SVM','ACTINN','LR(reject)','sCN(reject)','SVM(reject)','DOC(reject)','OnClass']
unseen_methods = ['LR(reject)','sCN(reject)','SVM(reject)','DOC(reject)','OnClass']
run_methods = ['lr','actinn','svm','lr_reject','doc','svm_reject','singlecellnet']
lab2col, lab2marker = generate_colors(methods)

niter = 5
score = {}
for method in methods:
	score[method] = {}
	for metric in metrics:
		score[method][metric] = {}
unseen_ratio = 0.5
nn_nhidden = [100,50,25]
keep_prob = 0.7
start = timeit.default_timer()
pct = -1
for KNN in [0.7]:
	for dname in dnames:
		pct += 1
		if total_pid>1 and pct%total_pid != pid:
			continue
		fig_dir = OUTPUT_DIR + '/Figures/BarPlotsTissue/' + dname + '/'
		figure_para = str(KNN)
		if 'muris' in dname:
			file = DATA_DIR + 'TMS_official_060520/' + 'tabula-muris-senis-'+dname.split('_')[1]+'-official-raw-obj.h5ad'
		else:
			file = DATA_DIR + 'TMS_official_060520/' + dname +'.h5ad'
		x = read_h5ad(file)
		tissues = np.unique(x.obs['tissue'].tolist())
		if not os.path.exists(fig_dir):
			os.makedirs(fig_dir)
		for iter in range(niter):
			if dname == 'microcebusMartine' and iter==1: # microcebusMartine only has 1 class in the training data when unseen_ratio=0.9, so we only average 1 iterations here.
				continue

			output_prefix = output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio)+'_'
			print (output_prefix)
			npzfile = np.load(output_prefix + 'truth_vec_tissues.npz',allow_pickle=True)
			
			test_Y_ind = npzfile['test_Y_ind']
			onto_net = npzfile['onto_net'].item()
			nseen = npzfile['nseen']
			unseen_l = npzfile['unseen_l']
			test_tissue = npzfile['test_tissue']
			cls2cls = npzfile['cls2cls']
			test_Y = npzfile['test_Y']
			ntest = len(test_Y)
			ncls = nseen + len(unseen_l)
			for tissue in tissues:
				#if tissue == 'Liver':
				#	break
				tis_ind = np.where(test_tissue == tissue)[0]
				if len(tis_ind) < 10 or len(np.unique(test_Y[tis_ind])) < 2:
					continue

				print (tissue, len(tis_ind),len(np.unique(test_Y[tis_ind])))
				for method in score:
					for metric in score[method]:
						if tissue not in score[method][metric]:
							score[method][metric][tissue] = []
				for method in run_methods:
					pred_Y_all = np.zeros((ntest, ncls))
					if not os.path.isfile(output_prefix+method+'_mat'):
						print (output_prefix+method+'_mat', 'not found')
						continue

					pred_Y_all[:,:nseen] = np.loadtxt(output_prefix+method+'_mat')
					if method in ['lr_reject','doc','svm_reject','singlecellnet']:
						vec = np.loadtxt(output_prefix+method+'_vec')
						pred_Y_all = ImputeUnseenCls(vec, pred_Y_all, cls2cls, nseen, knn=1)
					res = evaluate(pred_Y_all[tis_ind,:], test_Y[tis_ind], unseen_l, nseen, Y_ind = test_Y_ind, Y_net = onto_net, write_screen = False, prefix = method, metrics = metrics)
					for metric in res:
						score[method2name[method]][metric][tissue].append(res[metric])

				pname = translate_paramter([nn_nhidden,keep_prob,KNN])
				if not os.path.isfile(our_output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio) + '/'+ pname + 'pred_Y_all.npy'):
					print (our_output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio) + '/'+ pname + 'pred_Y_all.npy', 'not found')
					continue
				pred_Y_all = np.load(our_output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio) + '/'+ pname + 'pred_Y_all.npy')
				res = evaluate(pred_Y_all[tis_ind,:], test_Y[tis_ind], unseen_l, nseen, Y_ind = test_Y_ind, Y_net = onto_net, write_screen = False, metrics = metrics, prefix = str(KNN))
				for metric in res:
					score['OnClass'][metric][tissue].append(res[metric])
		tissues = list(score['OnClass'][metric].keys())
		ntissue = len(tissues)
		nratio = ntissue
		#group_l = tissues
		group_l = np.array([tis.replace('_','\n').lower().capitalize() for tis in tissues])
		ncols = 1
		nrows = int(len(metrics)/ncols)
		fig, axs = plt.subplots(int(len(metrics)/ncols),ncols,figsize=(ntissue*ncols,(len(metrics)*3+2)/ncols))
		axs = axs.flat
		#for n, ax in enumerate(axs):
		for mi, metric in enumerate(metrics):
			if 'unseen' in metric:
				cur_methods = unseen_methods
			else:
				cur_methods = methods
			nmethod = len(cur_methods)
			error = np.zeros((nratio, nmethod))
			mean = np.zeros((nratio, nmethod))
			for clsi, tissue in enumerate(tissues):
				for methodi, method in enumerate(cur_methods):
					#print (score[method][metric][unseen_ratio])
					error[clsi, methodi] = np.nanstd(score[method][metric][tissue]) / np.sqrt(len(score[method][metric][tissue]))
					mean[clsi, methodi] = np.nanmean(score[method][metric][tissue])
				#for method in methods:
				#	pv = stats.ttest_rel(score[method][metric][unseen_ratio],score['OnClass'][metric][unseen_ratio])[1] / 2.
				#	if pv > 0.05 and method!='OnClass':
				#		print ('%s %s %f %e' % (metric, method, unseen_ratio, pv))

			if np.floor(mi / ncols) == 0:
				write_title = False
			else:
				write_title = False
			if np.floor(mi / ncols) == (nrows-1):
				write_xlabel = False
			else:
				write_xlabel = False
			print (mean)
			print (error)
			axs[mi] = plot_comparison_baselines_bar(axs[mi], mean, error, group_l, cur_methods, write_title = write_title, write_xlabel = write_xlabel, fig_dir = 'test', title=dname, ylabel=metric,lab2col=lab2col)
			axs[mi].text(0.05, 1.15, string.ascii_lowercase[mi], transform=axs[mi].transAxes,size=12, weight='bold')
		#axs[0] = plot_comparison_baselines_bar_legend(axs[1], mean, error, group_l, cur_methods, write_title = write_title, write_xlabel = write_xlabel, fig_dir = output_file, title=dname, ylabel=metric,lab2col=lab2col)
		legend_keys = []
		legend_handles,legend_labels =  axs[0].get_legend_handles_labels()
		fig.tight_layout()
		#plt.title(dname, fontsize=20)
		fig.legend(legend_handles,     # The line objects
           labels=legend_labels,   # The labels for each line
           loc="upper center",   # Position of legend
           borderaxespad=0,    # Small spacing around legend box
		   ncol=len(legend_labels), frameon=False,fontsize=6)
		#
		output_file = fig_dir + dname + figure_para
		plt.savefig(output_file+'.pdf')
