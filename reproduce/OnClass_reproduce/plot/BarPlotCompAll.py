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

#dnames = ['microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy',']
dnames = ['muris_facs','muris_droplet','allen','microcebusAntoine','microcebusBernard','microcebusStumpy','microcebusMartine']#
method2name = {'lr':'LR','actinn':'ACTINN','svm':'SVM','lr_reject':'LR(reject)','doc':'DOC(reject)','svm_reject':'SVM(reject)','singlecellnet':'sCN(reject)'}
metrics = ['AUROC', 'AUPRC', 'Accuracy@3', 'Accuracy@5', 'AUROC(unseen)', 'AUPRC(unseen)']
methods = ['LR','SVM','ACTINN','LR(reject)','sCN(reject)','SVM(reject)','DOC(reject)','OnClass']
unseen_methods = ['LR(reject)','sCN(reject)','SVM(reject)','DOC(reject)','OnClass']
run_methods = ['lr','actinn','svm','lr_reject','doc','svm_reject','singlecellnet']
lab2col, lab2marker = generate_colors(methods)
unseen_ratios = [0.1,0.3,0.5,0.7,0.9]
niter = 5
score = {}
for method in methods:
	score[method] = {}
	for metric in metrics:
		score[method][metric] = {}
		for unseen_ratio in unseen_ratios:
			score[method][metric][unseen_ratio] = []

nn_nhidden = [100,50,25]
keep_prob = 0.7
start = timeit.default_timer()
pct = -1
for KNN in [0.7]:
	for dname in dnames:
		pct += 1
		if total_pid>1 and pct%total_pid != pid:
			continue
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
				#print (output_prefix)
				npzfile = np.load(output_prefix + 'truth_vec.npz',allow_pickle=True)
				test_Y_ind = npzfile['test_Y_ind']
				onto_net = npzfile['onto_net'].item()
				nseen = npzfile['nseen']
				unseen_l = npzfile['unseen_l']
				cls2cls = npzfile['cls2cls']
				test_Y = npzfile['test_Y']
				ntest = len(test_Y)
				ncls = nseen + len(unseen_l)

				for method in run_methods:
					pred_Y_all = np.zeros((ntest, ncls))
					if not os.path.isfile(output_prefix+method+'_mat'):
						print (output_prefix+method+'_mat', 'not found')
						continue
					print (output_prefix+method+'_mat')
					pred_Y_all[:,:nseen] = np.loadtxt(output_prefix+method+'_mat')
					if method in ['lr_reject','doc','svm_reject','singlecellnet']:
						vec = np.loadtxt(output_prefix+method+'_vec')
						pred_Y_all = ImputeUnseenCls(vec, pred_Y_all, cls2cls, nseen, knn=1)
					res = evaluate(pred_Y_all, test_Y, unseen_l, nseen, Y_ind = test_Y_ind, Y_net = onto_net, write_screen = False, prefix = method, metrics = metrics)
					for metric in res:
						score[method2name[method]][metric][unseen_ratio].append(res[metric])

				pname = translate_paramter([nn_nhidden,keep_prob,KNN])
				if not os.path.isfile(our_output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio) + '/'+ pname + 'pred_Y_all.npy'):
					print (our_output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio) + '/'+ pname + 'pred_Y_all.npy', 'not found')
					continue
				pred_Y_all = np.load(our_output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio) + '/'+ pname + 'pred_Y_all.npy')
				res = evaluate(pred_Y_all, test_Y, unseen_l, nseen, Y_ind = test_Y_ind, Y_net = onto_net, write_screen = False, metrics = metrics, prefix = str(KNN))
				for metric in res:
					score['OnClass'][metric][unseen_ratio].append(res[metric])

		nratio = len(unseen_ratios)
		group_l =  np.array(unseen_ratios) * 100
		group_l = [str(int(x))+'%' for x in group_l]
		ncols = 2
		nrows = int(len(metrics)/ncols)

		plt.clf()
		fig, axs = plt.subplots(int(len(metrics)/ncols),ncols,figsize=(6, 4))
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
			for clsi, unseen_ratio in enumerate(unseen_ratios):
				for methodi, method in enumerate(cur_methods):
					error[clsi, methodi] = np.std(score[method][metric][unseen_ratio]) / np.sqrt(len(score[method][metric][unseen_ratio]))
					mean[clsi, methodi] = np.mean(score[method][metric][unseen_ratio])
			if np.floor(mi / ncols) == 0:
				write_title = False
			else:
				write_title = False
			if np.floor(mi / ncols) == (nrows-1):
				write_xlabel = True
			else:
				write_xlabel = False
			axs[mi] = plot_comparison_baselines_bar(axs[mi], mean, error, group_l, cur_methods, write_title = write_title, write_xlabel = write_xlabel, fig_dir = 'test', title=dname, ylabel=metric,lab2col=lab2col)
			axs[mi].text(-0.2, 1.15, string.ascii_lowercase[mi], transform=axs[mi].transAxes,size=8, weight='bold')
		#axs[0] = plot_comparison_baselines_bar_legend(axs[1], mean, error, group_l, cur_methods, write_title = write_title, write_xlabel = write_xlabel, fig_dir = output_file, title=dname, ylabel=metric,lab2col=lab2col)
		legend_keys = []
		legend_handles,legend_labels =  axs[0].get_legend_handles_labels()
		fig.tight_layout()

		#plt.title(dname)
		'''
		fig.legend(legend_handles,	 # The line objects
		   labels=legend_labels,   # The labels for each line
		   loc="upper center",   # Position of legend
		   borderaxespad=0,	# Small spacing around legend box
		   ncol=len(legend_labels), frameon=False)
		'''
		#
		output_file = fig_dir + dname2keyword[dname] + figure_para
		plt.savefig(output_file+'.pdf')
