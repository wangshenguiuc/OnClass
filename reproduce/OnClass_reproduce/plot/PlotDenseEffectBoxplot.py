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

def prepare_data(sps, aucs, cutoff=3):
	max_depths = int(np.max(sps))
	datas = []
	jitter_datas = []
	xlabels = []
	for i in range(min(sps),max_depths+1):
		ind = np.where(sps==i)[0]
		data = aucs[ind]
		#if len(data)<5:
		#	break
		datas.append(data)
		jitter_datas.append(data)
		xlabels.append(str(i)+'\nn='+str(len(data)))
		if i<=cutoff:
			print (i,np.mean(data))
	datas[cutoff] = np.concatenate(datas[cutoff:])
	jitter_datas[cutoff] = np.concatenate(jitter_datas[cutoff:])
	xlabels[cutoff] = '>'+str(cutoff-1)+'\nn='+str(len(datas[cutoff]))
	datas = datas[:cutoff+1]
	xlabels = xlabels[:cutoff+1]
	jitter_datas = jitter_datas[:cutoff+1]
	#for i in range(cutoff+1):
	#	print (i)
	#	jitter_datas[i] = np.random.choice(jitter_datas[i], min(200, len(jitter_datas[i])), replace=False)
	#print (datas)
	return datas, jitter_datas, xlabels

result_dir = OUTPUT_DIR + '/DenseEffect/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

fig_dir = OUTPUT_DIR + '/Figures/DenseEffect/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
dnames = ['microcebusBernard','microcebusStumpy','microcebusMartine','muris_facs','muris_droplet','microcebusAntoine']#

cutoffs = [2,3,4,5]
sps_all = {}
aucs_all = {}
for cutoff in cutoffs:
	sps_all[cutoff] = []
	aucs_all[cutoff] = []
for dname in dnames:
	sps = np.load(result_dir+dname+'region_sps.npy', allow_pickle = True).item()
	aucs = np.load(result_dir+dname+'region_aucs.npy', allow_pickle = True).item()
	for cutoff in cutoffs:
		sps_all[cutoff].extend(sps[cutoff])
		aucs_all[cutoff].extend(aucs[cutoff])

for cutoff in cutoffs:
	aucs = np.array(aucs_all[cutoff], dtype=float)
	sps = np.array(sps_all[cutoff], dtype=int)
	datas, jitter_datas, xlabels =  prepare_data(sps, aucs)
	plot_auc_region_violin(datas, jitter_datas, fig_file = fig_dir + str(cutoff)+'_region_all.pdf', xticks = xlabels, cutoff = cutoff)
	print ('done')
