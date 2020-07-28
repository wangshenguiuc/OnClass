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

def prepare_data(sps, aucs, cutoff=5):
	max_depths = int(np.max(sps))
	datas = []
	jitter_datas = []
	xlabels = []
	for i in range(2,max_depths+1):
		ind = np.where(sps==i)[0]
		data = aucs[ind]
		#if len(data)<5:
		#	break
		datas.append(data)
		jitter_datas.append(data)
		xlabels.append(str(i)+'\nn='+str(len(data)))
	datas[cutoff] = np.concatenate(datas[cutoff:])
	jitter_datas[cutoff] = np.concatenate(jitter_datas[cutoff:])
	xlabels[cutoff] = '>'+str(cutoff+1)+'\nn='+str(len(datas[cutoff]))
	datas = datas[:cutoff+1]
	xlabels = xlabels[:cutoff+1]
	jitter_datas = jitter_datas[:cutoff+1]
	#print (datas)
	return datas, jitter_datas, xlabels

result_dir = OUTPUT_DIR + '/DenseEffect/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

fig_dir = OUTPUT_DIR + '/Figures/DenseEffect/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
dnames = ['muris_facs','microcebusBernard','microcebusStumpy','microcebusMartine','muris_droplet','microcebusAntoine']#

sps_all = []
aucs_all = []
for dname in dnames:
	sps = np.load(result_dir+dname+'_sps.npy')
	aucs = np.load(result_dir+dname+'_aucs.npy')
	sps_all.extend(sps)
	aucs_all.extend(aucs)
	datas, jitter_datas, xlabels =  prepare_data(sps, aucs)
	plot_auc_shortest_distance_boxplot(datas, jitter_datas, fig_file = fig_dir + dname + '.pdf', xticks = xlabels)

aucs_all = np.array(aucs_all, dtype=float)
sps_all = np.array(sps_all, dtype=int)
datas, jitter_datas, xlabels =  prepare_data(sps_all, aucs_all)
plot_auc_shortest_distance_boxplot(datas, jitter_datas, fig_file = fig_dir + 'all.pdf', xticks = xlabels)
