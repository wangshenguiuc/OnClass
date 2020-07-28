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
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/OnClass/'
data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/'
sys.path.append(repo_dir)
sys.path.append(repo_dir+'src/task/OnClass/')
sys.path.append(repo_dir+'src/task/OnClass/model/')
os.chdir(repo_dir)
from model.OnClassPred import OnClassPred
from model.BilinearNNDebug import BilinearNN
from utils import *
from libs import my_assemble, datanames_26datasets
from plots import *



if len(sys.argv) <= 2:
	pid = 0
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

fig_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/Figures/CrossDataset/'
if not os.path.exists(fig_dir):
	os.makedirs(fig_dir)
output_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/Crossdatasets/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
ontology_dir =  '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/Ontology/CellOntology/'
#read TMS
co2name, name2co = get_ontology_name(DATA_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/')
l2i = {}
i2l = {}
for i,co in enumerate(co2name):
	l2i[co] = i
	i2l[i] = co
ncls = len(l2i)
cls2cls = np.zeros((ncls, ncls))
child2parent = np.zeros((ncls, ncls))
parent2child = np.zeros((ncls, ncls))
fin = open('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology')
for line in fin:
	w = line.strip().split('\t') #w[1] is parent of w[0]
	cls2cls[l2i[w[0]], l2i[w[1]]] = 1
	cls2cls[l2i[w[1]], l2i[w[0]]] = 1
	child2parent[l2i[w[0]], l2i[w[1]]] = 1
	parent2child[l2i[w[1]], l2i[w[0]]] = 1
fin.close()

all = 0
same = 0
for parent in range(ncls):
	childs = np.where(parent2child[parent,:]!=0)[0]
	#print (childs)
	for c1 in childs:
		for c2 in childs:
			if c1>=c2:
				continue
			all += 1
			v1 = cls2cls[c1,:][:]
			v2 = cls2cls[c2,:][:]
			v1[c2] = 0
			v2[c1] = 0
			#print (c1,c2,np.sum(np.abs(v1-v2)),np.sum(v1))
			if np.sum(np.abs(v1-v2)) == 0 and np.sum(v1)>0:
				same += 1
print (same, all, same/all)
