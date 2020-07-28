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
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/OnClass/'
data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/'
sys.path.append(repo_dir)
sys.path.append(repo_dir+'src/task/OnClass/')
sys.path.append(repo_dir+'src/task/OnClass/model/')
os.chdir(repo_dir)
from model.OnClassPred import OnClassPred
from model.BilinearNNDebug import BilinearNN
from utils import *
from plots import *
from libs import my_assemble, datanames_26datasets

if len(sys.argv) <= 2:
	pid = 0
	total_pid = -1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])



dnames = ['muris_facs','muris_droplet','microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy','allen']
for dname in dnames:
	feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data(dname, data_dir,nsample=50000000)
	print (dname, len(np.unique(label)), len(np.unique(genes)), np.shape(feature))
