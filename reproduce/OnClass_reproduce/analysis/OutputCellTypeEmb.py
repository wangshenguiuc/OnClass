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
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn import metrics
from collections import Counter
from sklearn.preprocessing import normalize
from scipy.stats import norm as dist_model
from scipy.sparse import csr_matrix
from sklearn import preprocessing
import umap
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from scipy.sparse.linalg import svds, eigs
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

output_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/26datasets/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
fig_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/Figures/EmbeddingAnalyze/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
ontology_dir =  '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/Ontology/CellOntology/'
#read TMS


file = data_dir + 'TMS_official_060520/tabula-muris-senis-facs-official-raw-obj.h5ad'
dim = 10
feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data('muris_facs', data_dir,nsample=50000000)
co_dim = 5
ontology_emb_file = ontology_dir + str(co_dim)
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(label, co_dim = co_dim, use_pretrain = ontology_emb_file, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
onto_net_rwr = RandomWalkOntology(onto_net, l2i, rst=0.7, ontology_nlp_file = ontology_nlp_file, ontology_nlp_emb_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology.nlp.emb')
#y2y = metrics.pairwise.pairwise_distances(onto_net_rwr)
u, s, vt = svds(np.log(onto_net_rwr + 1e-5) - np.log(1e-5), k=dim)
y_emb = np.dot(u, np.diag(np.sqrt(s)))

fout = open('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology.graph.emb','w')
for l in l2i:
	fout.write(l)
	for w in y_emb[l2i[l],:]:
		fout.write('\t'+str(w))
	fout.write('\n')
fout.close()
