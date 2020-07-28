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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from model.BilinearNNDebug import BilinearNN

from libs import my_assemble, datanames_26datasets
from plots import *


if len(sys.argv) <= 2:
	pid = 0
	total_pid = -1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

output_dir = OUTPUT_DIR + '/26datasets/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
fig_dir = OUTPUT_DIR + '/Figures/BatchCorrection/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
ontology_dir =   OUTPUT_DIR+'/Ontology/CellOntology/'

scan_dim = 50
nn_nhidden = [100,50,25]
keep_prob = 0.7
dname = 'muris_facs'
use_pretrain = False
if use_pretrain:
	prefix = 'pretrained.'
else:
	prefix = ''
test_Y_pred = np.load(output_dir+'integrated.'+prefix+'pred_Y_all.npy')
print (np.shape(test_Y_pred))

datasets, genes_list, n_cells = load_names(datanames_26datasets,verbose=False,log1p=True)
onto_ids, keywords, keyword2cname = map_26_datasets_cell2cid(use_detailed=False)

datasets, genes = merge_datasets(datasets, genes_list)

dnames2offical = {}
dnames2offical['pbmc'] = 'PBMCs'
dnames2offical['brain'] = 'Neurons'
dnames2offical['pancreas'] = 'Pancreatic islets'
dnames2offical['hsc'] = 'HSCs'
dnames2offical['293t_jurkat'] = 'Jurkat + 293T'
dnames2offical['macrophage'] = 'Macrophages'
ndata = len(datanames_26datasets)
nlabel = np.shape(test_Y_pred)[1]
st_ind = 0
ed_ind = 0
labels = []
for i,dnames_raw in enumerate(datanames_26datasets):
	dnames_raw_v = dnames_raw.replace(DATA_DIR+ '/Scanorama','').split('/')
	data_ind = dnames_raw_v.index('data')
	dnames = dnames2offical[dnames_raw_v[data_ind+1]]
	ed_ind = st_ind + np.shape(datasets[i])[0]
	for ii in range(ed_ind-st_ind):
		labels.append(dnames)
	st_end = ed_ind
labels = np.array(labels)

dim = 50

test_Y_pred_red = umap.UMAP(random_state = 1,n_components=dim).fit_transform(test_Y_pred)
sil_sc = metrics.silhouette_samples(test_Y_pred_red, labels, metric='cosine')

datasets_dimred, genes = process_data(datasets, genes, dimred=100)
datasets_dimred = assemble(datasets_dimred, ds_names=datanames_26datasets, sigma=150, verbose=False)
datasets_dimred = np.vstack(datasets_dimred)

orig_sil_sc = metrics.silhouette_samples(datasets_dimred, labels, metric='cosine')
print ('%f %f'%(np.mean(sil_sc),np.mean(orig_sil_sc)))

plot_silhouette_boxplot(sil_sc, orig_sil_sc, fig_dir + str(dim)+prefix+'boxplot.pdf')

pvalue = scipy.stats.ttest_rel(sil_sc, orig_sil_sc)[1]
print (pvalue)
lab2col, lab2marker = generate_colors(labels)

index = np.random.choice(len(labels),len(labels),replace=False)
plot_umap(test_Y_pred_red[index,:], labels[index], lab2col, file = fig_dir + str(dim) + prefix+'our_umap.pdf',legend=False)
plot_umap(datasets_dimred[index,:], labels[index], lab2col, file = fig_dir + str(dim) + prefix+'scan_umap.pdf',legend=False)
