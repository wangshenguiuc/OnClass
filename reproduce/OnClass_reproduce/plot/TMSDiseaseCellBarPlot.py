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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn import metrics
from collections import Counter
from scipy.stats import norm as dist_model
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from plots import *
from model.BilinearNNDebug import BilinearNN

if len(sys.argv) <= 2:
	pid = 0
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])
ontology_dir =  OUTPUT_DIR+'/Ontology/CellOntology/'
output_dir = OUTPUT_DIR+'/DiseasedCells/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

figure_dir = OUTPUT_DIR + '/Figures/DiseasedCells/'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

nn_nhidden = [100,50,25]
keep_prob = 0.7

file = DATA_DIR + 'TMS_official_060520/' + 'tabula-muris-senis-facs-official-raw-obj.h5ad'
ontology_file = DATA_DIR + '/cell_ontology/cl.ontology'
ontology_nlp_file =  DATA_DIR + '/cell_ontology/cl.ontology.nlp'
feature_train, label_train, genes =parse_h5ad(file,cell_ontology_file = ontology_file,batch_key = '',
DATA_DIR = DATA_DIR,label_key='cell_ontology_class',
filter_key={'age':'3m'}, exclude_non_leaf_ontology = False, exclude_non_ontology=True)

feature_test1, label_test1, genes =parse_h5ad(file,cell_ontology_file = ontology_file,batch_key = '',
DATA_DIR = DATA_DIR,label_key='cell_ontology_class',
filter_key={'age':'18m'}, exclude_non_leaf_ontology = False, exclude_non_ontology=True)
feature_test2, label_test2, genes =parse_h5ad(file,cell_ontology_file = ontology_file,batch_key = '',
DATA_DIR = DATA_DIR,label_key='cell_ontology_class',
filter_key={'age':'24m'}, exclude_non_leaf_ontology = False, exclude_non_ontology=True)

feature_test = np.vstack((feature_test1, feature_test2))
label_test = np.concatenate((label_test1, label_test2))
print (np.shape(label_test1),np.shape(label_test),np.shape(feature_test))
co_dim = 5
ontology_emb_file = ontology_dir + str(co_dim)
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(label_train, co_dim = co_dim, use_pretrain = ontology_emb_file, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
ncls = np.shape(cls2cls)[0]
label_train = MapLabel2CL(label_train, l2i)
label_test = MapLabel2CL(label_test, l2i)
unseen_l = MapLabel2CL(unseen_l, l2i)
nunseen = len(unseen_l)
nseen = ncls - nunseen
onto_net_rwr = RandomWalkOntology(onto_net, l2i, rst=0.7, ontology_nlp_file = ontology_nlp_file, ontology_nlp_emb_file = ontology_nlp_emb_file)


train_X = np.log1p(feature_train)
test_X = np.log1p(feature_test)

pred_Y_all = np.load(output_dir+'pred_Y_all.npy')

co2name, name2co = get_ontology_name(DATA_DIR=DATA_DIR)
co_id = l2i[name2co['microglial cell']]
print (co_id, co_id in label_train)
test_Y = ConvertLabels(label_test, ncls)
auc = roc_auc_score(test_Y[:, co_id], pred_Y_all[:,co_id])
print (auc)
plot_auroc_curve(test_Y[:, co_id], pred_Y_all[:,co_id], 'microglial cell', figure_dir + 'microglial cell.pdf')
