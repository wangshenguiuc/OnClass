import sys
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/tools/single_cell/scanorama/bin/'
sys.path.append(repo_dir)
from sklearn.cross_decomposition import CCA
from sklearn import metrics
from process import load_names
from scanorama import *
from time import time
import numpy as np
from collections import Counter
import os

repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
sys.path.append(repo_dir)
sys.path.append(repo_dir+'src/task/SCTypeClassifier/OurClassifier/DeepCCA')
sys.path.append(repo_dir+'src/task/SCTypeClassifier/')
os.chdir(repo_dir)
from utils import *
from NN import BilinearNN

output_dir = '../../OnClass_data/marker_gene/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
## read data

data_file = '../../OnClass_data/raw_data/tabula-muris-senis-facs'
train_X, train_Y_str = read_data(filename=data_file)
tms_genes_list = [x.upper() for x in genes_list[datanames[0]]]
ntrain,ngene = np.shape(train_X)
## embedd the cell ontology
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str)
train_Y = MapLabel2CL(train_Y_str, l2i)

## train and predict
OnClass_obj = OnClass()
OnClass_obj.train(train_X, train_Y, Y_emb)
test_Y_pred = OnClass_obj.predict(test_X)

np.save(output_dir + 'FACS-predicted_score_matrix.npy', test_Y_pred)
