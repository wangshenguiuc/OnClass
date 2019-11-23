import sys
from scanorama import *
import numpy as np
import os
from scipy import sparse
import sys
from OnClass.utils import *
from OnClass.OnClassPred import OnClassPred
from OnClass.other_datasets_utils import my_assemble, data_names_all, load_names

OUTPUT_DIR = '../../OnClass_data/26-datasets/'
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

## read TMS and 26-datasets data
DATA_DIR = '../../../OnClass_data/'
data_file = DATA_DIR + '/raw_data/tabula-muris-senis-facs.h5ad'
train_X, train_Y_str, genes_list = read_data(filename=data_file, DATA_DIR = DATA_DIR, return_genes=True)

## embedd the cell ontology
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, DATA_DIR=DATA_DIR,co_dim=5)
train_Y = MapLabel2CL(train_Y_str, l2i)


## annotate 26-datasets, train on TMS
ntrain,ngene = np.shape(train_X.todense())

OnClass_obj = OnClassPred()

OnClass_obj.train(train_X, train_Y, Y_emb, log_transform=True)

test_Y_pred = OnClass_obj.predict(test_X_corrected, log_transform=False)

