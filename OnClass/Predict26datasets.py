import sys
from scanorama import *
from time import time
import numpy as np
import os
from scipy import sparse
from utils import *
from OnClass import OnClass
from other_datasets_utils import my_assemble, data_names_all, load_names

output_dir = '../../OnClass_data/26-datasets/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

data_file = '../../OnClass_data/raw_data/tabula-muris-senis-facs'
## read data
train_X, train_Y_str, genes_list = read_data(filename=data_file, return_genes=True)
tms_genes_list = [x.upper() for x in genes_list[datanames[0]]]
ntrain,ngene = np.shape(train_X)
## embedd the cell ontology
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str)
train_Y = MapLabel2CL(train_Y_str, l2i)

unseen_l, l2i, i2l, train_X2Y, onto_net = create_labels(train_Y, combine_unseen = False)

datasets, genes_list, n_cells = load_names(data_names_all,verbose=False,log1p=True)

datasets.append(train_X)
genes_list.append(tms_genes_list)
data_names_all.append('TMS')
datasets, genes = merge_datasets(datasets, genes_list)
datasets_dimred, genes = process_data(datasets, genes, dimred=scan_dim)
datasets_dimred, expr_datasets = my_assemble(datasets_dimred, ds_names=data_names_all, expr_datasets = datasets, sigma=150)
expr_corrected = sparse.vstack(expr_datasets)
expr_corrected = np.log2(expr_corrected.toarray()+1)

nsample = np.shape(datasets_dimred)[0]
train_X_corrected= datasets_dimred[nsample-ntrain:,:]
test_X_corrected = datasets_dimred[:nsample-ntrain,:]

OnClass_obj = OnClass()
OnClass_obj.train(train_X_corrected, train_Y, Y_emb, log_transform=False)
test_Y_pred = OnClass_obj.predict(test_X_corrected, log_transform=False)

np.save(output_dir + '26-datasets-predicted_score_matrix.npy', test_Y_pred)
