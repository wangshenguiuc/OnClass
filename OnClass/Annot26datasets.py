import sys
from scanorama import *
import numpy as np
import os
from scipy import sparse
from utils import *
import OnClassPred
from other_datasets_utils import my_assemble, data_names_all, load_names 

output_dir = '../../OnClass_data/26-datasets/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
	
## read data
data_file = '../../OnClass_data/raw_data/tabula-muris-senis-facs'
train_X, train_Y_str, genes_list = read_data(filename=data_file, return_genes=True)
tms_genes_list = [x.upper() for x in list(genes_list.values())[0]]
datasets, genes_list, n_cells = load_names(data_names_all,verbose=False,log1p=True)
datasets.append(train_X)
genes_list.append(tms_genes_list)
data_names_all.append('TMS')

## embedd the cell ontology
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str)
train_Y = MapLabel2CL(train_Y_str, l2i)

## use Scanorama to correct batch effects
datasets, genes = merge_datasets(datasets, genes_list)
datasets_dimred, genes = process_data(datasets, genes, dimred=100)
expr_datasets = my_assemble(datasets_dimred, ds_names=data_names_all, expr_datasets = datasets, sigma=150)[1]
expr_corrected = sparse.vstack(expr_datasets)
expr_corrected = np.log2(expr_corrected.toarray()+1)

## annotate 26-datasets, train on TMS
ntrain,ngene = np.shape(train_X)
nsample = np.shape(expr_corrected)[0]
train_X_corrected= expr_corrected[nsample-ntrain:,:]
test_X_corrected = expr_corrected[:nsample-ntrain,:]
OnClass_obj = OnClassPred()
OnClass_obj.train(train_X_corrected, train_Y, Y_emb, log_transform=False)
test_Y_pred = OnClass_obj.predict(test_X_corrected, log_transform=False)

## save the prediction matrix, nsample (number of samples in 26-datasets) by nlabels
np.save(output_dir + '26_datasets_predicted_score_matrix.npy', test_Y_pred)
