import scanorama
import numpy as np
import os
import sys
from scipy import sparse
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass/'
sys.path.append(repo_dir)

from OnClass.utils import *
from OnClass.OnClassModel import OnClassModel
from OnClass.other_datasets_utils import my_assemble, data_names_all, load_names


OnClassModel = OnClassModel()
tp2emb, tp2i, i2tp = OnClassModel.EmbedCellTypes(dim=500,cell_type_network_file='../../../OnClass_data/cell_ontology/cl.ontology', use_pretrain='../../../OnClass_data/pretrain/tp2emb_500')
print ('compute cell type embedding finished')


data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-facs_cell_ontology.h5ad'
train_X, train_genes, train_Y = read_data(feature_file=data_file, tp2i = tp2i, AnnData_label='cell_ontology_class_reannotated')


#OnClassModel.train(train_X, train_Y, tp2emb, train_genes, nhidden=[2], max_iter=1, use_pretrain = None, save_model =  '../../../OnClass_data/pretrain/BilinearNN')
#test_label = OnClassModel.predict(train_X, train_genes)
#print (np.shape(test_label))
#print (test_label)
#print ('pretrain finished')

print ('../../../OnClass_data/pretrain/BilinearNN_500')
OnClassModel.train(train_X, train_Y, tp2emb, train_genes, nhidden=[500], log_transform = True, use_pretrain = '../../../OnClass_data/pretrain/BilinearNN_50019', pretrain_expression='../../../OnClass_data/pretrain/BilinearNN_500')
#test_label = OnClassModel.predict(train_X, train_genes)
#print (test_label)
print (len(data_names_all))

datasets, genes_list, n_cells = load_names(data_names_all,verbose=False,log1p=True, DATA_DIR='../../../OnClass_data/')
datasets, genes = scanorama.merge_datasets(datasets, genes_list)
datasets_dimred, genes = scanorama.process_data(datasets, genes, dimred=100)
expr_datasets = my_assemble(datasets_dimred, ds_names=data_names_all, expr_datasets = datasets, sigma=150)[1]
expr_corrected = sparse.vstack(expr_datasets)
test_label = OnClassModel.predict(expr_corrected, genes,log_transform=False,correct_batch=False)
print (test_label)