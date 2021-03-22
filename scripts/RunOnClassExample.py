'''
Running instructions:
1. Change feature_file1, feature_file2 to your data paths.
2. Change label_key1, label_key2 to your label keys in .h5ad object
3. Download OnClass_data and organize the files like this:
├── OnClass
│   ├── scripts
│   │	├── RunOnClassExample.py
│   └── setup.py
├── OnClass_data
│   ├── cell_ontology

4. cd OnClass
python scripts/RunOnClassExample.py
'''

import sys
from anndata import read_h5ad
from scipy import stats, sparse
import numpy as np
import os
from collections import Counter
from OnClass.OnClassModel import OnClassModel
from OnClass.utils import *
data_dir = '../OnClass_data/'
scrna_data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/'
train_file = scrna_data_dir + 'sapiens/Pilot12.training.h5ad'
test_file = scrna_data_dir + 'sapiens/Pilot12.training.h5ad'
train_label = 'cell_ontology_id'
test_label = 'cell_ontology_id'

# read data
print ('read training data')
cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file('muris', data_dir)
OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
train_feature, train_genes, train_label, _, _ = read_data(train_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
	exclude_non_leaf_ontology = False, tissue_key = 'tissue', AnnData_label_key=train_label,filter_key = {},
	nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp)

OnClass_train_obj.EmbedCellTypes(train_label)
#train_X, test_X = OnClass_train_obj.align_train_test_expression(train_X, test_X)

# train the model
print ('generate pretrain model')
model_path = data_dir + 'OnClass_Pilot12_test'
OnClass_train_obj.BuildModel(ngene = len(train_genes))
# OnClass will train a model and save to [model_path].
OnClass_train_obj.Train(train_feature, train_label, save_model = model_path, genes = train_genes)

print ('read test data')
x = read_h5ad(test_file)
test_label = x.obs[test_label].tolist()
test_feature = x.X.toarray()
test_genes = np.array([x.upper() for x in x.var.index])
print (np.sum(test_feature))

print ('test start')
OnClass_test_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
model_path = data_dir + 'OnClass_Pilot12_test'
OnClass_test_obj.BuildModel(ngene = None, use_pretrain = model_path)
#prediction
#use_normalize=False will return a tree-based prediction, where parent node often has higher score than child node. use_normalize=True will normalize among child nodes and parent nodes
pred_Y_seen, pred_Y_all, pred_label = OnClass_test_obj.Predict(np.log1p(test_feature), test_genes = test_genes, use_normalize=True)
pred_label_str = [OnClass_test_obj.i2co[l] for l in pred_label]
x.obs['OnClass_annotation_tree_based_ontology_ID'] = pred_label_str

pred_Y_seen, pred_Y_all, pred_label = OnClass_test_obj.Predict(np.log1p(test_feature), test_genes = test_genes, use_normalize=False)
pred_label_str = [OnClass_test_obj.i2co[l] for l in pred_label]
x.obs['OnClass_annotation_flat_based_ontology_ID'] = pred_label_str
