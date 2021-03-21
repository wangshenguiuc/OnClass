import numpy as np
import sys
import os
#repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/OnClass/'
data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/'
#sys.path.append(repo_dir)
#sys.path.append(repo_dir+'src/task/OnClass/')
sys.path.append('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass/OnClass')
#os.chdir(repo_dir)

from utils import *
from OnClassModel import OnClassModel
from other_datasets_utils import my_assemble, data_names_all, load_names

training_data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-facs_cell_ontology.h5ad'
cell_type_network_file = '../../../OnClass_data/cell_ontology/cl.ontology'
test_data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-droplet.h5ad'
cell_type_nlp_emb_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology.nlp.emb'

co_dim = 5

OnClassModel = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)

print (len(OnClassModel.cell_ontology_ids), list(OnClassModel.cell_ontology_ids)[:5])
train_X, train_genes, train_Y_str = read_data(feature_file=training_data_file, cell_ontology_ids = OnClassModel.cell_ontology_ids, nlp_mapping = True, co2emb = OnClassModel.co2vec_nlp, AnnData_label_key='cell_ontology_class_reannotated')
print (np.shape(train_X))
OnClassModel.EmbedCellTypes(train_Y_str, dim=co_dim, use_pretrain=None)

test_X, test_genes, test_Y_str, test_AnnData = read_data(feature_file=training_data_file, cell_ontology_ids = OnClassModel.cell_ontology_ids, nlp_mapping = True, co2emb =  OnClassModel.co2vec_nlp, return_AnnData=True, AnnData_label_key='cell_ontology_class_reannotated')
#test_X, test_genes, test_Y_str, test_AnnData = read_data(feature_file=test_data_file, cell_ontology_ids = OnClassModel.cell_ontology_ids, nlp_mapping = True, co2emb =  OnClassModel.co2vec_nlp, return_AnnData=True, AnnData_label_key='cell_ontology_id')
#train_X, test_X = process_expression([train_X, test_X])
train_X = np.log1p(train_X)
test_X = np.log1p(test_X)
OnClassModel.train(train_X, train_Y_str, OnClassModel.co2emb, train_genes, nhidden = [1000], max_iter = 10)

test_label = OnClassModel.predict(test_X, test_genes,use_normalize=False)
print (Counter(np.argmax(test_label,axis=1)))
test_label = OnClassModel.predict(test_X, test_genes,use_normalize=True)
print (Counter(np.argmax(test_label,axis=1)))
#x = write_anndata_data(test_label, test_AnnData, OnClassModel.i2co, name_mapping_file='../../../OnClass_data/cell_ontology/cl.obo')#output_file is optional

#print (x.obs['OnClass_annotation_ontology_ID'])
#print (x.obs['OnClass_annotation_ontology_name'])
