from anndata import read_h5ad
from scipy import stats, sparse
import numpy as np
import sys
from collections import Counter
from OnClass.OnClassModel import OnClassModel
from utils import read_ontology_file, read_data, run_scanorama_multiply_datasets
from config import ontology_data_dir, scrna_data_dir, model_dir, Run_scanorama_batch_correction, NHIDDEN, MAX_ITER
train_file = scrna_data_dir + '/Lemur/microcebusBernard.h5ad'
test_file = scrna_data_dir + '/Lemur/microcebusAntoine.h5ad'

train_label = 'cell_ontology_id'
test_label = 'cell_ontology_id'
model_path = model_dir + 'example_file_model'

# read data
print ('read ontology data and initialize training model...')
cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file('cell ontology', ontology_data_dir)
OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)

print ('read training single cell data...')
train_feature, train_genes, train_label, _, _ = read_data(train_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
	exclude_non_leaf_ontology = False, tissue_key = 'tissue', AnnData_label_key = train_label, filter_key = {},
	nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp)
#you can also replace it with your own data and make sure that:
#train_feature is a ncell by ngene matrix
#train_genes is a ngene long vector of gene names
#train_label is a ncell long vector
print ('embed cell types using the cell ontology...')
OnClass_train_obj.EmbedCellTypes(train_label)

print ('read test single cell data...')
x = read_h5ad(test_file)
test_label = x.obs[test_label].tolist()
test_feature = x.X.toarray()
test_genes = np.array([x.upper() for x in x.var.index])

# train the model
if Run_scanorama_batch_correction:
	train_feature, test_feature = run_scanorama_multiply_datasets([train_feature, test_feature], [train_genes, test_genes], scan_dim = 10)[1]
	print (np.shape(train_feature), np.shape(test_feature))

print ('generate pretrain model. Save the model to $model_path...')
cor_train_feature, cor_test_feature, cor_train_genes, cor_test_genes = OnClass_train_obj.ProcessTrainFeature(train_feature, train_label, train_genes, test_feature = test_feature, test_genes = test_genes)
OnClass_train_obj.BuildModel(ngene = len(cor_train_genes), nhidden = NHIDDEN)
OnClass_train_obj.Train(cor_train_feature, train_label, save_model = model_path, max_iter = MAX_ITER)

print ('initialize test model. Load the model from $model_path...')
OnClass_test_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
cor_test_feature = OnClass_train_obj.ProcessTestFeature(cor_test_feature, cor_test_genes, use_pretrain = model_path, log_transform = False)
OnClass_test_obj.BuildModel(ngene = None, use_pretrain = model_path)
#prediction
#use_normalize=False will return a tree-based prediction, where parent node often has higher score than child node. use_normalize=True will normalize among child nodes and parent nodes
pred_Y_seen, pred_Y_all, pred_label = OnClass_test_obj.Predict(cor_test_feature, test_genes = cor_test_genes, use_normalize=True)
pred_label_str = [OnClass_test_obj.i2co[l] for l in pred_label]
#x.obs['OnClass_annotation_flat_based_ontology_ID'] = pred_label_str

pred_Y_seen, pred_Y_all, pred_label = OnClass_test_obj.Predict(cor_test_feature, test_genes = cor_test_genes, use_normalize=False)
pred_label_str = [OnClass_test_obj.i2co[l] for l in pred_label]
#x.obs['OnClass_annotation_tree_based_ontology_ID'] = pred_label_str
#x.write(scrna_data_dir + 'Pilot12.annotated.h5ad')
