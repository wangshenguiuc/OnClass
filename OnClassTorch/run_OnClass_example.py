from anndata import read_h5ad

#Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import date, datetime
from scanorama.scanorama import correct
from scipy import stats, sparse
import numpy as np
import sys
from collections import Counter
from OnClass.OnClassModel_Torch import OnClassModel
from utils import read_ontology_file, read_data
from config import ontology_data_dir, scrna_data_dir, model_dir, NHIDDEN, MAX_ITER, MEMORY_SAVING_MODE

train_file = scrna_data_dir + '/Lemur/microcebusBernard.h5ad'
test_file = scrna_data_dir + '/Lemur/microcebusAntoine.h5ad'

train_label = 'cell_ontology_id'
test_label = 'cell_ontology_id'
model_path = model_dir + 'example_file_model'

# read data
print ('read ontology data and initialize training model...')
cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file('cell ontology', ontology_data_dir)
OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file, memory_saving_mode=MEMORY_SAVING_MODE)

print ('read training single cell data...')
train_feature, train_genes, train_label, _, _ = read_data(train_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
	exclude_non_leaf_ontology = False, tissue_key = 'tissue', AnnData_label_key = train_label, filter_key = {},
	nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp,
	memory_saving_mode=MEMORY_SAVING_MODE)

print ('embed cell types using the cell ontology...')
OnClass_train_obj.EmbedCellTypes(train_label)

# Read test data
if MEMORY_SAVING_MODE:	
	x = read_h5ad(test_file, backed='r')
	test_feature = x.X.to_memory() # Gets a sparse array in csr matrix form
else:
	x = read_h5ad(test_file)
	test_feature = x.X.toarray()
test_label = x.obs[test_label].tolist()
test_genes = np.array([x.upper() for x in x.var.index])

print ('generate pretrain model. Save the model to $model_path...')
cor_train_feature, cor_test_feature, cor_train_genes, cor_test_genes = OnClass_train_obj.ProcessTrainFeature(
	train_feature, train_label, train_genes, test_feature = test_feature, test_genes = test_genes,
	batch_correct=False)

OnClass_train_obj.BuildModel(ngene = len(cor_train_genes), nhidden = NHIDDEN)
OnClass_train_obj.Train(cor_train_feature, train_label, save_model = model_path, max_iter = MAX_ITER)

print ('initialize test model. Load the model from $model_path...')
OnClass_test_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file, memory_saving_mode=MEMORY_SAVING_MODE)
cor_test_feature, mapping = OnClass_train_obj.ProcessTestFeature(
	cor_test_feature, cor_test_genes, use_pretrain = model_path, log_transform = False)
OnClass_test_obj.BuildModel(ngene = None, use_pretrain = model_path)

# Prediction
pred_Y_seen, pred_Y_all, pred_label = OnClass_test_obj.Predict(
	cor_test_feature, test_genes = cor_test_genes, use_normalize=False)
pred_label_str = [OnClass_test_obj.i2co[l] for l in pred_label]

# Print accuracy
correct = 0.0
for i in range(len(pred_label_str)):
	if i % 1000 == 0:
		print(test_label[i], "\t\t", pred_label_str[i])
	if test_label[i] == pred_label_str[i]:
		correct += 1.0
print("Test accuracy is", 100.0 * correct / len(pred_label_str), "%")

# Save predictions
# pickle_these_objects(pred_Y_seen, pred_Y_all, pred_label, None, None, None, filename='predictions.pickle')


# pred_Y_seen, pred_Y_all, pred_label, _, _, _ = unpickle_from_file(filename='predictions.pickle')
# print ('read test single cell data...')
# x = read_h5ad(test_file)
# test_label = x.obs[test_label].tolist()

print("There are", len(np.unique(pred_label)), "different labels in predictions")
print("There are", len(np.unique(test_label)), "different labels in true")
