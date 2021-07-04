import sys
from scipy import stats, sparse
import numpy as np
import os
from collections import Counter
from OnClass.OnClassModel import OnClassModel
from utils import read_ontology_file, read_data, make_folder, read_data_file, find_gene_ind
from config import ontology_data_dir, scrna_data_dir, result_dir

if len(sys.argv) <= 2:
	pid = 0
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

print (result_dir)
output_dir = make_folder(result_dir+'/Crossdatasets/')
print (output_dir)
dnames = ['muris_droplet','microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy','muris_facs']
#dnames = ['microcebusBernard','microcebusMartine']
pct = -1
for dname1 in dnames:
	cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file(dname1, ontology_data_dir)
	OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
	feature_file, filter_key, label_key, label_file, gene_file = read_data_file(dname1, scrna_data_dir)
	if feature_file.endswith('.pkl'):
		feature1, label1, genes1 = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
	elif feature_file.endswith('.h5ad'):
		feature1, genes1, label1, _, _ = read_data(feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
		exclude_non_leaf_ontology = True, tissue_key = 'tissue', filter_key = filter_key, AnnData_label_key=label_key,
		nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp)
	for dname2 in dnames:
		if dname1 == dname2:
			continue
		feature_file, filter_key, label_key, label_file, gene_file = read_data_file(dname2, scrna_data_dir)
		if feature_file.endswith('.pkl'):
			feature2, label2, genes2 = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
		elif feature_file.endswith('.h5ad'):
			feature2, genes2, label2, _, _ = read_data(feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
			exclude_non_leaf_ontology = True, tissue_key = 'tissue', filter_key = filter_key, AnnData_label_key=label_key,
			nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp)
		pct += 1
		if total_pid>1 and pct%total_pid != pid:
			continue
		feature1 = np.array(feature1, dtype=np.float64)
		feature2 = np.array(feature2, dtype=np.float64)
		common_genes = np.sort(list(set(genes1) & set(genes2)))
		gid1 = find_gene_ind(genes1, common_genes)
		gid2 = find_gene_ind(genes2, common_genes)
		train_feature = np.array(feature1[:, gid1])
		test_feature = np.array(feature2[:, gid2])
		train_label = label1
		test_label = label2
		train_genes = common_genes
		test_genes = common_genes
		OnClass_train_obj.EmbedCellTypes(train_label)
		# train the model
		print ('generate pretrain model. Save the model to $model_path...')
		model_path = ontology_data_dir + 'test'
		cor_train_feature, cor_test_feature, cor_train_genes, cor_test_genes = OnClass_train_obj.ProcessTrainFeature(train_feature, train_label, train_genes, test_feature = test_feature, test_genes = test_genes)
		OnClass_train_obj.BuildModel(ngene = len(cor_train_genes))

		OnClass_train_obj.Train(cor_train_feature, train_label, save_model = model_path, log_transform = True)
		print (np.shape(cor_train_feature), np.shape(cor_test_feature), np.shape(cor_train_genes), np.shape(cor_test_genes))
		print ('initialize test model. Load the model from $model_path...')
		OnClass_test_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
		cor_test_feature = OnClass_train_obj.ProcessTestFeature(cor_test_feature, cor_test_genes, use_pretrain = model_path, log_transform = False)
		OnClass_test_obj.BuildModel(ngene = None, use_pretrain = model_path)

		pred_Y_seen, pred_Y_all, pred_label = OnClass_test_obj.Predict(cor_test_feature, test_genes = cor_test_genes, use_normalize=False, unseen_ratio = -1)
		#pred_label_str = [OnClass_test_obj.i2co[l] for l in pred_label]
		#pred_Y_seen, pred_Y_all, pred_label = OnClass.Predict(test_X)
		np.save(output_dir+dname1+'.'+dname2 + 'pred_Y_seen.released.npy',pred_Y_seen)
		np.save(output_dir+dname1+'.'+dname2 + 'pred_Y_all.released.npy',pred_Y_all)
