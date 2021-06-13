import sys
from scipy import stats, sparse
import numpy as np
import os
from collections import Counter
from OnClass.OnClassModel import OnClassModel
from utils import read_ontology_file, read_data, make_folder, read_data_file, read_data, parse_pkl, SplitTrainTest
from config import ontology_data_dir, scrna_data_dir, result_dir, model_dir, intermediate_dir


if len(sys.argv) <= 2:
	pid = 0
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

output_dir = intermediate_dir + '/Marker_genes/'

dnames = ['microcebusAntoine','allen','muris_droplet','krasnow_10x','microcebusBernard','microcebusMartine','microcebusStumpy','muris_facs']
#dnames = ['muris_facs','muris_droplet']
unseen_ratio = 0.5
pct = -1
for dname in dnames:
	pct += 1
	if total_pid>1 and pct%total_pid != pid:
		continue
	cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file(dname, ontology_data_dir)
	OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
	feature_file, filter_key, label_key, label_file, gene_file = read_data_file(dname, scrna_data_dir)
	if feature_file.endswith('.pkl'):
		feature, label, genes = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
	elif feature_file.endswith('.h5ad'):
		feature, genes, label, _, _ = read_data(feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
		exclude_non_leaf_ontology = True, tissue_key = 'tissue', filter_key = filter_key, AnnData_label_key=label_key,
		nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp)

	train_feature, train_label, test_feature, test_label = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = 0)
	train_feature = np.vstack((train_feature, test_feature))
	train_label = np.concatenate((train_label, test_label))
	train_genes = genes
	test_genes = genes
	OnClass_train_obj.EmbedCellTypes(train_label)
	print ('generate pretrain model. Save the model to $model_path...')
	model_path = model_dir + 'OnClass_full_'+dname
	train_feature, train_genes = OnClass_train_obj.ProcessTrainFeature(train_feature, train_label, train_genes)
	OnClass_train_obj.BuildModel(ngene = len(train_genes))
	OnClass_train_obj.Train(train_feature, train_label, save_model = model_path)

	OnClass_test_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
	OnClass_test_obj.BuildModel(ngene = None, use_pretrain = model_path)
	pred_Y_seen, pred_Y_all, pred_label = OnClass_test_obj.Predict(train_feature, test_genes = train_genes, use_normalize=False, use_unseen_distance = -1)
	np.save(output_dir+dname + 'pred_Y_seen.released.npy',pred_Y_seen)
	np.save(output_dir+dname + 'pred_Y_all.released.npy',pred_Y_all)
