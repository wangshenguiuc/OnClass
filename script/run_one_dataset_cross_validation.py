import sys
from scipy import stats, sparse
import numpy as np
import os
from collections import Counter
from OnClass.OnClassModel import OnClassModel
from utils import read_ontology_file, read_data, make_folder, read_data_file, read_data, parse_pkl, SplitTrainTest, MapLabel2CL, evaluate
from config import ontology_data_dir, scrna_data_dir, result_dir


if len(sys.argv) <= 2:
	pid = 0
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

niter = 5
output_dir = make_folder(result_dir+'/CompareBaselines/OneDatasetCrossValidation/')

dnames = ['allen','muris_droplet','krasnow_10x','microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy','muris_facs']
#dnames = ['muris_facs','muris_droplet']
pct = -1
for dname in dnames:
	cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file(dname, ontology_data_dir)
	OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
	feature_file, filter_key, label_key, label_file, gene_file = read_data_file(dname, scrna_data_dir)
	if feature_file.endswith('.pkl'):
		feature, label, genes = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
	elif feature_file.endswith('.h5ad'):
		feature, genes, label, _, _ = read_data(feature_file, cell_ontology_ids = OnClass.cell_ontology_ids,
		exclude_non_leaf_ontology = True, tissue_key = 'tissue', filter_key = filter_key, AnnData_label_key=label_key,
		nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass.co2vec_nlp)
	for iter in range(niter):
		for unseen_ratio in [0.1,0.3,0.5,0.7,0.9]:
			pct += 1
			if total_pid>1 and pct%total_pid != pid:
				continue

			folder = make_folder(output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio) + '/')
			train_feature, train_label, test_feature, test_label = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = iter)
			train_genes = genes
			test_genes = genes
			OnClass_train_obj.EmbedCellTypes(train_label)
			print ('generate pretrain model. Save the model to $model_path...')
			model_path = folder + 'model'
			cor_train_feature, cor_test_feature, cor_train_genes, cor_test_genes = OnClass_train_obj.ProcessTrainFeature(train_feature, train_label, train_genes, test_feature = test_feature, test_genes = test_genes)
			OnClass_train_obj.BuildModel(ngene = len(cor_train_genes))

			OnClass_train_obj.Train(cor_train_feature, train_label, save_model = model_path)
			print (np.shape(cor_train_feature), np.shape(cor_test_feature), np.shape(cor_train_genes), np.shape(cor_test_genes))
			print ('initialize test model. Load the model from $model_path...')
			OnClass_test_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
			cor_test_feature = OnClass_train_obj.ProcessTestFeature(cor_test_feature, cor_test_genes, use_pretrain = model_path, log_transform = False)
			OnClass_test_obj.BuildModel(ngene = None, use_pretrain = model_path)

			pred_Y_seen, pred_Y_all, pred_label = OnClass_test_obj.Predict(cor_test_feature, test_genes = cor_test_genes, use_normalize=False, unseen_ratio = -1)
			#np.save(folder+ 'pred_Y_seen.npy',pred_Y_seen)
			#np.save(folder+ 'pred_Y_all.npy',pred_Y_all)

			print (np.shape(pred_Y_all))
			nseen = OnClass_train_obj.nseen
			onto_net = OnClass_train_obj.ontology_dict
			unseen_l_str = OnClass_train_obj.unseen_co
			unseen_l = MapLabel2CL(unseen_l_str, OnClass_train_obj.co2i)
			test_Y = MapLabel2CL(test_label, OnClass_train_obj.co2i)
			train_Y = MapLabel2CL(train_label, OnClass_train_obj.co2i)
			test_Y_ind = np.sort(np.array(list(set(test_Y) |  set(train_Y))))
			res = evaluate(pred_Y_all, test_Y, unseen_l, nseen, Y_ind = test_Y_ind, Y_net = onto_net, write_screen = True, prefix = 'OnClass')

#8*5*5
