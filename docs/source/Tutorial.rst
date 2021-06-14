How to run OnClass
=========

To run OnClass, please first install OnClass, download datasets and then change file paths in `config.py <https://github.com/wangshenguiuc/OnClass/blob/master/run_OnClass_example.py>`__

We provide a `run_OnClass_example.py <https://github.com/wangshenguiuc/OnClass/blob/master/run_OnClass_example.py>`__ and Jupyter notebook as an example to run OnClass. This script trains an OnClass model on all cells from one Lemur dataset, saves that model to a model file, then use this model to classify cells from another Lemur dataset.

Run your own dataset for cell type annotation
----------------
You only need to modify line 9-13 in `run_OnClass_example.py <https://github.com/wangshenguiuc/OnClass/blob/master/run_OnClass_example.py>`__ by replacing train_file, test_file with your training and test file, and train_label and test_label with the cell ontology label key in your dataset.

Import OnClass and other libs as::

	from anndata import read_h5ad
	from scipy import stats, sparse
	import numpy as np
	import sys
	from collections import Counter
	from OnClass.OnClassModel import OnClassModel
	from utils import read_ontology_file, read_data, run_scanorama_multiply_datasets
	from config import ontology_data_dir, scrna_data_dir, model_dir, Run_scanorama_batch_correction, NHIDDEN, MAX_ITER

Read training and test data. If you don't want to use h5ad file, you can provide training and test data in the format of numpy array to OnClass. Training and test features (gene expression) should be cell by gene 2D array. Training label should be a vector of cell labels. ::

	train_file = scrna_data_dir + '/Lemur/microcebusBernard.h5ad'
	test_file = scrna_data_dir + '/Lemur/microcebusAntoine.h5ad'

	train_label = 'cell_ontology_id'
	test_label = 'cell_ontology_id'
	model_path = model_dir + 'example_file_model'

	cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file('cell ontology', ontology_data_dir)
	OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)

	train_feature, train_genes, train_label, _, _ = read_data(train_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
		exclude_non_leaf_ontology = False, tissue_key = 'tissue', AnnData_label_key = train_label, filter_key = {},
		nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp)

Embed the cell ontology::

	OnClass_train_obj.EmbedCellTypes(train_label)

Batch correction using Scanorama::

	if Run_scanorama_batch_correction:
		train_feature, test_feature = run_scanorama_multiply_datasets([train_feature, test_feature], [train_genes, test_genes], scan_dim = 10)[1]

Training::

	cor_train_feature, cor_test_feature, cor_train_genes, cor_test_genes = OnClass_train_obj.ProcessTrainFeature(train_feature, train_label, train_genes, test_feature = test_feature, test_genes = test_genes)
	OnClass_train_obj.BuildModel(ngene = len(cor_train_genes), nhidden = NHIDDEN)
	OnClass_train_obj.Train(cor_train_feature, train_label, save_model = model_path, max_iter = MAX_ITER)


Test::

	OnClass_test_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
	cor_test_feature = OnClass_train_obj.ProcessTestFeature(cor_test_feature, cor_test_genes, use_pretrain = model_path, log_transform = False)
	OnClass_test_obj.BuildModel(ngene = None, use_pretrain = model_path)

	pred_Y_seen, pred_Y_all, pred_label = OnClass_test_obj.Predict(cor_test_feature, test_genes = cor_test_genes, use_normalize=True)
	pred_label_str = [OnClass_test_obj.i2co[l] for l in pred_label]


One dataset cross-validation
----------------
`run_one_dataset_cross_validation.py <https://github.com/wangshenguiuc/OnClass/blob/master/script/run_one_dataset_cross_validation.py>`__ can be used to reproduce Figure 2 in our paper. All data are provided in figshare (please see Dataset and pretrained model)

Cross dataset prediction
----------------
`run_cross_dataset_prediction.py <https://github.com/wangshenguiuc/OnClass/blob/master/script/run_cross_dataset_prediction.py>`__ can be used to reproduce Figure 4 in our paper. All data are provided in figshare (please see Dataset and pretrained model)

Marker genes identification
----------------
Please first run `run_generate_pretrained_model.py <https://github.com/wangshenguiuc/OnClass/blob/master/script/run_generate_pretrained_model.py>`__ to generate the intermediate files for marker gene prediction. Then run `run_marker_genes_identification.py <https://github.com/wangshenguiuc/OnClass/blob/master/script/run_marker_genes_identification.py>`__ for marker gene identification (Figure 5c) and `run_marker_gene_based_prediction.py <https://github.com/wangshenguiuc/OnClass/blob/master/script/run_marker_gene_based_prediction.py>`__ for marker gene based prediction (Figure 5d,e,f, Extended Data Figure 7).
