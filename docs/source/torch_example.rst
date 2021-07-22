|PyPI| |Docs|

.. |PyPI| image:: https://img.shields.io/pypi/v/scanpy.svg
   :target: https://pypi.org/project/OnClass/
.. |Docs| image:: https://readthedocs.com/projects/icb-scanpy/badge/?version=latest
   :target: https://onclass.readthedocs.io/en/latest/introduction.html

PyTorch Version of OnClass
=========

We also wrote a version of OnClass in PyTorch. This version implements a memory saving mode toggle that allows the program to run with as low as 8GB of RAM. Using this model is almost identical to the TensorFlow version, but there are a couple differences that I will illustrate below.

Cell type annotation
----------------

An example script `run_OnClass_example.py <https://github.com/wangshenguiuc/OnClass/OnClass_Torch/run_OnClass_example.py>`__ for cell type annotation is at our `GitHub <https://github.com/wangshenguiuc/OnClass/OnClass_Torch/run_OnClass_example.py>`__. I will walk through the code in this tutorial.

First import PyTorch version of OnClass as::

	from OnClass_Torch.utils import *
	from OnClass_Torch.OnClass.OnClassModel_Torch import OnClassModel

This tutorial also uses the following settings::

	from utils import read_ontology_file, read_data
	from config import ontology_data_dir, scrna_data_dir, model_dir, NHIDDEN, MAX_ITER, MEMORY_SAVING_MODE

	train_file = scrna_data_dir + '/Lemur/microcebusBernard.h5ad'
	test_file = scrna_data_dir + '/Lemur/microcebusAntoine.h5ad'

	train_label = 'cell_ontology_id'
	test_label = 'cell_ontology_id'
	model_path = model_dir + 'example_file_model'

If you are interested in annotating your own datasets do so by replacing train_file and test_file with paths to your own files.

Next we must read the cell ontology data and initialize the model as::
    
	print ('read ontology data and initialize training model...')
	cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file('cell ontology', ontology_data_dir)
	OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file, memory_saving_mode=MEMORY_SAVING_MODE)

where `MEMORY_SAVING_MODE` is true if you want to run OnClass on low RAM, and false otherwise. This value can be set easily in the `config <https://github.com/wangshenguiuc/OnClass/OnClass_Torch/config.py>`__ file.

Read the training data from the training file::

	print ('read training single cell data...')
	train_feature, train_genes, train_label, _, _ = read_data(train_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
		exclude_non_leaf_ontology = False, tissue_key = 'tissue', AnnData_label_key = train_label, filter_key = {},
		nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp,
		memory_saving_mode=MEMORY_SAVING_MODE)

where `train_feature` is a sample-by-gene gene expression matrix, `train_label` is a label vector for each sample. The labels in `Y` should use the Cell Ontology Id (e.g., CL:1000398). The data (e.g., tabula muris raw gene expression matrix, the Cell Ontology obo file) can be downloaded from FigShare.(see dataset section in this tutorial).

It's important to add, that if the model is in memory saving mode, it will load train_feature as a sparse scipy matrix.

Next, we embed the cell onotology::

	print ('embed cell types using the cell ontology...')
	OnClass_train_obj.EmbedCellTypes(train_label)

Read the test data::

	test_label = x.obs[test_label].tolist()
	test_genes = np.array([x.upper() for x in x.var.index])
	if MEMORY_SAVING_MODE:	
		x = read_h5ad(test_file, backed='r')
		test_feature = x.X.to_memory()
	else:
		x = read_h5ad(test_file)
		test_feature = x.X.toarray()

Often the datasets are too large to read their dense representations into memory. To accomodate this, the low-memory mode loads a sparse representation of the dataset into memory. Namely, a `scipy.sparse.csr_matrix`.

Process the training features and start training the model as::

	print ('generate pretrain model. Save the model to $model_path...')
	cor_train_feature, cor_test_feature, cor_train_genes, cor_test_genes = OnClass_train_obj.ProcessTrainFeature(
		train_feature, train_label, train_genes, test_feature = test_feature, test_genes = test_genes,
		batch_correct=False)
	OnClass_train_obj.BuildModel(ngene = len(cor_train_genes), nhidden = NHIDDEN)
	OnClass_train_obj.Train(cor_train_feature, train_label, save_model = model_path, max_iter = MAX_ITER)

If `model_path` is not none, the model will be saved to that file path. You can load the pretrained model from there as::

	print ('initialize test model. Load the model from $model_path...')
	OnClass_test_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file, memory_saving_mode=MEMORY_SAVING_MODE)
	cor_test_feature, mapping = OnClass_train_obj.ProcessTestFeature(
		cor_test_feature, cor_test_genes, use_pretrain = model_path, log_transform = False)
	OnClass_test_obj.BuildModel(ngene = None, use_pretrain = model_path)

Predict cell-type annotations from the test dataset as::

	pred_Y_seen, pred_Y_all, pred_label = OnClass_test_obj.Predict(
	cor_test_feature, test_genes = cor_test_genes, use_normalize=False)
	pred_label_str = [OnClass_test_obj.i2co[l] for l in pred_label]

`pred_label_str[i]` will contain the predicted cell-type of the `i`th sample in the test dataset.