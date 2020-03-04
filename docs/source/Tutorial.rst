Tutorial
=========
Here, we provide an introduction of how to use OnClass. We are going to train a model on all FACS cells from Tabula Muris Senis (TMS) and then predict the cell types of all droplet cells in TMS. By using this example, you can see how OnClass embeds the Cell Ontology, reads gene expression data, uses the pretrained model, and makes the prediction on new cells.


Cell type annotation (Use the pretrained model)
----------------

The script `AnnotateTMS.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/AnnotateTMS.py>`__ for transferring cell type annotation is at our `GitHub <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/AnnotateTMS.py>`__ We provide a pretrained model in our data repo, which was trained on all FACS cells in tabula muris senis.

Import OnClass and other libs as::

	from OnClass.utils import *
	from OnClass.OnClassModel import OnClassModel
	from OnClass.other_datasets_utils import my_assemble, data_names_all, load_names

Embed the cell ontology.::

	OnClassModel = OnClassModel()
	tp2emb, tp2i, i2tp = OnClassModel.EmbedCellTypes(dim=500,cell_type_network_file='../../../OnClass_data/cell_ontology/cl.ontology', use_pretrain='../../../OnClass_data/pretrain/tp2emb_500')

Here, we used the pretrained cell type embedding file tp2emb_500, which is the 500-dimensional vectors of cell types from cl.ontology. All files are provided on figshare. Please download them and put them in the corresponding directory. At this step, we are not using gene expression or cell type annotations when embedding the Cell Ontology. If you want to generate your own embeddings, please set use_pretrain = None.


Read TMS h5ad gene expression data. This file is also on figshare which is the data used in our paper. cell_ontology_class_reannotated is the attribute of labels in the h5ad file::

	data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-facs_cell_ontology.h5ad'
	train_X, train_genes, train_Y = read_data(feature_file=data_file, tp2i = tp2i, AnnData_label='cell_ontology_class_reannotated')

Train the model ::

	OnClassModel.train(train_X, train_Y, tp2emb, train_genes, nhidden=[500], log_transform = True, use_pretrain = '../../../OnClass_data/pretrain/BilinearNN_50019')

Here, we use the pretrained model BilinearNN_500 which can be downloaded from figshare. If you want to train your own model, please set use_pretrain = None.

Predict the labels of cells in droplet cells. ::

	test_data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-droplet.h5ad'
	test_X, test_genes, test_AnnData = read_data(feature_file=test_data_file, tp2i = tp2i, return_AnnData = True)

	test_label = OnClassModel.predict(test_X, test_genes,log_transform=True,correct_batch=False)

	x = write_anndata_data(test_label, test_AnnData, i2tp, name_mapping_file='../../../OnClass_data/cell_ontology/cl.obo')#output_file is optional

	print (x.obs['OnClass_annotation_ontology_ID'])
	print (x.obs['OnClass_annotation_ontology_name'])

Cell type annotation (Train from scratch)
----------------

The script `TrainOnClassFromScratch.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/TrainOnClassFromScratch.py>`__ showing an example of training from scratch is at our `GitHub <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/TrainOnClassFromScratch.py>`__
To run this script, you need a training data (tabula-muris-senis-facs_cell_ontology.h5ad in this script) and a test data (tabula-muris-senis-droplet.h5ad in this script). The two datasets used in our script are in figshare.

Import OnClass and other libs as::

	from OnClass.utils import *
	from OnClass.OnClassModel import OnClassModel
	from OnClass.other_datasets_utils import my_assemble, data_names_all, load_names

Embed the cell ontology.::

	OnClassModel = OnClassModel()
	tp2emb, tp2i, i2tp = OnClassModel.EmbedCellTypes(dim=500,cell_type_network_file='../../../OnClass_data/cell_ontology/cl.ontology', use_pretrain='../../../OnClass_data/pretrain/tp2emb_500')

Here, we used the pretrained cell type embedding file tp2emb_500, which is the 500-dimensional vectors of cell types from cl.ontology. All files are provided on figshare. Please download them and put them in the corresponding directory. At this step, we are not using gene expression or cell type annotations when embedding the Cell Ontology. If you want to generate your own embeddings, please set use_pretrain = None.


Read the training data and the test dataset. cell_ontology_class_reannotated is the attribute of labels in the h5ad file::

	test_data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-droplet.h5ad'
	test_X, test_genes, test_AnnData = read_data(feature_file=test_data_file, tp2i = tp2i, return_AnnData = True)
	print (np.shape(test_X))
	data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-facs_cell_ontology.h5ad'
	train_X, train_genes, train_Y = read_data(feature_file=data_file, tp2i = tp2i, AnnData_label='cell_ontology_class_reannotated')
	print (np.shape(train_X))


Train the model. Here, we trained a model from scratch. To train your own model, please set use_pretrain = None.:

	model_name = 'human'
	model_path = '../../../OnClass_data/pretrain/' + model_name
	OnClassModel.train(train_X, train_Y, tp2emb, train_genes, nhidden=[500], max_iter=20, use_pretrain = None, save_model =  model_path)


Predict the labels of cells in the test data and write it to the test anndata object. ::

	test_label = OnClassModel.predict(test_X, test_genes,log_transform=True,correct_batch=False)
	x = write_anndata_data(test_label, test_AnnData, i2tp, name_mapping_file='../../../OnClass_data/cell_ontology/cl.obo')#output_file is optional
	print (x.obs['OnClass_annotation_ontology_ID'])
	print (x.obs['OnClass_annotation_ontology_name'])

Data Integration (integrate 26-datasets using OnClass)
----------------

A example script `DataIntegration.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/DataIntegration/DataIntegration.py>`__ for transferring cell type annotation is at our `GitHub <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/DataIntegration/DataIntegration.py>`__

Load the pre-computed scoring matrix (see the above section for detail).::

	test_Y_pred = np.load(OUTPUT_DIR + '26_datasets_predicted_score_matrix.npy')
	datasets, genes_list, n_cells = load_names(data_names_all,verbose=False,log1p=True, DATA_DIR=DATA_DIR)
	datasets, genes = merge_datasets(datasets, genes_list)

Integration based on our method.::

	pca = PCA(n_components=50)
	test_Y_pred_red = pca.fit_transform(test_Y_pred[:, :nseen])

Please check `DataIntegration.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/DataIntegration/DataIntegration.py>`__ for how to obtain the UMAP plots.


Marker genes identification
----------------

A example script `FindMarkerGenes.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/MarkerGenesIdentification/FindMarkerGenes.py>`__ for transferring cell type annotation is at our `GitHub <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/MarkerGenesIdentification/FindMarkerGenes.py>`__

To find maker genes, we first train on all FACS cells and then generate the scoring matrix for all FACS cells.::

	train_X, train_Y_str, genes_list = read_data(filename=data_file, return_genes=True)
	tms_genes_list = [x.upper() for x in list(genes_list.values())[0]]
	ntrain,ngene = np.shape(train_X)
	## embedd the cell ontology
	unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, co_dim = 200, co_mi = 0)
	train_Y = MapLabel2CL(train_Y_str, l2i)

	## train and predict
	OnClass_obj = OnClassPred()
	OnClass_obj.train(train_X, train_Y, Y_emb, max_iter=20, nhidden=[100])
	test_Y_pred = OnClass_obj.predict(test_X)

	np.save(OUTPUT_DIR + 'FACS-predicted_score_matrix.npy', test_Y_pred)


Differential expression analysis.::

	ncell = np.shape(test_Y_pred)[0]
	co2name, name2co = get_ontology_name()
	tp2genes = read_type2genes(g2i)
	thres = np.array(range(1,1000))
	topk = 50
	in_tms_ranks = []
	not_tms_ranks = []
	n_in_tms =0
	for tp in tp2genes:
		ci = l2i[tp]
		k_bot_cells = np.argsort(test_Y_pred[:,ci])[:topk]
		k_top_cells = np.argsort(test_Y_pred[:,ci])[ncell-topk:]
		pv = scipy.stats.ttest_ind(train_X[k_top_cells,:], train_X[k_bot_cells,:], axis=0)[1]
		top_mean = np.mean(train_X[k_top_cells,:],axis=0)
		bot_mean = np.mean(train_X[k_bot_cells,:],axis=0)
		for g in range(ngene):
			if top_mean[0,g] < bot_mean[0,g]:
				pv[g] = 1.
		pv_sort = list(np.argsort(pv))

Here, `pv_sort` is the rank list of marker genes for each cell type.

Please check `FindMarkerGenes.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/MarkerGenesIdentification/FindMarkerGenes.py>`__ for how to marker genes. Please check `Marker_genes_based_prediction_droplet.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/MarkerGenesIdentification/Marker_genes_based_prediction_droplet.py>`__  and `Marker_genes_based_prediction_26_datasets.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/MarkerGenesIdentification/Marker_genes_based_prediction_26_datasets.py>`__  for how to use these marker genes to predict cell types for cells in TMS droplets and 26-datasets.
