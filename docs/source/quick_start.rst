Quick start
=========
Here, we provide an introduction of how to use OnClass

Cell type annotation (cross validation on the TMS)
----------------

A example script `CellTypeAnnotation_TMS.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/CellTypeAnnotation_TMS.py>`__ for cell type annotation is at our `GitHub <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/CellTypeAnnotation_TMS.py>`__

Import OnClass as::
~~~~~~~~~~~~~~~~~~~

	from OnClass.utils import *
	from OnClass.OnClassPred import OnClassPred
	

Read the single cell data as. Please change the DATA_DIR to where you store the downloaded data. Here we use TMS FACS cells as an example.::
    
	DATA_DIR = '../../OnClass_data/'
	data_file = DATA_DIR + 'raw_data/tabula-muris-senis-facs'
	X, Y = read_data(filename=data_file)
	train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(X, Y)
	
where `X` is a sample by gene gene expression matrix, `Y` is a label vector for each sample. The labels in `Y` should use the Cell Ontology Id (e.g., CL:1000398). The data (e.g., tabula muris raw gene expression matrix, the Cell Ontology obo file) can be downloaded from FigShare.(see dataset section in this tutorial).


Embedd the cell ontology.::
~~~~~~~~~~~~~~~~~~~

	unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, DATA_DIR = DATA_DIR)
	train_Y = MapLabel2CL(train_Y_str, l2i)
	test_Y = MapLabel2CL(test_Y_str, l2i)
	unseen_l = MapLabel2CL(unseen_l, l2i)

`Y_emb` is a label by dimension matrix of the embeddings of all terms in the Cell Ontology (both seen and unseen), `MapLabel2CL` convert Cell Ontology Id to integer labels. `unseen_l` is a list of all the unseen cell types (i.e., cell types not in training data, but in test data)

Train the model::

	OnClass_obj = OnClassPred()
	OnClass_obj.train(train_X, train_Y, Y_emb)
	
Predict the label for test data::

	test_Y_pred = OnClass_obj.predict(test_X)
	
`test_Y_pred` is a sample (number of test cells) by label (number of total cell types in the Cell Ontology) score matrix. A larger score means higher confidence.



Cell type annotation (Train on TMS, Test on 26-datasets)
----------------

A example script `Annot26datasets.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/Annot26datasets.py>`__ for transferring cell type annotation is at our `GitHub <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/CellTypeAnnotation_TMS.py>`__

We use Scanorama to remove batch effects across datasets. Please install `Scanorama <https://github.com/brianhie/scanorama>`__ before running this script. Import OnClass as::

	from scanorama import *
	from OnClass.utils import *
	from OnClass.OnClassPred import OnClassPred
	from OnClass.other_datasets_utils import my_assemble, data_names_all, load_names 
	

Read TMS and 26-datasets. Here we train on TMS and then predict the labels for each cell in 26-datasets.::
    
	DATA_DIR = '../OnClass_data/'
	data_file = DATA_DIR + '/raw_data/tabula-muris-senis-facs'
	train_X, train_Y_str, genes_list = read_data(filename=data_file, DATA_DIR = DATA_DIR, return_genes=True)
	tms_genes_list = [x.upper() for x in list(genes_list.values())[0]]
	datasets, genes_list, n_cells = load_names(data_names_all,verbose=False,log1p=True, DATA_DIR=DATA_DIR)
	datasets.append(train_X)
	genes_list.append(tms_genes_list)
	data_names_all.append('TMS')

Eembedd the cell ontology.::

	unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, DATA_DIR=DATA_DIR)
	train_Y = MapLabel2CL(train_Y_str, l2i)

Use Scanorama to correct batch effects. ::

	datasets, genes = merge_datasets(datasets, genes_list)
	datasets_dimred, genes = process_data(datasets, genes, dimred=100)
	expr_datasets = my_assemble(datasets_dimred, ds_names=data_names_all, expr_datasets = datasets, sigma=150)[1]
	expr_corrected = sparse.vstack(expr_datasets)
	expr_corrected = np.log2(expr_corrected.toarray()+1)
	
Annotate 26-datasets, train on TMS. ::

	ntrain,ngene = np.shape(train_X)
	nsample = np.shape(expr_corrected)[0]
	train_X_corrected= expr_corrected[nsample-ntrain:,:]
	test_X_corrected = expr_corrected[:nsample-ntrain,:]
	OnClass_obj = OnClassPred()
	OnClass_obj.train(train_X_corrected, train_Y, Y_emb, log_transform=False)
	test_Y_pred = OnClass_obj.predict(test_X_corrected, log_transform=False)
	

Save the prediction matrix, nsample (number of samples in 26-datasets) by nlabels. ::

	np.save(output_dir + '26_datasets_predicted_score_matrix.npy', test_Y_pred)
	
After obtaining the scoring matrix, we can run `Evaluate26datasets.py.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/Evaluate26datasets.py.py>` to calculate AUROC.



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
