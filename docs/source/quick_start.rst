Quick start
=========
Here, we provide an introduction of how to use OnClass

Cell type annotation
----------------

A example script `CellTypeAnnotation_TMS.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/CellTypeAnnotation_TMS.py>`__ for cell type annotation is at our `GitHub <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/CellTypeAnnotation_TMS.py>`__

Import OnClass as::

	from OnClass.utils import *
	from OnClass.OnClassPred import OnClassPred
	

Read the single cell data as. Please change the DATA_DIR to where you store the downloaded data. Here we use TMS FACS cells as an example.::
    
	DATA_DIR = '../../OnClass_data/'
	data_file = DATA_DIR + 'raw_data/tabula-muris-senis-facs'
	X, Y = read_data(filename=data_file)
	train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(X, Y)
	
where `X` is a sample by gene gene expression matrix, `Y` is a label vector for each sample. The labels in `Y` should use the Cell Ontology Id (e.g., CL:1000398). The data (e.g., tabula muris raw gene expression matrix, the Cell Ontology obo file) can be downloaded from FigShare.(see dataset section in this tutorial).


Embedd the cell ontology.::

	unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, DATA_DIR = DATA_DIR)
	train_Y = MapLabel2CL(train_Y_str, l2i)
	test_Y = MapLabel2CL(test_Y_str, l2i)
	unseen_l = MapLabel2CL(unseen_l, l2i)

`Y_emb` is a label by dimension matrix of the embeddings of all terms in the Cell Ontology (both seen and unseen), `MapLabel2CL` convert Cell Ontology Id to integer labels. `unseen_l` is a list of all the unseen cell types (i.e., cell types not in training data, but in test data)

Train the model::

	OnClass_obj = OnClassPred()
	OnClass_obj.train(train_X, train_Y, Y_emb, max_iter=1, nhidden=[2])
	
Predict the label for test data::

	test_Y_pred = OnClass_obj.predict(test_X)
	
`test_Y_pred` is a sample (number of test cells) by label (number of total cell types in the Cell Ontology) score matrix. A larger score means higher confidence.



Cell type annotation
----------------

A example script `CellTypeAnnotation_TMS.py <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/CellTypeAnnotation_TMS.py>`__ for cell type annotation is at our `GitHub <https://github.com/wangshenguiuc/OnClass/blob/master/scripts/CellTypeAnnotation/CellTypeAnnotation_TMS.py>`__

Import OnClass as::

	from OnClass.utils import *
	from OnClass.OnClassPred import OnClassPred
	


Read the single cell data as. Here we use TMS FACS cells as an example.::
    
	DATA_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/'
	data_file = DATA_DIR + 'raw_data/tabula-muris-senis-facs'
	X, Y = read_data(filename=data_file)
	train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(X, Y)
	
where `X` is a sample by gene gene expression matrix, `Y` is a label vector for each sample. The labels in `Y` should use the Cell Ontology Id (e.g., CL:1000398). The data (e.g., tabula muris raw gene expression matrix, the Cell Ontology obo file) can be downloaded from FigShare.(see dataset section in this tutorial) 

Embedd the cell ontology.::
	unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, DATA_DIR = DATA_DIR)
	train_Y = MapLabel2CL(train_Y_str, l2i)
	test_Y = MapLabel2CL(test_Y_str, l2i)
	unseen_l = MapLabel2CL(unseen_l, l2i)

`Y_emb` is a label by dimension matrix of the embeddings of all terms in the Cell Ontology (both seen and unseen), `MapLabel2CL` convert Cell Ontology Id to integer labels. `unseen_l` is a list of all the unseen cell types (i.e., cell types not in training data, but in test data)

Train the model::
	OnClass_obj = OnClassPred()
	OnClass_obj.train(train_X, train_Y, Y_emb, max_iter=1, nhidden=[2])
	
Predict the label for test data::
	test_Y_pred = OnClass_obj.predict(test_X)
	
`test_Y_pred` is a sample (number of test cells) by label (number of total cell types in the Cell Ontology) score matrix. A larger score means higher confidence.