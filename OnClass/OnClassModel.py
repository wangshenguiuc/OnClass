import numpy as np
import os
import sys
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise
#from OnClass.BilinearNN import BilinearNN
#from OnClass.utils import *
sys.path.append('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass/OnClass/')
#from OnClass.utils import *
#from OnClass.OnClassModel import OnClassModel
#from OnClass.other_datasets_utils import my_assemble, data_names_all, load_names
from utils import *
from BilinearNN import BilinearNN
from other_datasets_utils import my_assemble, data_names_all, load_names, run_scanorama


class OnClassModel:
	def __init__(self):
		print ('init OnClass')


	def EmbedCellTypes(self, cell_type_network_file='../../OnClass_data/cell_ontology/cl.ontology', dim=500, emb_method=3, use_pretrain = None, write2file=None):
		"""
		Embed the cell ontology
		Parameters
		----------
		cell_type_network_file : each line should be cell_type_1\tcell_type_2\tscore for weighted network or cell_type_1\tcell_type_2 for unweighted network
		dim: `int`, optional (500)
			Dimension of the cell type embeddings
		emb_method: `int`, optional (3)
			dimensionality reduction method
		use_pretrain: `string`, optional (None)
			use pretrain file. This should be the numpy file of cell type embeddings. It can read the one set in write2file parameter.
		write2file: `string`, optional (None)
			write the cell type embeddings to this file path
		Returns
		-------
		tp2emb, tp2i, i2tp
			returns three dicts, cell type name to embeddings, cell type name to cell type id and cell type id to embeddings.
		"""
		self.tp2emb, self.tp2i, self.i2tp, _ = cal_ontology_emb(dim=dim, mi=emb_method, cell_type_network_file = cell_type_network_file, write2file = write2file, use_pretrain = use_pretrain)
		return self.tp2emb, self.tp2i,  self.i2tp

	def train(self, train_feature, train_label, label_emb, genes, model = 'BilinearNN', save_model = None, nhidden=[500], max_iter=20, minibatch_size=32, lr = 0.0001, l2=0.01, use_pretrain=None, pretrain_expression = None, log_transform=True):
		"""
		Train the model or use the pretrain model
		Parameters
		----------
		train_feature : cell by gene matrix
		train_label: cell by nclass binarize matrix.
		label_emb: embedding vectors of labels (classes)
		genes: gene names of each column in the train feature
		Returns
		-------
		"""
		
		train_label = [self.tp2i[tp] for tp in train_label]
		if log_transform:
			train_feature = np.log1p(train_feature.todense())
		if use_pretrain is not None:
			pretrain_exp = pretrain_expression + '.exp.npy'
			pretrain_genes = pretrain_expression + '.genes.npy'
			train_feature = np.load(pretrain_exp)
			self.genes = np.load(pretrain_genes, allow_pickle = True)
		if save_model is not None:
			save_model_exp = save_model + '.exp.npy'
			save_model_genes = save_model + '.genes.npy'
			np.save(save_model_exp, train_feature)
			np.save(save_model_genes, genes)
		if model == 'BilinearNN':
			self.train_X, self.genes = train_feature, genes
			self.model = BilinearNN()
			self.train_BilinearNN(train_feature, train_label, label_emb, save_model = save_model, use_pretrain=use_pretrain, nhidden=nhidden, max_iter=max_iter, minibatch_size=minibatch_size, lr = lr, l2=l2)
			print ('training finished')

	def train_BilinearNN(self, train_X, train_Y, Y_emb, nhidden=[500], save_model = None, max_iter=20, minibatch_size=32, lr = 0.0001, l2=0.01, use_pretrain=None):
		"""
		Train a bilinear model
		"""
		
		self.nlabel = np.shape(Y_emb)[0]
		self.nseen = len(np.unique(train_Y))
		train_Y_pred = self.model.train(train_X, train_Y, Y_emb, self.nlabel, save_model = save_model, use_pretrain=use_pretrain,  nhidden=nhidden, max_iter=max_iter, minibatch_size=minibatch_size, lr = lr, l2= l2)
		return train_Y_pred

	def predict(self, test_X, test_genes, use_existing_model=None, label_networks=None, correct_batch = False, log_transform=True, normalize=True):
		"""
		Predict the label for new cells
		"""
		if log_transform:
			test_X = np.log1p(test_X.todense())
		if correct_batch:
			test_X = run_scanorama(test_X, test_genes, self.train_X, self.genes)
		else:
			test_X = map_genes(test_X, test_genes, self.train_X, self.genes)
		test_Y_pred = self.model.predict(test_X)
		if normalize:
			test_Y_pred = self.unseen_normalize(test_Y_pred)
		return test_Y_pred

	def predict_impute(self, cell2label, labels, tp2i, tp2emb, knn=5):
		ncell, nlabel = np.shape(cell2label)
		assert(nlabel == len(labels))
		all_label = len(tp2i)
		c2l = np.zeros((ncell, all_label))
		labid = []
		for l in labels:
			labid.append(tp2i[l])
		labid = np.array(labid)
		c2l[:, labid] = cell2label
		tp2tp = pairwise.cosine_similarity(tp2emb, tp2emb)
		c2l = impute_knn(c2l, labid, tp2tp, knn=knn)
		return c2l


	def unseen_normalize(self, test_Y_pred):
		ratio = self.nlabel * 1. / self.nseen
		test_Y_pred[:,:self.nseen] = normalize(test_Y_pred[:,:self.nseen],norm='l1',axis=1)
		test_Y_pred[:,self.nseen:] = normalize(test_Y_pred[:,self.nseen:],norm='l1',axis=1) * ratio
		return test_Y_pred
