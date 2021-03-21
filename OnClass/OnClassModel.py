import numpy as np
import os
import sys
from sklearn.preprocessing import normalize

from OnClass.OnClass_utils import *
from OnClass.BilinearNN import BilinearNN

class OnClassModel:
	def __init__(self, cell_type_network_file='../../OnClass_data/cell_ontology/cl.ontology', cell_type_nlp_emb_file='../../OnClass_data/cell_ontology/cl.ontology.nlp.emb'):
		self.cell_type_nlp_emb_file = cell_type_nlp_emb_file
		self.cell_type_network_file = cell_type_network_file
		print ('init OnClass')
		self.co2co_graph, self.co2co_nlp, self.co2vec_nlp, self.cell_ontology_ids = read_cell_type_nlp_network(self.cell_type_nlp_emb_file, self.cell_type_network_file)
		#self.cell_type_nlp_network_file = self.cell_type_nlp_emb_file+'network'
		#fout = open(self.cell_type_nlp_network_file,'w')
		#for w1 in self.co2co_nlp:
		#	for w2 in self.co2co_nlp[w1]:
		#		fout.write(w1+'\t'+w2+'\t'+str(self.co2co_nlp[w1][w2])+'\n')
		#fout.close()

	def EmbedCellTypes(self, train_Y_str, dim=5, emb_method=3, use_pretrain = None, write2file=None):
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
		co2emb, co2i, i2co
			returns three dicts, cell type name to embeddings, cell type name to cell type id and cell type id to embeddings.
		"""

		self.unseen_co, self.co2i, self.i2co, self.ontology_dict, self.ontology_mat = creat_cell_ontology_matrix(train_Y_str, self.co2co_graph, self.cell_ontology_ids, dfs_depth = 3)
		self.nco = len(self.i2co)
		Y_emb = emb_ontology(self.i2co, self.ontology_mat, dim = dim, mi=emb_method, co2co_nlp = self.co2co_nlp, unseen_l = self.unseen_co)
		self.co2emb = np.column_stack((np.eye(self.nco), Y_emb))
		self.nunseen = len(self.unseen_co)
		self.nseen = self.nco - self.nunseen
		self.co2vec_nlp_mat = np.zeros((self.nco, len(self.co2vec_nlp[self.i2co[0]])))
		for i in range(self.nco):
			self.co2vec_nlp_mat[i,:] = self.co2vec_nlp[self.i2co[i]]
		return self.co2emb, self.co2i, self.i2co, self.ontology_mat

	def align_train_test_expression(self, train_X, test_X):
		train_X, test_X = process_expression([train_X, test_X])
		return train_X, test_X

	def BuildModel(self, label_emb, ngene, nhidden=[1000], use_pretrain=None):
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
		self.ngene = ngene
		self.use_pretrain = use_pretrain
		self.label_emb = label_emb
		self.nhidden = nhidden
		if use_pretrain is not None:
			npzfile = np.load(use_pretrain+'.npz',allow_pickle=True)
			self.co2i = npzfile['co2i'].item()
			self.i2co = npzfile['i2co'].item()
			self.genes = npzfile['genes']
			print (self.genes)
			self.ngene = len(self.genes)
			self.ontology_mat = npzfile['ontology_mat']
			self.nco = npzfile['nco']
			self.nseen = npzfile['nseen']
			self.co2vec_nlp_mat = npzfile['co2vec_nlp_mat']
			self.nhidden = npzfile['nhidden']
			self.ontology_dict = npzfile['ontology_dict'].item()
		self.model = BilinearNN(label_emb, self.nseen, self.ngene, use_pretrain = use_pretrain, nhidden=self.nhidden)
		return self.model

	def Train(self, train_feature, train_label, save_model = None, genes = None, max_iter=50, log_transform=False, minibatch_size = 128):
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
		if self.use_pretrain:
			print ('Use pretrained model: ',self.use_pretrain)
			return
		unseen_l = [self.co2i[tp] for tp in self.unseen_co]
		train_label = [self.co2i[tp] for tp in train_label]
		if log_transform:
			train_feature = np.log1p(train_feature.todense())

		self.train_X = train_feature
		self.model.read_training_data(self.train_X , train_label)
		self.model.optimize(max_iter = max_iter, minibatch_size = minibatch_size, save_model =  save_model)

		if save_model is not None:
			save_model_file = save_model + '.npz'
			np.savez(save_model_file, co2i = self.co2i, nhidden = self.nhidden, i2co = self.i2co, genes = genes, nco = self.nco, nseen = self.nseen,
			 ontology_mat = self.ontology_mat, co2vec_nlp_mat = self.co2vec_nlp_mat, ontology_dict = self.ontology_dict)

	def Predict(self, test_X, test_genes=None, log_transform=False, use_normalize=False, refine = True):
		"""
		Predict the label for new cells
		"""

		if log_transform:
			test_X = np.log1p(test_X.todense())
		if test_genes is not None:
			test_X = map_genes(test_X, test_genes, self.genes)
		else:
			assert(np.shape(test_X)[1] == self.ngene)

		test_Y_pred_seen = self.model.predict(test_X)
		test_Y_pred_all = None
		if refine:
			ratio = (self.nco*1./self.nseen)**2
			network = create_propagate_networks_using_nlp(self.co2i, self.ontology_dict, self.ontology_mat, self.co2vec_nlp_mat)
			test_Y_pred_all = extend_prediction_2unseen(test_Y_pred_seen, network, self.nseen, ratio = ratio, use_normalize = use_normalize)

		return test_Y_pred_seen, test_Y_pred_all, np.argmax(test_Y_pred_all,axis=1)
