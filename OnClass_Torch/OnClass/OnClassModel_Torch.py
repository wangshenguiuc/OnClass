import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
from scipy.sparse.csr import csr_matrix
from sklearn.preprocessing import normalize
from scipy import stats
import warnings
from torch.optim import optimizer
from OnClass.OnClass_utils import *
from OnClass.BilinearNN_Torch import *



class OnClassModel:
	"""
	PyTorch implementation of the OnClass model
	"""
	def __init__(self, cell_type_network_file='../../OnClass_data/cell_ontology/cl.ontology', cell_type_nlp_emb_file='../../OnClass_data/cell_ontology/cl.ontology.nlp.emb', memory_saving_mode=False):
		"""
		Initialize OnClass model with a given cell-type network and cell-type embedding file.
		Also, you may set the memory_saving_mode to True to get a model that uses less RAM.
		"""
		self.cell_type_nlp_emb_file = cell_type_nlp_emb_file
		self.cell_type_network_file = cell_type_network_file
		self.co2co_graph, self.co2co_nlp, self.co2vec_nlp, self.cell_ontology_ids = read_cell_type_nlp_network(self.cell_type_nlp_emb_file, self.cell_type_network_file)
		self.mode = memory_saving_mode

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

	def BuildModel(self, ngene, nhidden=[1000], use_pretrain=None):
		"""
		Initialize the model or use the pretrain model
		Parameters
		----------
		ngene: `int`
			Number of genes
		nhidden: `list`, optional ([1000])
			Gives the hidden dimensions of the model
		use_pretrain: `string`, optional (None)
			File name of the pretrained model
		Returns
		-------
		"""
		self.ngene = ngene
		self.use_pretrain = use_pretrain
		self.nhidden = nhidden
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		if use_pretrain is not None:
			# Load all the OnClassModel state
			npzfile = np.load(use_pretrain+'.npz',allow_pickle=True)
			self.co2i = npzfile['co2i'].item()
			self.i2co = npzfile['i2co'].item()
			self.genes = npzfile['genes']
			self.co2emb = npzfile['co2emb']
			self.ngene = len(self.genes)
			self.ontology_mat = npzfile['ontology_mat']
			self.nco = npzfile['nco']
			self.nseen = npzfile['nseen']
			self.co2vec_nlp_mat = npzfile['co2vec_nlp_mat']
			self.nhidden = npzfile['nhidden']
			self.ontology_dict = npzfile['ontology_dict'].item()
			self.train_feature_mean = npzfile['train_feature_mean']
		
		self.model = BilinearNN(self.co2emb, self.nseen, self.ngene, use_pretrain = use_pretrain, nhidden=self.nhidden, memory_saving_mode = self.mode)
		
		if use_pretrain is not None:
			# Load the actual PyTorch model parameters
			self.model.load_state_dict(torch.load(use_pretrain + '.pt'))

		return self.model

	def ProcessTrainFeature(self, train_feature, train_label, train_genes, test_feature = None, test_genes = None, batch_correct = False, log_transform = True):
		"""
		Process the gene expression matrix used to train the model, and optionally the test data.
		Parameters
		----------
		train_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
			gene expression matrix of cell types
		train_label: `numpy.ndarray`
			labels for the training features
		train_genes: `list`
			list of genes used during training
		test_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode), optional (None)
			gene expression matrix of cell types for the test set
		test_genes: `list`, optional (None)
			list of genes used in test set
		batch_correct: `bool`, optional (False)
			whether to correct for batch effect in data
		log_transform:`bool`, optional (True)
			whether to apply log transform to data
		Returns
		-------
		train_feature, test_feature, self.genes, self.genes
			returns the training feature gene expression matrix and the list of genese associated
			with it. If test_feature was not none, also returns the test features and their genes.
		"""
		
		if log_transform is False and np.max(train_feature) > 1000:
			warnings.warn("Max expression is"+str(np.max(train_feature))+'. Consider setting log transform = True\n')
		self.genes = train_genes
		# batch correction is currently not supported for memory_saving_mode
		if batch_correct and test_feature is not None and test_genes is not None and self.mode:
			train_feature, test_feature, selected_train_genes = process_expression(train_feature, test_feature, train_genes, test_genes)
			self.genes = selected_train_genes
		elif log_transform:
			if self.mode:
				train_feature = csr_matrix.log1p(train_feature)
			else:
				train_feature = np.log1p(train_feature)

			if test_feature is not None:
				if self.mode:
					test_feature = csr_matrix.log1p(test_feature)
				else:
					test_feature = 	np.log1p(test_feature)
		self.train_feature = train_feature
		self.train_label = train_label
		if test_feature is not None:
			return train_feature, test_feature, self.genes, self.genes
		else:
			return train_feature, self.genes

	def ProcessTestFeature(self, test_feature, test_genes, use_pretrain = None, batch_correct = False, log_transform = True):
		"""
		Process the gene expression matrix used to test the model.
		Parameters
		----------
		test_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
			gene expression matrix of cell types for the test set
		test_genes: `list`
			list of genes used in test set
		use_pretrain: `string`, optional (None)
			name of the pretrained model
		batch_correct: `bool`, optional (False)
			whether to correct for batch effect in data
		log_transform:`bool`, optional (True)
			whether to apply log transform to data
		Returns
		-------
		gene_mapping or test_feature
			processes the test features and returns a data structure that encodes the gene
			expression matrix that should be used for testing. If the model is in memory saving
			mode, then the function will return a tuple of gene expression matrix and index array,
			otherwise, it will just return the matrix.
		"""		
		if log_transform is False and np.max(test_feature) > 1000:
			warnings.warn("Max expression is"+str(np.max(test_feature))+'. Consider setting log transform = True\n')
		
		if use_pretrain is not None:
			if log_transform:
				test_feature = np.log1p(test_feature)
			if batch_correct and not self.mode:
				test_feature = mean_normalization(self.train_feature_mean, test_feature)

		if self.mode:
			gene_mapping = get_gene_mapping(test_genes, self.genes)
			return test_feature, gene_mapping
		else:
			test_feature = map_genes(test_feature, test_genes, self.genes, memory_saving_mode=self.mode)
			return test_feature


	def Train(self, train_feature, train_label, save_model = None, max_iter=15, minibatch_size = 128, use_device = True):
		"""
		Train the model or use the pretrain model
		Parameters
		----------
		train_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
			gene expression matrix of cell types. Should be a cell by gene expression matrix
		train_label: `numpy.ndarray`
			labels for the training features. Should be a cell by class number binary matrix.
		save_model: `string`, optional (None)
			name of file to save model to.
		max_iter: `int`, optional (15)
			number of epochs to train for.
		minibatch_size: `int`, optional (128)
			size of each minibatch.
		use_device: `bool`, optional (True)
			whether to use the cpu or gpu that is avaible to this machine.
		"""
		if self.use_pretrain:
			print ('Use pretrained model: ', self.use_pretrain)
			return
		
		train_label = [self.co2i[tp] for tp in train_label]

		if use_device:
			print("Using", self.device, "for training")
			self.model.to(self.device)

		self.train_feature_mean = np.mean(train_feature, axis = 0)
		self.model.read_training_data(train_feature, train_label)
		self.model.optimize(max_iter = max_iter, mini_batch_size = minibatch_size, device=self.device)

		if save_model is not None:
			torch.save(self.model.state_dict(), save_model + '.pt')
			save_model_file = save_model + '.npz'
			np.savez(save_model_file, train_feature_mean = self.train_feature_mean, co2i = self.co2i, co2emb = self.co2emb, nhidden = self.nhidden, i2co = self.i2co, genes = self.genes, nco = self.nco, nseen = self.nseen,
			 ontology_mat = self.ontology_mat, co2vec_nlp_mat = self.co2vec_nlp_mat, ontology_dict = self.ontology_dict)

	def Predict(self, test_feature, test_genes=None, use_normalize=False, refine = True, unseen_ratio = 0.1, use_device = True):
		"""
		Predict the label for new cells
		Parameters
		----------
		test_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
			gene expression matrix of cell types for the test set
		test_genes: `list`, optional (None)
			list of genes used in test set
		use_device: `bool`, optional (True)
			whether to use the cpu or gpu that is avaible to this machine.
		"""
		if use_device:
			print("Using", self.device, "for predicting")
			self.model.to(self.device)

		if test_genes is not None:
			if not self.mode:
				test_feature = map_genes(test_feature, test_genes, self.genes, memory_saving_mode=self.mode)
			else:
				mapping = get_gene_mapping(test_genes, self.genes)
		else:
			assert(np.shape(test_feature)[1] == self.ngene)
		
		self.model.eval()
		with torch.no_grad():
			if not self.mode:
				test_Y_pred_seen = softmax(self.model.forward(test_feature, training=False, device=self.device), axis=1)
			else: # The test set will be in sparse matrix format
				num_batches = 20
				test_Y_pred_seen = softmax(self._batch_predict(test_feature, num_batches, mapping=mapping), axis=1)
		
		test_Y_pred_all = None
		if refine:
			ratio = (self.nco*1./self.nseen)**2
			network = create_propagate_networks_using_nlp(self.co2i, self.ontology_dict, self.ontology_mat, self.co2vec_nlp_mat)
			test_Y_pred_all = extend_prediction_2unseen(test_Y_pred_seen, network, self.nseen, ratio = ratio, use_normalize = use_normalize)
		if unseen_ratio>0:
			unseen_confidence = np.max(test_Y_pred_all[:,self.nseen:], axis=1) - np.max(test_Y_pred_all[:,:self.nseen], axis=1)
			nexpected_unseen = int(np.shape(test_Y_pred_seen)[0] * unseen_ratio) + 1
			unseen_ind = np.argpartition(unseen_confidence, -1 * nexpected_unseen)[-1 * nexpected_unseen:]
			seen_ind = np.argpartition(unseen_confidence, -1 * nexpected_unseen)[:-1 * nexpected_unseen]

			test_Y_pred_all[unseen_ind, :self.nseen] -= 1000000
			test_Y_pred_all[seen_ind, self.nseen:] -= 1000000
			test_Y_pred_all[:,self.nseen:] = stats.zscore(test_Y_pred_all[:,self.nseen:], axis = 0)
		return test_Y_pred_seen, test_Y_pred_all, np.argmax(test_Y_pred_all,axis=1)
		
	def _batch_predict(self, X, num_batches, mapping=None):
		"""
		Predicts the type of each cell in the test data, X, in batches.
		"""

		ns = X.shape[0]
		Y = np.zeros((ns, self.nseen))
		batch_size = int(X.shape[0] / num_batches)
		for k in range(0, num_batches):
			X_array = X[k * batch_size : (k+1) * batch_size,:].todense()
			# Remaps genes to match the test set
			X_array = X_array[:,mapping]
			Y[k * batch_size : (k+1) * batch_size,:] = self.model.forward(X_array, training=False, device=self.device)
			
		# handling the end case (last batch < batch_size)
		if ns % batch_size != 0:
			X_array = X[num_batches * batch_size : ns,:].todense()
			# Remaps genes to match the test set
			X_array = X_array[:,mapping]
			Y[num_batches * batch_size : ns,:] = self.model.forward(X_array, training=False)
		
		return Y
