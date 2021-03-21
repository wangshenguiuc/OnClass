import numpy as np
import os
import sys
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise
from utils import *
from BilinearNN import BilinearNN

class OnClassModel:
	def __init__(self, cell_type_network_file='../../OnClass_data/cell_ontology/cl.ontology', cell_type_nlp_emb_file='../../OnClass_data/cell_ontology/cl.ontology.nlp.emb'):
		self.cell_type_nlp_emb_file = cell_type_nlp_emb_file
		self.cell_type_network_file = cell_type_network_file
		print ('init OnClass')
		self.co2co_graph, self.co2co_nlp, self.co2vec_nlp = read_cell_type_nlp_network(self.cell_type_nlp_emb_file, self.cell_type_network_file)
		self.cell_ontology_ids = set(self.co2co_graph.keys())
		self.cell_type_nlp_network_file = self.cell_type_nlp_emb_file+'network'
		fout = open(self.cell_type_nlp_network_file,'w')
		for w1 in self.co2co_nlp:
			for w2 in self.co2co_nlp[w1]:
				fout.write(w1+'\t'+w2+'\t'+str(self.co2co_nlp[w1][w2])+'\n')
		fout.close()

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

		unseen_l, l2i, i2l, train_X2Y, self.onto_net, self.cls2cls = create_labels(train_Y_str, combine_unseen = False, ontology_nlp_file = self.cell_type_nlp_network_file, ontology_file = self.cell_type_network_file)
		Y_emb = emb_ontology(i2l, dim = dim, mi=emb_method,  ontology_nlp_file = self.cell_type_nlp_network_file, ontology_file = self.cell_type_network_file, use_pretrain = use_pretrain, use_seen_only = True, unseen_l = unseen_l)
		Y_emb = np.column_stack((np.eye(len(i2l)), Y_emb))
		#return unseen_l, l2i, i2l, onto_net, Y_emb, onto_net_mat
		self.co2emb = Y_emb
		self.co2i = l2i
		self.i2co = i2l
		return self.co2emb, self.co2i,  self.i2co

	def train(self, train_feature, train_label, label_emb, genes, model = 'BilinearNN',early_stopping_step = 5, save_model = None, nhidden=[500], max_iter=20, minibatch_size=128, lr = 0.0001, l2=0.005, use_pretrain=None, pretrain_expression = None, log_transform=True):
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

		train_label = [self.co2i[tp] for tp in train_label]
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
			self.train_BilinearNN(train_feature, train_label, label_emb, save_model = save_model, early_stopping_step = early_stopping_step, use_pretrain=use_pretrain, nhidden=nhidden, max_iter=max_iter, minibatch_size=minibatch_size, lr = lr, l2=l2)
			print ('training finished')

	def train_BilinearNN(self, train_X, train_Y, Y_emb, nhidden=[1000], keep_prob = 1.0, save_model = None, early_stopping_step = 5, max_iter=20, minibatch_size=128, lr = 0.0001, l2=0.005, use_pretrain=None):
		"""
		Train a bilinear model
		"""

		self.nlabel = np.shape(Y_emb)[0]
		self.nseen = len(np.unique(train_Y))
		train_Y_pred = self.model.train(train_X, train_Y, Y_emb, self.nseen, use_y_emb = True,
		nhidden = nhidden, max_iter = max_iter, minibatch_size = minibatch_size, lr = lr, l2 = l2,
		keep_prob = keep_prob, early_stopping_step = early_stopping_step, use_pretrain = use_pretrain, save_model =  save_model)

		#train_Y_pred = self.model.train(train_X, train_Y, Y_emb, self.nlabel, save_model = save_model, use_pretrain=use_pretrain,  nhidden=nhidden, max_iter=max_iter, minibatch_size=minibatch_size, lr = lr, l2= l2)
		return train_Y_pred

	def predict(self, test_X, test_genes, use_existing_model=None, correct_batch = False, log_transform=True, use_normalize=True):
		"""
		Predict the label for new cells
		"""

		if log_transform:
			test_X = np.log1p(test_X.todense())
		if correct_batch:
			test_X = run_scanorama(test_X, test_genes, self.train_X, self.genes)
		else:
			test_X = map_genes(test_X, test_genes, self.train_X, self.genes)

		test_Y_pred_seen = self.model.predict(test_X)
		print (np.shape(test_X))
		ratio = 1.#(self.nlabel*1./self.nseen)#**2
		network = create_propagate_networks(self.co2i, self.onto_net, self.cls2cls, self.cell_type_nlp_network_file)
		test_Y_pred_all = extend_prediction_2unseen(test_Y_pred_seen, network, self.nseen, ratio = ratio, use_normalize = use_normalize)

		return test_Y_pred_all

	def predict_impute(self, cell2label, labels, co2i, co2emb, knn=5):
		ncell, nlabel = np.shape(cell2label)
		assert(nlabel == len(labels))
		all_label = len(co2i)
		c2l = np.zeros((ncell, all_label))
		labid = []
		for l in labels:
			labid.append(co2i[l])
		labid = np.array(labid)
		c2l[:, labid] = cell2label
		tp2tp = pairwise.cosine_similarity(co2emb, co2emb)
		c2l = impute_knn(c2l, labid, tp2tp, knn=knn)
		return c2l
