import numpy as np
import os
import sys
from sklearn.preprocessing import normalize
from BilinearNN import BilinearNN

class OnClass:
	def __init__(self):
		self.model = BilinearNN()

	def train(self, train_X, train_Y, Y_emb, nhidden=[500], max_iter=500, minibatch_size=32, lr = 0.0001, l2=0.01, log_transform=True):
		if log_transform:
			train_X = np.log1p(train_X.todense())
		self.nlabel = np.shape(Y_emb)[0]
		self.nseen = len(np.unique(train_Y))
		train_Y_pred = self.model.train(train_X, train_Y, Y_emb, self.nlabel,  nhidden=nhidden, max_iter=max_iter, minibatch_size=minibatch_size, lr = lr, l2= l2)
		return train_Y_pred

	def predict(self, test_X, log_transform=True, normalize=True):
		if log_transform:
			test_X = np.log1p(test_X.todense())
		test_Y_pred = self.model.predict(test_X)
		if normalize:
			test_Y_pred = self.unseen_normalize(test_Y_pred)
		return test_Y_pred

	def unseen_normalize(self, test_Y_pred):
		ratio = self.nlabel * 1. / self.nseen
		test_Y_pred[:,:self.nseen] = normalize(test_Y_pred[:,:self.nseen],norm='l1',axis=1)
		test_Y_pred[:,self.nseen:] = normalize(test_Y_pred[:,self.nseen:],norm='l1',axis=1) * ratio
		return test_Y_pred
