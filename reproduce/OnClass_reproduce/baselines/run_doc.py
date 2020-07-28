import os
import numpy as np
import pandas as pd
import time as tm
from scipy.stats import norm as dist_model
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
def predict_proba(clf, X):
	prob = clf.decision_function(X)
	expit(prob, out=prob)
	return prob

def fit_gaussian(prob_pos_X):
	prob_pos = [p for p in prob_pos_X]+[2-p for p in prob_pos_X]
	pos_mu, pos_std = dist_model.fit(prob_pos)
	return pos_mu, pos_std


def run_doc(train_X, test_X, train_Y, OutputFile,seed=0):
	np.random.seed(seed)
	nseen = len(np.unique(train_Y))
	train_X = np.log1p(train_X)
	test_X = np.log1p(test_X)
	clf = LogisticRegression(random_state=0, multi_class='ovr',solver='lbfgs')#
	clf.fit(train_X, train_Y)
	train_Y_pred = predict_proba(clf, train_X)

	mu_stds = []
	for i in range(nseen):
		pos_mu, pos_std = fit_gaussian(train_Y_pred[train_Y==i, i])
		mu_stds.append([pos_mu, pos_std])
	prob = predict_proba(clf, test_X)
	test_Y = []
	scale = 1.
	for p in prob:# loop every test prediction
		max_class = np.argmax(p)# predicted class
		max_value = np.max(p)# predicted probability
		threshold = max(0.5, 1. - scale * mu_stds[max_class][1])#find threshold for the predicted class
		if max_value > threshold:
			test_Y.append(max_class)#predicted probability is greater than threshold, accept
		else:
			test_Y.append(-1)
	np.savetxt(OutputFile+'_vec', test_Y)
	np.savetxt(OutputFile+'_mat', prob)
	return test_Y, prob
