import os
import numpy as np
import pandas as pd
import time as tm
from sklearn.linear_model import LogisticRegression

def run_lr(train_X, test_X, train_Y, OutputFile, Threshold = 0.7, reject = True):
	clf = LogisticRegression(random_state=0, multi_class='ovr',solver='lbfgs')
	train_X = np.log1p(train_X)
	test_X = np.log1p(test_X)
	clf.fit(train_X, train_Y)
	prob = clf.predict_proba(test_X)
	test_Y = []
	scale = 1.
	for p in prob:# loop every test prediction
		max_class = np.argmax(p)# predicted class
		max_value = np.max(p)# predicted probability
		if max_value > Threshold or not reject:
			test_Y.append(max_class)#predicted probability is greater than threshold, accept
		else:
			test_Y.append(-1)
	np.savetxt(OutputFile+'_vec', test_Y)
	np.savetxt(OutputFile+'_mat', prob)
	return test_Y, prob
