import os
import numpy as np
import pandas as pd
import time as tm
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def run_svm_rejection(train_X, test_X, train_Y, OutputFile, Threshold = 0.7, reject = True):

	train_X = np.log1p(train_X)
	test_X = np.log1p(test_X)

	Classifier = LinearSVC()
	clf = CalibratedClassifierCV(Classifier)

	clf.fit(train_X, train_Y)

	test_Y = clf.predict(test_X)
	prob = clf.predict_proba(test_X)
	ref_prob = np.max(clf.predict_proba(test_X), axis = 1)
	unlabeled = np.where(ref_prob < Threshold)
	if reject:
		test_Y[unlabeled] = -1
	np.savetxt(OutputFile+'_vec', test_Y)
	np.savetxt(OutputFile+'_mat', prob)

	return test_Y, prob
