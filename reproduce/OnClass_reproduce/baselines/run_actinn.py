import os
import numpy as np
import pandas as pd
import time as tm
from baselines.ACTINN import run_ACTINN

def run_actinn(train_X, test_X, train_Y, OutputFile, Threshold = 0.7,num_epochs=50):

	ncls = np.max(train_Y) + 1
	prob = run_ACTINN(train_X, np.array(train_Y), test_X, ncls,num_epochs=num_epochs) #ACTINN will do log2 by itself
	test_Y = np.argmax(prob, axis = 1)
	np.savetxt(OutputFile+'_vec', test_Y)
	np.savetxt(OutputFile+'_mat', prob)
	return test_Y, prob
