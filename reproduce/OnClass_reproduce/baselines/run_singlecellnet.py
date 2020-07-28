from anndata import read_h5ad
import numpy as np
import pandas as pd
import scipy.io
import os
import subprocess
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/tools/single_cell/actinn/ACTINN/'
os.chdir(repo_dir)

def run_singlecellnet(train_X, test_X, train_Y, ncls, OutputFile, reject = True, ntrees=20, suffix='',data_dir='/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/OnClass/CompareBaselines/singlecellnet/',working_dir='/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/SCTypeClassifier/Baselines/'):
	#

	if not os.path.exists(data_dir):
	    os.makedirs(data_dir)
	nseen = len(np.unique(train_Y))
	test_file = OutputFile+'scn_test_feat_tmp.csv'
	train_file = OutputFile+ 'scn_train_feat_tmp.csv'
	train_label_file = OutputFile+ 'scn_train_lab_tmp.csv'
	return_file = OutputFile+ 'scn_test_lab_tmp.csv'
	ntrain, ngene = np.shape(train_X)
	ntest, ngene = np.shape(test_X)
	genes = ['gene'+str(s) for s in range(ngene)]
	train_ind = ['train_ind'+str(s) for s in range(ntrain)]
	test_ind = ['test_ind'+str(s) for s in range(ntest)]
	train_ind = np.array(train_ind)
	test_ind = np.array(test_ind)
	genes = np.array(genes)
	#test_X = test_X.todense()
	#train_X = train_X.todense()
	train_label = pd.DataFrame(data = train_Y, index = train_ind)
	test_set = pd.DataFrame(data = test_X, columns = genes, index = test_ind)
	train_set = pd.DataFrame(data = train_X, columns = genes, index = train_ind)


	test_set.to_csv(test_file, sep=",", index=True)
	train_set.to_csv(train_file, sep=",", index=True)
	train_label.to_csv(train_label_file, sep=",", index=True)

	cmd = 'Rscript %ssingleCellNet.R %s %s %s %s %d' % (working_dir,test_file,train_file,train_label_file,return_file,ntrees)
	py2output  = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
	#print (py2output)

	test_Y_pred_table = pd.read_csv(return_file, header=0, sep=",")
	os.remove(test_file)
	os.remove(train_file)
	os.remove(train_label_file)
	os.remove(return_file)
	#os.remove(OutputFile+ 'scn_test_lab_tmp.csv')
	ncell = np.shape(test_X)[0]
	test_Y_pred_raw = np.zeros((ncell, nseen+1))
	test_Y_pred_all = np.zeros((ncell, ncls))

	for ind in test_Y_pred_table.index:
		if ind!='rand':
			test_Y_pred_raw[:,int(ind)] = test_Y_pred_table.loc[ ind, test_ind]
			test_Y_pred_all[:,int(ind)] = test_Y_pred_table.loc[ ind, test_ind]
		else:
			test_Y_pred_raw[:,nseen] = test_Y_pred_table.loc[ ind, test_ind]

	test_Y = np.argmax(test_Y_pred_raw, axis=1)
	for i in range(len(test_Y)):
		if test_Y[i] == nseen:
			if reject:
				test_Y[i] = -1
			else:
				test_Y[i] = np.argmax(test_Y_pred_raw[i,:nseen])
	test_Y_pred_raw[:,nseen] = 0
	prob = test_Y_pred_raw[:,:nseen]
	np.savetxt(OutputFile+'_vec', test_Y)
	np.savetxt(OutputFile+'_mat', prob)
	return test_Y, prob
