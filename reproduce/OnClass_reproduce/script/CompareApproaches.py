import sys
from time import time
import timeit
from datetime import datetime
from scipy import stats, sparse
from scipy.sparse.linalg import svds, eigs
from scipy.special import expit
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn import metrics
from collections import Counter
from scipy.stats import norm as dist_model
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from baselines.run_svm_rejection import run_svm_rejection
from baselines.run_actinn import run_actinn
from baselines.run_doc import run_doc
from baselines.run_singlecellnet import run_singlecellnet
from baselines.run_lr import run_lr


if len(sys.argv) <= 2:
	pid = 0
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])


output_dir = OUTPUT_DIR + '/CompareBaselines/baseline_scores/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)


niter = 5
dnames = ['microcebusMartine','microcebusAntoine','microcebusBernard','microcebusStumpy','muris_facs','allen','muris_droplet']
for dname in dnames:
	for i in range(niter):
		folder = output_dir +'/'+ dname + '/'+ str(i) + '/'
		if not os.path.exists(folder):
		    os.makedirs(folder)
pct = -1
start = timeit.default_timer()
for dname in dnames:
	for iter in range(5):
		for unseen_ratio in [0.1, 0.3, 0.5, 0.7, 0.9	]:
			pct += 1
			if total_pid>1 and pct%total_pid != pid:
				continue
			output_prefix = output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio)+'_'
			#if os.path.isfile(output_prefix+'result.txt'):
			#	continue

			feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data(dname, DATA_DIR,nsample=200000000)
			print (output_prefix)
			print (dname, np.shape(feature))
			train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = iter)
			unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, co_dim = 5, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)

			train_Y = MapLabel2CL(train_Y_str, l2i)
			test_Y = MapLabel2CL(test_Y_str, l2i)
			unseen_l = MapLabel2CL(unseen_l, l2i)
			test_Y_ind = np.sort(np.array(list(set(test_Y) |  set(train_Y))))
			ncls = np.shape(cls2cls)[0]
			nunseen = len(unseen_l)
			nseen = ncls - nunseen

			ntest = np.shape(test_X)[0]
			ntrain = np.shape(train_X)[0]
			train_X_emb, test_X_emb = emb_cells(train_X, test_X, dim = 500)
			np.savetxt(output_dir + 'truth_vec', test_Y)
			print ('%s ratio:%f ntrain:%d ntest:%d nseen:%d ntrain_clas:%d ntest_clas:%d ntotal:%d' % (dname, unseen_ratio, ntrain, ntest, nseen, len(set(train_Y)), len(test_Y_ind),  ncls))
			vec = {}
			prob = {}
			print (train_Y)
			print (iter,np.unique(train_Y))
			continue

			vec['lr'], prob['lr'] = run_lr(train_X, test_X, train_Y, output_prefix + 'lr', Threshold = 0.0, reject = False)
			print ('lr', timeit.default_timer() - start)
			start = timeit.default_timer()
			print (np.shape(prob['lr']))
			vec['svm'], prob['svm'] = run_svm_rejection(train_X, test_X, train_Y, output_prefix + 'svm', Threshold = 0.7, reject = False)
			print ('svm', timeit.default_timer() - start)
			start = timeit.default_timer()

			vec['doc'], prob['doc'] = run_doc(train_X, test_X, train_Y, output_dir + 'doc', reject = False)
			print ('doc', timeit.default_timer() - start)
			start = timeit.default_timer()

			vec['actinn'], prob['actinn'] = run_actinn(train_X, test_X, train_Y, output_prefix + 'actinn')
			print ('actinn', timeit.default_timer() - start)
			start = timeit.default_timer()

			start = timeit.default_timer()
			vec['singlecellnet'], prob['singlecellnet'] = run_singlecellnet(train_X_emb, test_X_emb, train_Y,  ncls, output_prefix + 'singlecellnet', reject = False)
			print ('singlecellnet', timeit.default_timer() - start)
			print (vec['singlecellnet'])

			fout = open(output_prefix+'result.txt','w')
			start = timeit.default_timer()
			for method in vec:
				mat = np.zeros((len(vec[method]), ncls))
				mat[:,:nseen] = prob[method]
				res = evaluate(mat, test_Y, unseen_l, nseen, Y_pred_vec = vec[method],Y_ind = test_Y_ind, Y_net = onto_net, write_screen = True, write_to_file= fout, prefix = method)
			fout.close()
