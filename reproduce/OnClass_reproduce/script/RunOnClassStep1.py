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
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn import metrics
from collections import Counter
from sklearn.preprocessing import normalize
from scipy.stats import norm as dist_model
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from model.OnClassPred import OnClassPred
from model.BilinearNNDebug import BilinearNN
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

output_dir = OUTPUT_DIR + '/CompareBaselines/TuneOnclass/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

#dnames = ['allen_part','microcebus_smartseq2','microcebus_10x','muris_facs','muris_droplet']
nn_nhidden = [100,50,25]
co_dim = 500
keep_prob = 0.7
niter = 5
dnames = ['microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy','muris_facs','muris_droplet','allen']
for dname in dnames:
	for i in range(niter):
		folder = output_dir +'/'+ dname + '/'+ str(i) + '/'
		if not os.path.exists(folder):
		    os.makedirs(folder)
pct = -1
start = timeit.default_timer()

for dname in dnames:
	if 'allen' in dname:
		ontology_dir =  OUTPUT_DIR+'/Ontology/CellTypeOntology/'
		co_dim = 100
	else:
		ontology_dir =   OUTPUT_DIR+'/Ontology/CellOntology/'
	if not os.path.exists(ontology_dir):
		os.makedirs(ontology_dir)
	for iter in range(niter):
		for unseen_ratio in [0.1,0.3,0.5,0.7,0.9]:
			pct += 1
			if total_pid>1 and pct%total_pid != pid:
				continue
			folder = output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio) + '/'
			if not os.path.exists(folder):
				os.makedirs(folder)
			pname = translate_paramter([nn_nhidden,keep_prob])

			output_prefix = folder + pname
			print (output_prefix)
			#if os.path.isfile(output_prefix+'pred_Y_seen.npy'):
			#	continue
			ontology_emb_file = ontology_dir + str(co_dim)

			feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data(dname, DATA_DIR,nsample=10000000)

			train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = iter)

			unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, co_dim = co_dim, use_pretrain = ontology_emb_file, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
			ncls = np.shape(cls2cls)[0]

			train_Y = MapLabel2CL(train_Y_str, l2i)
			test_Y = MapLabel2CL(test_Y_str, l2i)

			unseen_l = MapLabel2CL(unseen_l, l2i)
			test_Y_ind = np.sort(np.array(list(set(test_Y) |  set(train_Y))))

			nunseen = len(unseen_l)
			nseen = ncls - nunseen
			ntest = np.shape(test_X)[0]
			ntrain = np.shape(train_X)[0]
			print (ntest,ntrain)
			Y2Y = cosine_similarity(Y_emb)
			Y2Y_ngh = knn_ngh(Y2Y[:,:nseen])
			#print (Y2Y_ngh)
			#print (Y2Y)

			assert(np.shape(Y2Y)[0] == ncls)
			assert(np.shape(Y2Y_ngh)[1] == nseen)

			train_X, test_X = process_expression([train_X, test_X])
			BN = BilinearNN()
			BN.train(train_X, train_Y, Y_emb, nseen, use_y_emb = False,
			nhidden = nn_nhidden, max_iter = 50, minibatch_size = 128, lr = 0.0001, l2=0.005, keep_prob = 1.0, early_stopping_step = 5)
			pred_Y_seen = BN.predict(test_X)
			np.save(output_prefix+'pred_Y_seen.npy',pred_Y_seen)
