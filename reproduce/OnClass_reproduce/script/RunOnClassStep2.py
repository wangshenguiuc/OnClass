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
from scipy import spatial
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from utils import *
from model.OnClassPred import OnClassPred
from model.BilinearNNDebug import BilinearNN


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
keep_prob = 0.7
niter = 5
dnames = ['microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy','muris_facs','muris_droplet','allen']
#dnames = ['muris_facs']
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

			seen_output_prefix = folder + pname
			print (pname)


			feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data(dname, DATA_DIR,nsample=100000000)

			train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = iter)
			co_dim = 5
			ontology_emb_file = ontology_dir + str(co_dim)
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
			if dname != 'allen':
				onto_net_nlp, _, onto_nlp_emb = read_cell_ontology_nlp(l2i, ontology_nlp_file = ontology_nlp_file, ontology_nlp_emb_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology.nlp.emb')
				onto_net_nlp = (cosine_similarity(onto_nlp_emb) + 1 ) /2#1 - spatial.distance.cosine(onto_nlp_emb, onto_nlp_emb)
				onto_net_mat = np.zeros((ncls, ncls))
				for n1 in onto_net:
					for n2 in onto_net[n1]:
						if n1==n2:
							continue
						onto_net_mat[n1,n2] = onto_net_nlp[n1, n2]
						onto_net_mat[n2,n1] = onto_net_nlp[n2, n1]
			else:
				onto_net_mat = np.zeros((ncls, ncls))
				for n1 in onto_net:
					for n2 in onto_net[n1]:
						if n1==n2:
							continue
						onto_net_mat[n1,n2] = 1
						onto_net_mat[n2,n1] = 1


			rst = 0.7
			onto_net_rwr = RandomWalkRestart(onto_net_mat, rst)
			pred_Y_seen = np.load(seen_output_prefix+'pred_Y_seen.npy')
			pred_Y_seen_norm = pred_Y_seen / pred_Y_seen.sum(axis=1)[:, np.newaxis]
			ratio = (ncls*1./nseen)**2

			pred_Y_all = np.dot(pred_Y_seen_norm, onto_net_rwr[:nseen,:])
			#pred_Y_all[:,:nseen] = pred_Y_seen
			pred_Y_all[:,:nseen] = normalize(pred_Y_all[:,:nseen],norm='l1',axis=1)
			pred_Y_all[:,nseen:] = normalize(pred_Y_all[:,nseen:],norm='l1',axis=1) * ratio

			pname = translate_paramter([nn_nhidden,keep_prob,rst])
			all_output_prefix = folder + pname
			np.save(all_output_prefix+'pred_Y_all.npy',pred_Y_all)
