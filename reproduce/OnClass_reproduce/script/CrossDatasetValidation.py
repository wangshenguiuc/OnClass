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
from model.BilinearNNDebug import BilinearNN

if len(sys.argv) <= 2:
	pid = 0
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])


ontology_dir =   OUTPUT_DIR+'/Ontology/CellOntology/'
output_dir = OUTPUT_DIR + '/CompareBaselines/TuneOnclass/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

#read TMS
nn_nhidden = [100,50,25]
keep_prob = 0.7
dname = 'muris_facs'
unseen_ratio = 0.5
dnames = ['muris_facs','microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy','muris_droplet']

pct = -1
for dname1 in dnames:
	for dname2 in dnames:
		if dname1 == dname2:
			continue
		pct += 1
		if total_pid>1 and pct%total_pid != pid:
			continue
		print (dname2, pct)

		feature1, label1, genes1, ontology_nlp_file, ontology_file = read_singlecell_data(dname1, DATA_DIR,nsample=50000000)
		co_dim = 5
		ontology_emb_file = ontology_dir + str(co_dim)
		unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(label1, co_dim = co_dim, use_pretrain = ontology_emb_file, ontology_nlp_file = ontology_nlp_file, ontology_file = ontology_file)
		ncls = np.shape(cls2cls)[0]
		label1 = MapLabel2CL(label1, l2i)
		unseen_l = MapLabel2CL(unseen_l, l2i)
		nunseen = len(unseen_l)
		nseen = ncls - nunseen
		onto_net_rwr = RandomWalkOntology(onto_net, l2i, rst=0.7, ontology_nlp_file = ontology_nlp_file, ontology_nlp_emb_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/cell_ontology/cl.ontology.nlp.emb')

		feature2, label2, genes2, ontology_nlp_file, ontology_file = read_singlecell_data(dname2, DATA_DIR,nsample=50000000)
		label2 = MapLabel2CL(label2, l2i)

		common_genes = np.array(list(set(genes1) & set(genes2)))
		gid1 = find_gene_ind(genes1, common_genes)
		gid2 = find_gene_ind(genes2, common_genes)

		train_X = feature1[:, gid1]
		test_X = feature2[:, gid2]
		train_Y = label1
		test_Y = label2
		#print (test_Y)
		#print (train_Y)
		#print (np.shape(feature1), np.shape(test_X))
		print ('ngenes:%d. seen: %d, ntrainY: %d, ntestY: %d, unseen: %d %d' % (len(common_genes), nseen, len(np.unique(train_Y)), len(np.unique(test_Y)), len(set(test_Y) - set(train_Y)), len(set(label2) - set(label1))))

		#train_X, test_X = process_expression([train_X, test_X])
		train_X = np.log1p(train_X)
		test_X = np.log1p(test_X)
		BN = BilinearNN()
		BN.train(train_X, train_Y, Y_emb, nseen, use_y_emb = False, test_X = test_X, test_Y = test_Y,
		nhidden = nn_nhidden, max_iter = 50, minibatch_size = 128, lr = 0.0001, l2=0.005, keep_prob = 1.0, early_stopping_step = 5)

		#predict on 26
		pred_Y_seen = BN.predict(test_X)
		np.save(output_dir+dname1+'.'+dname2 + 'pred_Y_seen.npy',pred_Y_seen)
		test_Y_ind = np.sort(np.array(list(set(test_Y) |  set(train_Y))))


		pred_Y_seen_norm = pred_Y_seen / pred_Y_seen.sum(axis=1)[:, np.newaxis]
		ratio = (ncls*1./nseen)**2
		pred_Y_all = np.dot(pred_Y_seen_norm, onto_net_rwr[:nseen,:])
		pred_Y_all[:,:nseen] = normalize(pred_Y_all[:,:nseen],norm='l1',axis=1)
		pred_Y_all[:,nseen:] = normalize(pred_Y_all[:,nseen:],norm='l1',axis=1) * ratio
		np.save(output_dir+dname1+'.'+dname2+'pred_Y_all.npy',pred_Y_all)
		res = evaluate(pred_Y_all, test_Y, unseen_l, nseen, Y_ind = test_Y_ind, Y_net = onto_net, write_screen = True, prefix = dname1+'.'+dname2)
