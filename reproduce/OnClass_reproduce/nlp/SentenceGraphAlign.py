import sys
from time import time
from scipy import stats, sparse
from scipy.sparse.linalg import svds, eigs
from scipy.special import expit
import numpy as np
import os
import pandas as pd
import collections
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sentence_transformers import SentenceTransformer
from scipy import spatial
from plots import *
from sklearn.metrics.pairwise import cosine_similarity

fig_dir = OUTPUT_DIR + '/Figures/NLP_figures/'

net = collections.defaultdict(dict)
rev_net = collections.defaultdict(dict)
fin = open(DATA_DIR + 'cell_ontology/cl.ontology')
lset = set()
for line in fin:
	s,p = line.lower().strip().split('\t') #p is s' parent
	net[s][p] = 1
	rev_net[p][s] = 1
	lset.add(p)
	lset.add(s)
fin.close()
l2i = {}
i2l = {}

nterm = len(lset)
for i,l in enumerate(lset):
	i2l[i] = l
	l2i[l] = i

mat = np.zeros((nterm, nterm))
for s in net:
	for p in net[s]:
		mat[l2i[s], l2i[p]] = 1
		mat[l2i[p], l2i[s]] = 1

sp = graph_shortest_path(mat,method='FW',directed =False)

fin = open(DATA_DIR + 'cell_ontology/cl.ontology.nlp.emb')
id2vec = {}
for line in fin:
	w = line.strip().split('\t')
	vec = []
	for i in range(1,len(w)):
		vec.append(float(w[i]))
	id2vec[w[0]] = np.array(vec)
fin.close()

id2vec_mat = np.zeros((nterm, len(w[1:])))
for i in range(nterm):
	id2vec_mat[i,:] = id2vec[i2l[i]]

nlp_sim = cosine_similarity(id2vec_mat, id2vec_mat)
print (np.shape(nlp_sim), np.shape(sp))

print (stats.spearmanr(sp.ravel(), nlp_sim.ravel()))
sp = sp.flatten()
dst = nlp_sim.flatten()
dist_up = 5
sp2d = {}
for i,d in enumerate(sp):
	if d==0:
		continue
	if d>dist_up:
		d = dist_up
	if d not in sp2d:
		sp2d[d] = []
	sp2d[d].append(dst[i])
datas = []
jitter_datas = []
xticks = []
for i in range(1,dist_up+1):
	datas.append(sp2d[i])
	jitter_datas.append(np.random.choice(sp2d[i], 100))
	if i!=5:
		xticks.append(str(i)+'\nn='+str(len(sp2d[i])))
	else:
		xticks.append('>4\nn='+str(len(sp2d[i])))

for data in datas:
	print (np.mean(data))
plot_nlp_shortest_distance_boxplot(datas, jitter_datas, fig_file = fig_dir + 'nlp_shortest_distance.pdf', xticks = xticks)
