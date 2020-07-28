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
import string
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn import metrics
from collections import Counter
from scipy.special import softmax
from scipy.stats import norm as dist_model
from sklearn import preprocessing
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.preprocessing import normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from plots import *

if len(sys.argv) <= 2:
	pid = 0
	total_pid = -1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

supple_dir = OUTPUT_DIR + '/SupplementaryTable/CytoScape/'
if not os.path.exists(supple_dir):
    os.makedirs(supple_dir)
output_dir = OUTPUT_DIR + '/MarkerGenes/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
ontology_dir =   OUTPUT_DIR+'/Ontology/CellOntology/'
dnames = ['muris_facs','microcebusBernard','microcebusMartine','muris_droplet','microcebusAntoine','microcebusStumpy']
datasets, genes_list, n_cells = load_names(datanames_26datasets,verbose=False,log1p=True)
onto_ids, keywords, keyword2cname = map_26_datasets_cell2cid(use_detailed=False)
co2name, name2co = get_ontology_name(DATA_DIR=DATA_DIR,lower=False)
l2i = {}
i2l = {}
fin = open(DATA_DIR+'/cell_ontology/cl.ontology')
for line in fin:
	w = line.strip().split('\t') #w[1] is parent of w[0]
	if w[1] not in l2i:
		l2i[w[1]] = len(l2i)
		i2l[l2i[w[1]]] = w[1]
	if w[0] not in l2i:
		l2i[w[0]] = len(l2i)
		i2l[l2i[w[0]]] = w[0]
fin.close()
ncls = len(l2i)
cls2cls = np.zeros((ncls, ncls))
fin = open(DATA_DIR+'/cell_ontology/cl.ontology')
for line in fin:
	w = line.strip().split('\t') #w[1] is parent of w[0]
	cls2cls[l2i[w[1]], l2i[w[0]]] = 1
fin.close()
sp = graph_shortest_path(cls2cls,method='FW',directed =False)

seen_label = []
for dname in dnames:
	unseen_ratio = 0.5
	feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data(dname, DATA_DIR,nsample=50000000)
	seen_label.extend(np.unique(label))

seen_label = np.unique(seen_label)
cutoff = 2
for onto_id in onto_ids:
	fnode = open(supple_dir + co2name[onto_id] + '.node.txt','w')
	fnode.write('node\tcolor\n')
	fedge = open(supple_dir + co2name[onto_id] + '.edge.txt','w')
	fedge.write('source\ttarget\n')
	select_ctps = np.where(sp[l2i[onto_id],:]<=cutoff)[0]
	for n in select_ctps:
		if i2l[n] == onto_id:
			color = 'red'
		elif i2l[n] in seen_label:
			color = 'blue'
		else:
			color = 'grey'
		fnode.write(co2name[i2l[n]]+'\t'+color+'\n')
	for n1 in select_ctps:
		for n2 in select_ctps:
			if cls2cls[n1,n2] == 1:
				fedge.write(co2name[i2l[n1]]+'\t'+co2name[i2l[n2]]+'\n')
fedge.close()
fnode.close()
