import sys
from time import time
from scipy import stats, sparse
from scipy.sparse.linalg import svds, eigs
from scipy.special import expit
import numpy as np
import os
import seaborn as sns
import pandas as pd
import collections
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)
from sentence_transformers import SentenceTransformer
from scipy import spatial
from plots import *

output_dir = OUTPUT_DIR + '/NLP/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fig_dir = OUTPUT_DIR + '/Figures/NLP_figures/'

if not os.path.exists(fig_dir):
	os.makedirs(fig_dir)

net = collections.defaultdict(dict)
rev_net = collections.defaultdict(dict)
fin = open(DATA_DIR + 'cell_ontology/cl.ontology')
for line in fin:
	s,p = line.lower().strip().split('\t') #p is s' parent
	net[s][p] = 1
	rev_net[p][s] = 1
fin.close()

model = SentenceTransformer('bert-base-nli-mean-tokens')

id2sent = {}
fin = open(DATA_DIR + 'cell_ontology/cl.obo')
for line in fin:
	if line.lower().startswith('def: '):
		sent = line.lower().strip().split('"')[1]
		id2sent[id] = sent
	if line.lower().startswith('name: '):
		name = line.lower().strip().split('name: ')[1]
		id2sent[id] = name
	if line.lower().startswith('id: '):
		id = line.lower().strip().split('id: ')[1]

if not os.path.isfile(output_dir + 'dep2sim.npy'):
	sentences = list(id2sent.values())
	sentence_embeddings = model.encode(sentences)

	sent2vec = {}
	for sentence, embedding in zip(sentences, sentence_embeddings):
	    sent2vec[sentence] = embedding

	dep2sim = {}

	for id1 in rev_net:
		for id2 in rev_net[id1]:
			for id3 in rev_net[id1]:
				if id2<=id3:
					continue
				sc = 1 - spatial.distance.cosine(sent2vec[id2sent[id2]], sent2vec[id2sent[id3]])
				depth = query_depth_ontology(net, id1) + 1
				if depth not in dep2sim:
					dep2sim[depth] = []
				dep2sim[depth].append(sc)
	np.save(output_dir + 'dep2sim.npy', dep2sim)
else:
	dep2sim = np.load(output_dir + 'dep2sim.npy', allow_pickle = True).item()

depths = list(dep2sim.keys())
max_depths = max(depths)
for depth in dep2sim:
	print (depth, np.mean(dep2sim[depth]), len(dep2sim[depth]))

datas = []
jitter_datas = []
xlabels = []
for i in range(2,12):
	datas.append(dep2sim[i])
	jitter_datas.append(np.random.choice(dep2sim[i], min(200, len(dep2sim[i])), replace=False))
	xlabels.append(str(i)+'\nn='+str(len(dep2sim[i]))+'')

plot_nlp_text_sibling_similarity(jitter_datas, datas, fig_dir, xlabels)
