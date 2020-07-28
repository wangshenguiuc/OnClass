import sys
from time import time
from scipy import stats, sparse
from scipy.sparse.linalg import svds, eigs
from scipy.special import expit
import numpy as np
import os
import pandas as pd
import collections
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/'
sys.path.append(repo_dir)
sys.path.append(repo_dir+'src/task/OnClass/')
sys.path.append(repo_dir+'src/task/OnClass/model/')
os.chdir(repo_dir)
from utils import *
from sentence_transformers import SentenceTransformer
from scipy import spatial


DATA_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass_data/'
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

sentences = list(id2sent.values())
sentence_embeddings = model.encode(sentences)



sent2vec = {}
for sentence, embedding in zip(sentences, sentence_embeddings):
    sent2vec[sentence] = embedding

fout = open(DATA_DIR + 'cell_ontology/cl.ontology.nlp.emb','w')
for id in id2sent:
	fout.write(id)
	vec = sent2vec[id2sent[id]]
	for v in vec:
		fout.write('\t'+str(v))
	fout.write('\n')
fout.close()

dep2sim = {}

for id1 in rev_net:
	for id2 in rev_net[id1]:
		for id3 in rev_net[id1]:
			if id2==id3:
				continue
			sc = 1 - spatial.distance.cosine(sent2vec[id2sent[id2]], sent2vec[id2sent[id3]])
			depth = query_depth_ontology(net, id1) + 1
			if depth not in dep2sim:
				dep2sim[depth] = []
			dep2sim[depth].append(sc)

for depth in dep2sim:
	print (depth, np.mean(dep2sim[depth]), len(dep2sim[depth]))

fout = open(DATA_DIR + 'cell_ontology/cl.ontology.nlp','w')
for id1 in net:
	for id2 in net[id1]:
		sc = 1 - spatial.distance.cosine(sent2vec[id2sent[id1]], sent2vec[id2sent[id2]])
		dep1 = query_depth_ontology(net, id1)
		dep2 = query_depth_ontology(net, id2)
		fout.write(id1+'\t'+id2+'\t'+str(sc)+'\n')
fout.close()
