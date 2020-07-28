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
from plots import *
from model.BilinearNNDebug import BilinearNN

if len(sys.argv) <= 2:
	pid = 0
	total_pid = -1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])


only_consider_seen = False
only_vis_unseen = True

co2coname = {}
co2coname['aortic endothelial cell'] ='Aortic endothelial'
co2coname['basal cell'] ='Basal cells'
co2coname['brush cell of epithelium proper of large intestine'] ='Brush cell of epithelium proper of large intestine'
co2coname[ 'cd8-positive, alpha-beta t cell'] ='CD8+ alpha-beta T'
co2coname['ciliated columnar cell of tracheobronchial tree'] ='Ciliated columnar cell of tracheobronchial tree'
co2coname['club cell'] ='Club cells'
co2coname['dn4 thymocyte'] ='DN4 thymocyte'
co2coname['epithelial cell'] ='Epithelial cells'
co2coname['epithelial cell of large intestine'] ='Epithelial cell\nof large intestine'
co2coname['fibroblast'] ='Fibroblast'
co2coname['fibroblast of lung'] ='Fibroblast of lung'
co2coname['glial cell'] ='Glial cells'
co2coname['leukocyte'] ='Leukocyte'
co2coname[ 'kidney collecting duct epithelial cell'] = 'Kidney collecting duct epithelial'
co2coname[ 'lung endothelial cell'] =  'Lung endothelial'
co2coname[ 'mesenchymal stem cell'] =  'Mesenchymal stem cells'
co2coname[ 'monocyte'] ='Monocyte'
co2coname[ 'pancreatic pp cell'] = 'Pancreatic PP cells'
co2coname[ 'proerythroblast'] = 'Proerythroblast'
co2coname[ 'regular ventricular cardiac myocyte'] = 'Regular ventricular\ncardiac myocyte'
co2coname[ 'respiratory basal cell'] = 'Respiratory basal cells'
co2coname[ 'fenestrated cell'] = 'Fenestrated cells'

co2name, name2co = get_ontology_name(DATA_DIR = DATA_DIR, lower = False)

fig_dir = OUTPUT_DIR + '/Figures/Sankey_selected/'

if not os.path.exists(fig_dir):
	os.makedirs(fig_dir)

output_dir = OUTPUT_DIR + '/CompareBaselines/TuneOnclass/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

nn_nhidden = [100,50,25]
keep_prob = 0.7
niter = 5
dnames = ['muris_facs']
for dname in dnames:
	for i in range(niter):
		folder = output_dir +'/'+ dname + '/'+ str(i) + '/'
		if not os.path.exists(folder):
		    os.makedirs(folder)
pct = -1
for dname in dnames:
	if 'allen' in dname:
		ontology_dir =  OUTPUT_DIR+'/Ontology/CellTypeOntology/'
	else:
		ontology_dir =   OUTPUT_DIR+'/Ontology/CellOntology/'
	for unseen_ratio in [0.1,0.2,0.05]:
		pct += 1
		if total_pid>1 and pct%total_pid != pid:
			continue
		folder = output_dir +'/'+ dname + '/3/' + str(unseen_ratio) + '/'
		if not os.path.exists(folder):
			os.makedirs(folder)
		pname = translate_paramter([nn_nhidden,keep_prob])

		seen_output_prefix = folder + pname

		feature, label, genes, ontology_nlp_file, ontology_file = read_singlecell_data(dname, DATA_DIR,nsample=100000000)

		train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = 3)
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

		rst = 0.7
		pname = translate_paramter([nn_nhidden,keep_prob,rst])
		pred_Y_all = np.load(folder+pname+'pred_Y_all.npy')
		unseen_Y_ind = np.array(list(set(test_Y) - set(train_Y)))
		non_unseen_Y_ind = np.array(list(set(range(ncls)) - set(unseen_Y_ind)))
		pred_Y_all[:,non_unseen_Y_ind] = -1*np.inf

		pred_Y = np.argmax(pred_Y_all, axis=1)
		unseen_index = np.where(test_Y>=nseen)[0]
		pred_Y = pred_Y[unseen_index]
		truth_Y = test_Y[unseen_index]
		ntest_only_Y = len(np.unique(truth_Y))
		acc = len(np.where(pred_Y == truth_Y)[0]) / len(unseen_index)
		pred_Y = np.array([co2name[i2l[y]] for y in pred_Y])
		truth_Y = np.array([co2name[i2l[y]] for y in truth_Y])
		#print (np.unique(truth_Y))
		print (dname, iter, unseen_ratio, acc, ntest_only_Y)
		#continue
		fname = translate_paramter([dname,iter,unseen_ratio, acc,ntest_only_Y])
		lab2col, lab2marker = generate_colors(truth_Y, use_man_colors = False)
		plot_sankey_diagram(pred_Y, truth_Y, fig_dir + fname, colorDict=lab2col)
		_, test_X_emb = emb_cells(train_X, test_X, dim = 20)
		print (lab2marker)
		plot_umap(test_X_emb[unseen_index,:], truth_Y, lab2col, file = fig_dir +fname +'_truth_resize.pdf',title='Ground truth\n%d unseen cell types' %  (len(set(test_Y) - set(train_Y))), lab2marker=lab2marker,size=0.1,legend=False,legendmarker=4)
		plot_umap(test_X_emb[unseen_index,:], pred_Y, lab2col, file = fig_dir +fname +'_predicted_resize.pdf',title='OnClass\n%d unseen cell types' %  (len(set(test_Y) - set(train_Y))),lab2marker=lab2marker, size=0.1, legend=False,legendmarker=4)
