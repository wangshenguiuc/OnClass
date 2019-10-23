import sys
import numpy as np
import os
from OnClass.utils import *
from OnClass.OnClassPred import OnClassPred
from OnClass.other_datasets_utils import my_assemble, data_names_all, load_names 

DATA_DIR = '../../../OnClass_data/'
OUTPUT_DIR = DATA_DIR + '/marker_genes/'
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)
	
## read data
data_file = DATA_DIR + '/raw_data/tabula-muris-senis-facs'

if not os.path.isfile(OUTPUT_DIR + 'FACS-predicted_score_matrix.npy'):
	train_X, train_Y_str, genes_list = read_data(filename=data_file, DATA_DIR=DATA_DIR, return_genes=True)
	tms_genes_list = [x.upper() for x in list(genes_list.values())[0]]
	ntrain,ngene = np.shape(train_X)
	## embedd the cell ontology
	unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, co_dim = 200, co_mi = 0, DATA_DIR=DATA_DIR)
	train_Y = MapLabel2CL(train_Y_str, l2i)

	## train and predict
	OnClass_obj = OnClassPred()
	OnClass_obj.train(train_X, train_Y, Y_emb, max_iter=20, nhidden=[100])
	test_Y_pred = OnClass_obj.predict(train_X)

	np.save(OUTPUT_DIR + 'FACS-predicted_score_matrix.npy', test_Y_pred)

fig_dir = '../../OnClass_data/figures/marker_gene/'
if not os.path.exists(fig_dir):
	os.makedirs(fig_dir)
	
## read data
train_X, train_Y_str, genes_list = read_data(filename=data_file, DATA_DIR=DATA_DIR, return_genes=True)
train_X = np.log1p(train_X.todense()+1)
tms_genes_list = [x.upper() for x in list(genes_list.values())[0]]
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, DATA_DIR=DATA_DIR)
train_Y = MapLabel2CL(train_Y_str, l2i)
genes = genes_list[datanames[0]]
g2i = {}
i2g = {}
for i,g in enumerate(genes):
	g2i[g.lower()] = i
	i2g[i] = g
ngene = len(genes)
test_Y_pred = np.load(OUTPUT_DIR + 'FACS-predicted_score_matrix.npy')

## Differential expression analysis
ncell = np.shape(test_Y_pred)[0]
co2name, name2co = get_ontology_name(DATA_DIR=DATA_DIR)
tp2genes = read_type2genes(g2i, DATA_DIR=DATA_DIR)
thres = np.array(range(1,1000))
topk = 50
in_tms_ranks = []
not_tms_ranks = []
n_in_tms =0
for tp in tp2genes:
	ci = l2i[tp]
	k_bot_cells = np.argsort(test_Y_pred[:,ci])[:topk]
	k_top_cells = np.argsort(test_Y_pred[:,ci])[ncell-topk:]
	pv = scipy.stats.ttest_ind(train_X[k_top_cells,:], train_X[k_bot_cells,:], axis=0)[1]
	top_mean = np.mean(train_X[k_top_cells,:],axis=0)
	bot_mean = np.mean(train_X[k_bot_cells,:],axis=0)
	for g in range(ngene):
		if top_mean[0,g] < bot_mean[0,g]:
			pv[g] = 1.
	pv_sort = list(np.argsort(pv))
	min_rank = 1000000
	for g in tp2genes[tp]:
		gid = g2i[g.lower()]
		rank = pv_sort.index(gid)
		min_rank = min(rank, min_rank)
	if ci in np.unique(train_Y):
		in_tms_ranks.append(min_rank)
	else:
		not_tms_ranks.append(min_rank)

not_tms_ranks = np.array(not_tms_ranks)
in_tms_ranks = np.array(in_tms_ranks)


in_tms_y = []
not_tms_y = []
for t in thres:
	in_tms_y.append( len(np.where(in_tms_ranks <= t)[0]) / len(in_tms_ranks))
	not_tms_y.append( len(np.where(not_tms_ranks <= t)[0]) / len(not_tms_ranks))


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(thres, in_tms_y, 'g', label='Seen cell types (n=%d)'%len(in_tms_ranks), linewidth=2)
plt.plot(thres, not_tms_y, 'y', label='Unseen cell types (n=%d)'%len(not_tms_ranks), linewidth=2)
plt.legend(loc="lower right",frameon=False)

plt.ylabel('Percentage of cell types')
plt.xlabel('Rank of marker genes')
#ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
plt.tight_layout()
plt.savefig(fig_dir+'mark_genes.pdf')

## Write marker genes to file
fmarker_genes = open(OUTPUT_DIR+'marker_genes.txt','w')
for ci in range(nlabel):
	tp = i2l[ci]
	k_bot_cells = np.argsort(test_Y_pred[:,ci])[:topk]
	k_top_cells = np.argsort(test_Y_pred[:,ci])[ncell-topk:]
	pv = scipy.stats.ttest_ind(train_X[k_top_cells,:], train_X[k_bot_cells,:], axis=0)[1]
	top_mean = np.mean(train_X[k_top_cells,:],axis=0)
	bot_mean = np.mean(train_X[k_bot_cells,:],axis=0)
	for g in range(ngene):
		if top_mean[0,g] < bot_mean[0,g]:
			pv[g] = 1.
	pv_sort = list(np.argsort(pv))
	min_rank = 1000000
	fmarker_genes.write(tp+'\t')
	for i in range(100):
		fmarker_genes.write(i2g[pv_sort[i]]+'\t')
	fmarker_genes.write('\n')
fmarker_genes.close()