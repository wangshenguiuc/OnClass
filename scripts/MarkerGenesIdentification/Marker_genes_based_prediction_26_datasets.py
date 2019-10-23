from scipy import sparse
from scanorama import *
from time import time
import numpy as np
from collections import Counter
import os
from sklearn.metrics import roc_auc_score, roc_curve
from time import time
import numpy as np
from collections import Counter
import os
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
	


datasets, genes_list, n_cells = load_names(data_names_all,verbose=False,log1p=True)
datasets, genes = merge_datasets(datasets, genes_list)
train_X = sparse.vstack(datasets)
train_X = train_X.toarray()
g2i = {}
for i,g in enumerate(genes):
	g2i[g.lower()] = i
#'CL:0000037','hsc',
onto_ids = ['CL:0000236','CL:0000235','CL:0000037','CL:0002338','CL:0000492','CL:0000815','CL:0000910','CL:0000794','CL:2000001']
keywords = ['b_cells','infected','hsc','cd56_nk','cd4_t_helper','regulatory_t','cytotoxic_t','cd14_monocytes','pbmc']


tp2genes_base = read_type2genes(g2i)

tp2genes_our = {}
fin = open(OUTPUT_DIR+'marker_genes.txt')
for line in fin:
	w = line.strip().split('\t')
	tp2genes_our[w[0]] = w[1:]
fin.close()

our_auc={}
base_auc={}
topk  = 100

for i in range(len(onto_ids)):
	labels = []
	st_ind = 0
	for ii,dnames_raw in enumerate(data_names_all):
		dnames = dnames_raw.split('/')[-1]
		ed_ind = st_ind + np.shape(datasets[ii])[0]
		if dnames.startswith(keywords[i]):
			labels.extend(np.ones(ed_ind - st_ind))
		else:
			labels.extend(np.zeros(ed_ind - st_ind))
		st_ind = ed_ind
	labels = np.array(labels)
	marker = tp2genes_our[onto_ids[i]][:topk]
	marker = [g2i[g.lower()] for g in marker if g.lower() in g2i]
	marker = np.array(marker)
	if len(marker)==0:
		continue
	pred = np.sum(train_X[:,marker], axis=1)
	our_auc[i] = roc_auc_score(labels, pred)
	if False and onto_ids[i] in tp2genes_base:
		marker = tp2genes_base[onto_ids[i]]
		marker = [g2i[g.lower()] for g in marker if g.lower() in g2i]
		marker = np.array(marker)
		if len(marker)==0:
			continue
		pred = np.sum(train_X[:,marker], axis=1)
		base_auc[i] = roc_auc_score(labels, pred)
	print ('%s %f'%(keywords[i],our_auc[i]))
	plt.clf()

	mpl.rcParams['pdf.fonttype'] = 42
	MEDIUM_SIZE = 30
	BIGGER_SIZE = 50

	plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

	fpr,tpr = roc_curve(labels, pred)[:2]
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
			 lw=lw, label='ROC (area = %0.2f)'%our_auc[i])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate',fontsize=30)
	plt.ylabel('True Positive Rate',fontsize=30)

	plt.title(keywords[i] )
	plt.legend(loc="lower right",fontsize=30,frameon=False)
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR+keywords[i]+'_auroc.pdf')#

our = []
base = []
for k in base_auc:
	our.append(our_auc[k])
	base.append(base_auc[k])
print ('%d %d %d %f %f %f'%(topk, len(our), len(list(our_auc.values())), np.mean(list(our_auc.values())), np.mean(list(base_auc.values())),np.mean(our)))
mean_auc = []
mean_rank = []
ctype = []
for k in our_auc:
	ctype.append(keywords[k])
	mean_auc.append(our_auc[k])
	if keywords[k] == 'b_cells' or keywords[k] == 'infected':
		mean_rank.append(our_auc[k]+100)
	else:
		mean_rank.append(our_auc[k])
mean_auc = np.array(mean_auc)
ctype = np.array(ctype)
mean_rank = np.array(mean_rank)

print (mean_auc)
print (ctype)
print (mean_rank)

ind = np.argsort(mean_rank*-1)
mean_auc = mean_auc[ind]
ctype = ctype[ind]

print (mean_auc)
print (ctype)
print (mean_rank)

mpl.rcParams['pdf.fonttype'] = 42
MEDIUM_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title



plt.clf()
plt.figure()
y_pos = np.arange(len(ctype))
y_pos[2:] = y_pos[2:] + 1
width = 0.9
#plt.plot(xs, horiz_line_data, 'r--')
bars = plt.bar(y_pos, mean_auc, align='edge', alpha=1, width=width, color='y')
bars[0].set_color('g')
bars[1].set_color('g')
plt.plot([], c='g', label='Seen cell types')
plt.plot([], c='y', label='Unseen cell types')
plt.xticks(y_pos+width/2, ctype, rotation=90)
plt.ylabel('AUROC')
plt.legend(frameon=False)
plt.ylim([0.5, 1.0])
plt.tight_layout()
plt.savefig(OUTPUT_DIR+'predict_26_bar.pdf')
