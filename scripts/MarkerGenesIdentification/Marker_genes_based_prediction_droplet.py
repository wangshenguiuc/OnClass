from sklearn import metrics
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


def ct2bin(n):
	if n<200:
		return 0
	if n<500:
		return 1
	if n<2000:
		return 2
	return 3

input_dir = '../../OnClass_data/marker_genes/'
fig_dir = '../../OnClass_data/figures/marker_genes/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

data_file = '../../OnClass_data/raw_data/tabula-muris-senis-droplets'
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


unseen_l, l2i, i2l, train_X2Y, onto_net = create_labels(train_Y, combine_unseen = False)
train_X = np.log1p(train_X.todense()+1)
genes = genes_list[datanames[0]]
g2i = {}
i2g = {}
for i,g in enumerate(genes):
	g2i[g.lower()] = i
	i2g[i] = g
ngene = len(genes)
train_Y = [l2i[y] for y in train_Y]
unseen_l = [l2i[y] for y in unseen_l]
train_Y = np.array(train_Y)
nunseen = len(unseen_l)
nlabel = np.shape(train_X2Y)[1]
nseen = nlabel - nunseen

co2name, name2co = get_ontology_name()
tp2genes_base = read_type2genes(g2i)

tp2genes_our = {}
fin = open(input_dir+'marker_genes.txt')
for line in fin:
	w = line.strip().split('\t')
	tp2genes_our[w[0]] = w[1:]
fin.close()



topks = [10]
width = 0.9

plt.clf()
plt.figure()

ctype = []
mean_auc = []
our_auc ={}
base_auc = {}
topk = 10
ctype.append(str(topk))
for i in range(nseen):
	#
	marker = tp2genes_our[i2l[i]][:topk]
	marker = [g2i[g.lower()] for g in marker]
	marker = np.array(marker)
	pred = np.sum(train_X[:,marker], axis=1)
	our_auc[i] = roc_auc_score(train_X2Y[:,i], pred)
	if i2l[i] not in tp2genes_base:
		continue
	marker = tp2genes_base[i2l[i]]
	marker = [g2i[g] for g in marker]
	marker = np.array(marker)
	pred = np.sum(train_X[:,marker], axis=1)
	base_auc[i] = roc_auc_score(train_X2Y[:,i], pred)
	#print ('%f %f %d'%(our_auc, base_auc, l2count[i2l[i]]))
	#print ('%d'%(l2count[i2l[i]]))

our_bar = []
base_bar = []
for i in range(4):
	bases = []
	ours = []
	for j in range(nseen):
		#print (ct2bin(l2count[i2l[j]]))
		if ct2bin(l2count[i2l[j]])!=i:
			continue
		if j in base_auc:
			bases.append(base_auc[j])
		ours.append(our_auc[j])
	print (np.mean(bases),np.mean(ours))
	our_bar.append(ours)
	base_bar.append(bases)


ticks = ['<200', '200-500', '501-2000','>2000']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.rc('xtick', labelsize=20)

plt.figure()
#bpl = plt.violinplot(our_bar, positions=np.array(range(len(our_bar)))*2.0-0.4,  widths=0.6)
bpl = plt.boxplot(our_bar, positions=np.array(range(len(our_bar)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(base_bar, positions=np.array(range(len(base_bar)))*2.0+0.4, sym='',widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='SConType')
plt.plot([], c='#2C7BB6', label='Expert curation')
plt.legend(frameon=False)
plt.ylabel('AUROC')
plt.xlabel('Number of annotations')
plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.title('Cell types with expert curation')
plt.xlim(-2, len(ticks)*2)
plt.ylim(0.2, 1)
plt.tight_layout()
plt.savefig(fig_dir+'predict_tms_bar_only_annotations.pdf')

our_bar = []
for i in range(4):
	bases = []
	ours = []
	for j in range(nseen):
		#print (ct2bin(l2count[i2l[j]]))
		if ct2bin(l2count[i2l[j]])!=i:
			continue
		if j in base_auc:
			continue
		ours.append(our_auc[j])
	our_bar.append(ours)


ticks = ['<200', '200-500', '501-2000','>2000']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

bpl = plt.boxplot(our_bar, positions=np.array(range(len(our_bar)))-0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='SConType')
plt.legend(frameon=False)
plt.ylabel('AUROC')
plt.xlabel('Number of annotations')
plt.title('Cell types without expert curation')
plt.xticks([-0.5,0.5,1.5,2.5], ticks)
plt.xlim(-1, len(ticks)-1)
plt.ylim(0.3, 1)
plt.tight_layout()
plt.savefig(fig_dir+'predict_tms_bar_all.pdf')
