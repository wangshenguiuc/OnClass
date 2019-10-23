import sys
from scanorama import *
import numpy as np
import os
from scipy import sparse
from utils import *
import OnClassPred
from other_datasets_utils import my_assemble, data_names_all, load_names 
from sklearn.metrics import roc_auc_score, roc_curve

input_dir = '../../OnClass_data/26-datasets/'
output_dir = '../../OnClass_data/figures/26-datasets/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

#read data
co2name, name2co = get_ontology_name()
data_file = '../../OnClass_data/raw_data/tabula-muris-senis-facs'
train_X, train_Y_str, genes_list = read_data(filename=data_file, return_genes=True)
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str)
train_Y = MapLabel2CL(train_Y_str, l2i)
datasets, genes_list, n_cells = load_names(data_names_all,verbose=False,log1p=True)
exist_Y = np.unique(train_Y)

#read pre-computed prediction score. Please run Annot26datasets.py first to generate this score matrix.
test_Y_pred = np.load(input_dir + '26_datasets_predicted_score_matrix.npy')
test_Y_vec = np.argmax(test_Y_pred, axis=1)

# calculate AUROC for each cell type
onto_ids, keywords, keyword2cname = map_26_datasets_cell2cid(use_detailed=False)

ctype = []
mean_auc = []
for onto_id, keyword in zip(onto_ids,keywords):
	ndata = len(data_names_all)
	nlabel = np.shape(test_Y_pred)[1]
	st_ind = 0
	ed_ind = 0
	labels = []
	pred = []
	for i,dnames_raw in enumerate(data_names_all):
		dnames = dnames_raw.split('/')[-1]
		ed_ind = st_ind + np.shape(datasets[i])[0]
		if dnames.startswith(keyword):
			labels.extend(np.ones(ed_ind - st_ind))
		else:
			labels.extend(np.zeros(ed_ind - st_ind))
		pred.extend(test_Y_pred[st_ind:ed_ind,l2i[onto_id]])
		st_ind = ed_ind
	labels = np.array(labels)
	if np.sum(labels)==0:
		continue
	auc = roc_auc_score(labels, pred)
	if auc>0.5:
		mean_auc.append(auc)
		ctype.append(keyword2cname[keyword])
	known_y = onto_id in exist_Y
	print ('%s %f %d %d'%(keyword,auc,np.sum(labels),known_y))

	fpr,tpr = roc_curve(labels, pred)[:2]


	mpl.rcParams['pdf.fonttype'] = 42
	MEDIUM_SIZE = 30
	BIGGER_SIZE = 50

	plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	#plt.rcParams['figure.labelweight'] = 'bold'

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
			 lw=lw, label='AUROC = %0.2f'%auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])

	plt.xlabel('False Positive Rate',fontsize=40)
	plt.ylabel('True Positive Rate',fontsize=40)
	if known_y:
		suffix = 'in TMS'
	else:
		suffix = 'not in TMS'

	plt.title(keyword2cname[keyword],fontsize=40)
	plt.legend(loc="lower right",fontsize=30,frameon=False)
	plt.tight_layout()
	plt.savefig(output_dir+keyword+'_auroc.pdf')#

# plot bar plot
mean_auc = np.array(mean_auc)
ctype = np.array(ctype)
ind = np.argsort(mean_auc*-1)
mean_auc = mean_auc[ind]
ctype = ctype[ind]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
y_pos = np.arange(len(ctype))
y_pos[2:] = y_pos[2:] + 1
width = 0.9
xs = np.linspace(0,np.max(y_pos),200)
horiz_line_data = np.array([0.5 for i in range(len(xs))])
#plt.plot(xs, horiz_line_data, 'r--')
bars = plt.bar(y_pos, mean_auc, align='edge', alpha=1, width=width, color='y')
bars[0].set_color('g')
bars[1].set_color('g')
plt.plot([], c='g', label='Cell types in TMS')
plt.plot([], c='y', label='Cell types not in TMS')
plt.legend(frameon=False)
plt.xticks(y_pos+width/2, ctype, rotation=90, fontsize=30)
plt.ylabel('AUROC',fontsize=40)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylim([0.5, 1.05])
plt.yticks([0.5, 0.6,0.7,0.8,0.9,1.])
plt.tight_layout()
plt.savefig(output_dir+'bar.pdf')
