import sys
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/tools/single_cell/scanorama/bin/'
sys.path.append(repo_dir)
from sklearn.decomposition import PCA
from sklearn import metrics
from process import load_names
from scanorama import *
from time import time
import numpy as np
from collections import Counter
import os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, roc_curve
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
sys.path.append(repo_dir)
sys.path.append(repo_dir+'src/task/SCTypeClassifier/OurClassifier/DeepCCA')
sys.path.append(repo_dir+'src/task/SCTypeClassifier/')
os.chdir(repo_dir)
from utils import *
from NN import BilinearNN
from scanorama_utils import data_names_all

output_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/SingleCell/SCTyping/Integrate_all_datasets/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

co2name, name2co = get_ontology_name()
data_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/tabula_muris_senis/NewAnnotation_20190912/tabula-muris-senis-facs-reannotated-except-for-marrow.h5ad'

test_Y_pred = np.load('0_126_datasets_predict_score.npy')
print (np.shape(test_Y_pred))


datanames, genes_list, labels, datasets, types, month_labels = read_data_and_split_by_months(filename=data_file,nsample=300000)
train_X, train_Y = extract_data(datanames, datasets, labels)
unseen_l, l2i, i2l, train_X2Y, onto_net = create_labels(train_Y, combine_unseen = 0)


nseen = len(l2i) - len(unseen_l)
ratio = len(l2i) * 1. / nseen

test_Y_pred[:,:nseen] = normalize(test_Y_pred[:,:nseen],norm='l1',axis=1)
test_Y_pred[:,nseen:] = normalize(test_Y_pred[:,nseen:],norm='l1',axis=1) * ratio

test_Y_vec = np.argmax(test_Y_pred, axis=1)

datasets, genes_list, n_cells = load_names(data_names_all,verbose=False,log1p=True)
exist_Y = np.unique(train_Y)

onto_ids, keywords, keyword2cname = map_26_datasets_cell2cid(use_detailed=False)
#'CL:0000037','hsc',
#onto_ids = ['CL:0000236','CL:0000235','CL:0002338','CL:0000492','CL:0000815','CL:0000910','CL:0000794','CL:2000001']
#keywords = ['b_cells','infected','cd56_nk','cd4_t_helper','regulatory_t','cytotoxic_t','cd14_monocytes','pbmc']

k_ind = []
k2i = {}
for i,k in enumerate(keywords):
	k2i[k] = len(k_ind)
	k_ind.append(l2i[onto_ids[i]])
k_ind = np.array(k_ind)
print (k_ind)
print (k2i)
st_ind = 0
labels = []
pred = []
for i,dnames_raw in enumerate(data_names_all):
	dnames = dnames_raw.split('/')[-1]
	ed_ind = st_ind + np.shape(datasets[i])[0]
	for keyword in keywords:
		if dnames.startswith(keyword):
			labels.extend(np.ones(ed_ind - st_ind) * k2i[keyword])
			pred.extend(test_Y_pred[st_ind:ed_ind,k_ind])
			break
	st_ind = ed_ind
	#print (len(labels))
labels = np.array(labels)
print (np.unique(labels))
pred = np.array(pred)
pred = np.argmax(pred, axis=1)
print (np.unique(pred))
print (len(labels))
print (pred)
print (labels)
print (metrics.accuracy_score(pred, labels))


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
		#if '10x' not in dnames_raw:
		#	continue
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
