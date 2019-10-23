import sys
from scanorama import *
import numpy as np
import os
from scipy import sparse
from sklearn.metrics import silhouette_samples, roc_curve
from OnClass.utils import *
from OnClass.OnClassPred import OnClassPred
from OnClass.other_datasets_utils import my_assemble, data_names_all, load_names 


DATA_DIR = '../../../OnClass_data/'
FIG_DIR = DATA_DIR + '/figures/26-datasets/'
if not os.path.exists(FIG_DIR):
	os.makedirs(FIG_DIR)
INPUT_DIR = DATA_DIR + '/26-datasets/'

## read TMS and 26-datasets data
test_Y_pred = np.load(INPUT_DIR + '26_datasets_predicted_score_matrix.npy')
datasets, genes_list, n_cells = load_names(data_names_all,verbose=False,log1p=True, DATA_DIR=DATA_DIR)
datasets, genes = merge_datasets(datasets, genes_list)

## integration based on our method
data_file = DATA_DIR + '/raw_data/tabula-muris-senis-facs'
X, Y = read_data(filename=data_file, DATA_DIR=DATA_DIR)
train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(X, Y)
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, DATA_DIR=DATA_DIR)
nseen = len(l2i) - len(unseen_l)
pca = PCA(n_components=50)
test_Y_pred_red = pca.fit_transform(test_Y_pred[:, :nseen])

## integration based on Scanorama
datasets_dimred, genes = process_data(datasets, genes, dimred=100)
datasets_dimred = assemble(datasets_dimred, ds_names=data_names_all, sigma=150, verbose=False)
datasets_dimred = np.vstack(datasets_dimred)

## read labels used for plotting umap
ndata = len(data_names_all)
nlabel = np.shape(test_Y_pred)[1]
st_ind = 0
ed_ind = 0
labels = []
for i,dnames_raw in enumerate(data_names_all):
	dnames_raw_v = dnames_raw.split('/')
	data_ind = dnames_raw_v.index('26-datasets')
	dnames = dnames_raw_v[data_ind+1]
	ed_ind = st_ind + np.shape(datasets[i])[0]
	for ii in range(ed_ind-st_ind):
		labels.append(dnames)
	st_end = ed_ind
labels = np.array(labels)


## plot UMAP and calculate silhouette scores
sil_sc = silhouette_samples(test_Y_pred_red, labels, metric='cosine')
orig_sil_sc = silhouette_samples(datasets_dimred, labels, metric='cosine')
print ('%f %f'%(np.mean(sil_sc),np.mean(orig_sil_sc)))
pvalue = scipy.stats.ttest_rel(sil_sc, orig_sil_sc)[1]
print (pvalue)

lab2col, lab2marker = generate_colors(labels)
index = np.random.choice(len(labels),5000)
plot_umap(test_Y_pred[index, :nseen], labels[index], lab2col, file = FIG_DIR + 'our_umap.pdf', lab2marker=lab2marker)
plot_umap(datasets_dimred[index,:], labels[index], lab2col, file = FIG_DIR + 'scan_umap.pdf', lab2marker=lab2marker)
all_data = [sil_sc, orig_sil_sc]
fig = plt.figure(1, figsize=(5, 5))

bpl = plt.boxplot(sil_sc, positions=[1], sym='', widths=0.6)
bpr = plt.boxplot(orig_sil_sc, positions=[2], sym='', widths=0.6)

def set_box_color(bp, color):
	plt.setp(bp['boxes'], color=color)
	plt.setp(bp['whiskers'], color=color)
	plt.setp(bp['caps'], color=color)
	plt.setp(bp['medians'], color=color)

set_box_color(bpl, '#D7191C')
set_box_color(bpr, '#2C7BB6')
plt.plot([], c='#D7191C', label='OnClass')
plt.plot([], c='#2C7BB6', label='Scanorama')
plt.xlim([0,3])
plt.ylabel('Silhouette coefficient')
plt.xticks([1,2],['OnClass','Scanorama'])
plt.ylim([-0.4, 1])
plt.yticks([-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
plt.tight_layout()
fig.savefig(FIG_DIR + 'boxplot.pdf', bbox_inches='tight')
