import sys
from scanorama import *
import numpy as np
import os
from scipy import sparse
from utils import *
import OnClassPred
from other_datasets_utils import my_assemble, data_names_all, load_names 
from sklearn.metrics import silhouette_samples, roc_curve

output_dir = '../../OnClass_data/figures/26-datasets/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

input_dir =  '../../OnClass_data/26-datasets/'
test_Y_pred = np.load(input_dir + '26_datasets_predicted_score_matrix.npy')
datasets, genes_list, n_cells = load_names(data_names_all,verbose=False,log1p=True)
datasets, genes = merge_datasets(datasets, genes_list)


ndata = len(data_names_all)
nlabel = np.shape(test_Y_pred)[1]
st_ind = 0
ed_ind = 0
labels = []
for i,dnames_raw in enumerate(data_names_all):
	dnames_raw_v = dnames_raw.replace('../../OnClass_data/','').split('/')
	data_ind = dnames_raw_v.index('26-datasets')
	dnames = dnames_raw_v[data_ind+1]
	ed_ind = st_ind + np.shape(datasets[i])[0]
	for ii in range(ed_ind-st_ind):
		labels.append(dnames)
	st_end = ed_ind
labels = np.array(labels)

datasets_dimred, genes = process_data(datasets, genes, dimred=100)
datasets_dimred = assemble(datasets_dimred, ds_names=data_names_all, sigma=150, verbose=False)
datasets_dimred = np.vstack(datasets_dimred)

data_file = '../../OnClass_data/raw_data/tabula-muris-senis-facs'
X, Y = read_data(filename=data_file)
train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(X, Y)
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str)
nseen = len(l2i) - len(unseen_l)

pca = PCA(n_components=50)
test_Y_pred_red = pca.fit_transform(test_Y_pred[:, :nseen])

sil_sc = silhouette_samples(test_Y_pred_red, labels, metric='cosine')
orig_sil_sc = silhouette_samples(datasets_dimred, labels, metric='cosine')
print ('%f %f'%(np.mean(sil_sc),np.mean(orig_sil_sc)))
pvalue = scipy.stats.ttest_rel(sil_sc, orig_sil_sc)[1]
print (pvalue)


lab2col, lab2marker = generate_colors(labels)
index = np.random.choice(len(labels),5000)
plot_umap(test_Y_pred[index, :nseen], labels[index], lab2col, file = output_dir + 'our_umap.pdf', lab2marker=lab2marker)
plot_umap(datasets_dimred[index,:], labels[index], lab2col, file = output_dir + 'scan_umap.pdf', lab2marker=lab2marker)
all_data = [sil_sc, orig_sil_sc]
fig = plt.figure(1, figsize=(5, 5))

bpl = plt.boxplot(sil_sc, positions=[1], sym='', widths=0.6)
bpr = plt.boxplot(orig_sil_sc, positions=[2], sym='', widths=0.6)

def set_box_color(bp, color):
	plt.setp(bp['boxes'], color=color)
	plt.setp(bp['whiskers'], color=color)
	plt.setp(bp['caps'], color=color)
	plt.setp(bp['medians'], color=color)

set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='OnClass')
plt.plot([], c='#2C7BB6', label='Scanorama')
plt.xlim([0,3])
plt.ylabel('Silhouette coefficient')
plt.xticks([1,2],['OnClass','Scanorama'])
#plt.xlim(-2, len(ticks)*2)

plt.ylim([-0.4, 1])
plt.yticks([-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

plt.tight_layout()
fig.savefig(output_dir + 'boxplot.pdf', bbox_inches='tight')
