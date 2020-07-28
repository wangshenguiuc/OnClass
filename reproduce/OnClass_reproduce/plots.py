import sys
from time import time
from scipy import stats, sparse
from scipy.sparse.linalg import svds, eigs
from scipy.special import expit
import numpy as np
import os
import math
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn import metrics
import collections
from scipy.stats import norm as dist_model
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA
import pandas as pd
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap
from collections import defaultdict
import seaborn as sns

from utils import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

#Lemur 1 = Bernard (male); Lemur 2 = Stumpy (female); Lemur 3 = Martine (female); Lemur 4 = Antoine (male)
MEDIUM_SIZE = 8
SMALLER_SIZE = 6
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)	 # fontsize of the axes title
plt.rc('xtick', labelsize=SMALLER_SIZE)	# fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)	# fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
FIG_HEIGHT = 2
FIG_WIDTH = 2
dname2keyword = {'muris_facs':'Muris FACS','muris_droplet':'Muris droplet','allen':'Allen',
'microcebusAntoine':'Lemur 4','microcebusBernard':'Lemur 1','microcebusMartine':'Lemur 3','microcebusStumpy':'Lemur 2'}
for dname in dname2keyword:
	st = dname2keyword[dname]
	dname2keyword[dname] = ''
	for i in range(len('Muris droplet') - len(st)):
		dname2keyword[dname] += ' '
	dname2keyword[dname] += st
dnames = ['muris_facs','muris_droplet','microcebusBernard','microcebusStumpy','microcebusMartine','microcebusAntoine']
def get_man_colors():
	import matplotlib.colors as pltcolors

	cmap = [plt.cm.get_cmap("tab20b")(0)] # Aorta
	for i in range(3,5): # BAT, Bladder
		cmap.append(plt.cm.get_cmap("tab20b")(i))
	for i in range(6,9): # Brain_Myeloid, Brain_Non_Myeloid, Diaphgram
		cmap.append(plt.cm.get_cmap("tab20b")(i))
	for i in range(9,13): # GAT, Heart, Kidney, Large_Intestine
		cmap.append(plt.cm.get_cmap("tab20b")(i))
	for i in range(14,20): # Limb_Muscle, Liver, Lung, MAT, Mammary_Gland, Marrow
		cmap.append(plt.cm.get_cmap("tab20b")(i))
	for i in range(0,20): # Pancreas, SCAT
		cmap.append(plt.cm.get_cmap("tab20c")(i))
	manual_colors = []
	for c in cmap:
		manual_colors.append(pltcolors.to_hex(c))
	return manual_colors

def plot_nlp_text_sibling_similarity(jitter_datas, datas, fig_dir, xlabels):
	plt.clf()
	fig, ax = plt.subplots(figsize=(5,FIG_HEIGHT))
	ax = sns.stripplot(data=jitter_datas, jitter=True, size=1)

	ax = sns.violinplot( data=datas,
	                    inner='box', color=".8",showmeans=True,
	                         showextrema=True)
	ax.set_ylabel('Text-based\ncell type similarity')
	ax.set_xlabel('Depth in the Cell Ontology')
	#ax.violinplot(datas)
	ax.set_xticklabels(xlabels)
	#ax.set_yticklabels(np.arange(0, 1.0+0.001, 0.2))
	#ax.set_ylim([0., 1.0])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.tight_layout()
	plt.savefig(fig_dir+'nlp_emb_violin.pdf')


def plot_expression_embedding_spearman(d, e, tis,fig_file):
	tis = tis.replace('_',' ')


	plt.clf()
	fig, ax = plt.subplots(figsize=(FIG_WIDTH,FIG_HEIGHT))# fontsize of the figure title
	n = len(d)
	ax.set_ylabel('Ontology-based similarity')
	ax.set_xlabel('Gene expression similarity')
	pear,pv = stats.pearsonr(d,e)
	ax.set_title(tis.capitalize())#+'\n(r = %.2f, p = %.2e)' % (pear, pv))
	plt.scatter(d, e, s=5, c="black", marker='s')

	xmin,xmax = min(d),max(d)
	ymin,ymax = min(e),max(e)
	if (xmax-xmin)/0.2 <= 2:
		xstep_size = 0.1
	else:
		xstep_size = 0.2
	print (tis, xmin, xmax, xstep_size)
	xmax = math.ceil(xmax*10.0)/10
	ymax = math.ceil(ymax*10.0)/10
	xmin = math.floor(xmin*10.0)/10
	ymin = math.floor(ymin*10.0)/10
	y_step_size = 0.2
	plt.yticks(np.arange(ymin, ymax+0.01, y_step_size))
	plt.xticks(np.arange(xmin, ymax+0.01, xstep_size))



	y0,y1 = ax.get_ylim()
	x0,x1 = ax.get_xlim()
	ax.set_aspect(abs(x1-x0)/abs(y1-y0))
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.tight_layout()
	plt.savefig(fig_file+'(r = %.2f, p = %.2e, n = %d)' % (pear, pv, n)+'.pdf')


def plot_silhouette_boxplot(sil_sc, orig_sil_sc, output_file):
	all_data = [sil_sc, orig_sil_sc]
	pvalue = scipy.stats.ttest_rel(sil_sc, orig_sil_sc)[1]
	print (pvalue)

	fig, ax = plt.subplots(figsize=(FIG_WIDTH,FIG_HEIGHT))
	#ax = fig.add_subplot(1,1,1)
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
	plt.plot([], c='#D7191C', label='ONClass')
	plt.plot([], c='#2C7BB6', label='Expression')
	plt.xlim([0,3])
	plt.ylabel('Silhouette coefficient')
	plt.xticks([1,2],['OnClass','Expression'])
	#plt.xlim(-2, len(ticks)*2)

	plt.ylim([-0.4, 1])
	plt.yticks([-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.tight_layout()
	fig.savefig(output_file, bbox_inches='tight')

def plot_umap(embedding, lab, lab2col, file,  lab2marker = None, legend=True, size=1,title='',legendmarker=10):


	if np.shape(embedding)[1]!=2:
		embedding = umap.UMAP(random_state = 1).fit_transform(embedding)
	assert(np.shape(embedding)[1]==2)
	print (size)
	plt.clf()
	fig, ax = plt.subplots(figsize=(FIG_WIDTH,FIG_HEIGHT))
	for l in lab2col:
		ind = np.where(lab==l)[0]
		if len(ind)==0:
			continue
		if lab2marker is None:
			plt.scatter(embedding[ind, 0], embedding[ind, 1],c=lab2col[l], label=l, s=size)
		else:
			plt.scatter(embedding[ind, 0], embedding[ind, 1],c=lab2col[l], label=l, s=size, marker=lab2marker[l])
	#plt.title()
	if legend:
		plt.legend(loc='lower left',ncol=6,fontsize=6)
	plt.title(title)
	plt.tick_params(
	axis='x',		  # changes apply to the x-axis
	which='both',	  # both major and minor ticks are affected
	bottom=False,	  # ticks along the bottom edge are off
	top=False,		 # ticks along the top edge are off
	labelbottom=False) # labels along the bottom edge are off
	plt.tick_params(
	axis='y',		  # changes apply to the x-axis
	which='both',	  # both major and minor ticks are affected
	left=False,	  # ticks along the bottom edge are off
	right=False,		 # ticks along the top edge are off
	labelleft=False) # labels along the bottom edge are off
	#plt.axis('off')
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2)
	plt.xlabel('UMAP 1')
	plt.ylabel('UMAP 2')
	plt.tight_layout()
	plt.savefig(file,dpi=100)
	#plt.savefig(file+'.png',dpi=600)
	if legend:
		return

	handles,labels = ax.get_legend_handles_labels()

	fig_legend = plt.figure(figsize=(6,6))
	axi = fig_legend.add_subplot(111)
	fig_legend.legend(handles, labels, loc='center', scatterpoints = 1, ncol=1, frameon=False,markerscale=legendmarker)
	axi.xaxis.set_visible(False)
	axi.yaxis.set_visible(False)
	plt.savefig(file +'_legend.pdf',dpi=100)
	plt.gcf().clear()


def generate_colors(labels ,use_man_colors = True):
	labels = np.unique(labels)
	n = len(labels)
	man_colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999', '#621e15', '#e59076', '#128dcd', '#083c52', '#64c5f2', '#61afaf', '#0f7369', '#9c9da1', '#365e96', '#983334', '#77973d', '#5d437c', '#36869f', '#d1702f', '#8197c5', '#c47f80', '#acc484', '#9887b0', '#2d588a', '#58954c', '#e9a044', '#c12f32', '#723e77', '#7d807f', '#9c9ede', '#7375b5', '#4a5584', '#cedb9c', '#b5cf6b', '#8ca252', '#637939', '#e7cb94', '#e7ba52', '#bd9e39', '#8c6d31', '#e7969c', '#d6616b', '#ad494a', '#843c39', '#de9ed6', '#ce6dbd', '#a55194', '#7b4173', '#000000', '#0000FF']
	man_colors = np.sort(man_colors)
	if n >= len(man_colors):
		man_colors = get_man_colors()
	nman_colors = len(man_colors)
	man_step = int(np.floor(nman_colors*1./n))
	#print (man_step)
	#print (labels)cmap=ListedColormap(generate_colormap(N*N))
	color_map = plt.cm.get_cmap('gist_rainbow', n)#tab20b
	marker = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x"]
	lab2marker = {}
	lab2col = {}
	for i in range(n):
		if n > len(man_colors) or not use_man_colors:
			lab2col[labels[i]] = color_map(i)
		else:
			lab2col[labels[i]] = man_colors[i*man_step]
		lab2marker[labels[i]] = marker[i % len(marker)]
	return lab2col, lab2marker

def plot_random_cl_plot(mean, error, group_l, method_l, methods2name, fig_dir, title, lab2col, ylabel='', xlabel='Ratio of seen cell types in the test set'):
	mpl.rcParams['pdf.fonttype'] = 42
	SMALL_SIZE = 15
	MEDIUM_SIZE = 25
	BIGGER_SIZE = 15

	plt.rc('font', size=SMALL_SIZE)		  # controls default text sizes
	plt.rc('axes', titlesize=SMALL_SIZE)	 # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)	# fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)	# fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)	# fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)	# legend fontsize
	plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

	fig, ax = plt.subplots(figsize=(8,4))
	#fig, ax = plt.subplots()
	n_groups = len(group_l)
	nmethod = len(method_l)
	index = np.arange(n_groups)
	bar_width = 1. / nmethod * 0.8
	opacity = 0.8

	method2col = generate_colors(method_l)[0]
	ax.errorbar(index + 1, mean, yerr=error, color='black')#, fmt='o'

	#ax.set_xlabel('Different time points',fontsize=20)
	ax.set_ylabel(ylabel,fontsize=20)
	ax.set_xlabel(xlabel)
	#ax.set_title(title,fontsize=18)
	ax.set_xticks(index+1)


	print (index + bar_width * (n_groups-3) / 2 - 1.5 * bar_width)
	ax.set_xticklabels(group_l)
	#fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
	#xticks = mtick.FormatStrFormatter(fmt)
	#ax.xaxis.set_major_formatter(xticks)
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	#x0,x1 = ax.get_xlim()
	#y0,y1 = ax.get_ylim()
	#ax.set_aspect(abs(x1-x0)/abs(y1-y0))
	max_y = min(np.ceil(np.max(mean*10))/10,1.0)
	min_y = max(np.floor(np.min(mean*10))/10,0)
	ax.set_ylim([min_y, max_y])
	if title == 'unseen_auc_macro':
		min_y = 0.4
	print (min_y)
	print (max_y)
	print (ylabel)
	step_size = 0.1
	ax.yaxis.set_ticks(np.arange(min_y, max_y + 0.1-0.0000001, step_size))
	fig.tight_layout()
	plt.savefig(fig_dir +'.pdf')




def heatmap(data, row_labels, col_labels, ax=None,
			cbar_kw={}, cbarlabel="", **kwargs):
	"""
	Create a heatmap from a numpy array and two lists of labels.

	Parameters
	----------
	data
		A 2D numpy array of shape (N, M).
	row_labels
		A list or array of length N with the labels for the rows.
	col_labels
		A list or array of length M with the labels for the columns.
	ax
		A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
		not provided, use current axes or create a new one.  Optional.
	cbar_kw
		A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
	cbarlabel
		The label for the colorbar.  Optional.
	**kwargs
		All other arguments are forwarded to `imshow`.
	"""

	if not ax:
		ax = plt.gca()

	# Plot the heatmap
	im = ax.imshow(data, **kwargs)

	# Create colorbar
	#cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	#cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

	# We want to show all ticks...
	ax.set_xticks(np.arange(data.shape[1]))
	ax.set_yticks(np.arange(data.shape[0]))
	# ... and label them with the respective list entries.
	ax.set_xticklabels(col_labels)
	ax.set_yticklabels(row_labels,)

	# Let the horizontal axes labeling appear on top.
	ax.tick_params(top=False, bottom=True,
				   labeltop=False, labelbottom=True)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=90,ha="right", va="center",
			 rotation_mode="anchor")#, ha="right"

	# Turn spines off and create white grid.
	#for edge, spine in ax.spines.items():
	#	spine.set_visible(False)

	ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
	ax.tick_params(which="minor", bottom=False, left=False)

	return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
					 textcolors=["black", "white"],
					 threshold=None, **textkw):
	"""
	A function to annotate a heatmap.

	Parameters
	----------
	im
		The AxesImage to be labeled.
	data
		Data used to annotate.  If None, the image's data is used.  Optional.
	valfmt
		The format of the annotations inside the heatmap.  This should either
		use the string format method, e.g. "$ {x:.2f}", or be a
		`matplotlib.ticker.Formatter`.  Optional.
	textcolors
		A list or array of two color specifications.  The first is used for
		values below a threshold, the second for those above.  Optional.
	threshold
		Value in data units according to which the colors from textcolors are
		applied.  If None (the default) uses the middle of the colormap as
		separation.  Optional.
	**kwargs
		All other arguments are forwarded to each call to `text` used to create
		the text labels.
	"""

	if not isinstance(data, (list, np.ndarray)):
		data = im.get_array()

	# Normalize the threshold to the images color range.
	if threshold is not None:
		threshold = im.norm(threshold)
	else:
		threshold = im.norm(data.max())/2.

	# Set default alignment to center, but allow it to be
	# overwritten by textkw.
	kw = dict(horizontalalignment="center",
			  verticalalignment="center")
	kw.update(textkw)

	# Get the formatter in case a string is supplied
	if isinstance(valfmt, str):
		valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

	# Loop over the data and create a `Text` for each "pixel".
	# Change the text's color depending on the data.
	texts = []
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if i==j:
				kw.update(color='black')
				#im.axes.text(j, i, valfmt(data[i, j], None), **kw)
				continue
			kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
			text = im.axes.text(j, i, valfmt(data[i, j], None), **kw, fontsize=4)
			texts.append(text)

	return texts


def plot_heatmap_cross_dataset(heat_mat, methods, file_name, ylabel, title='cell_line'):
	plt.clf()
	fig, ax = plt.subplots(figsize=(FIG_WIDTH,FIG_HEIGHT))
	im = heatmap(heat_mat, methods, methods, ax=ax,
					   cmap="YlGn", cbarlabel=ylabel)
	texts = annotate_heatmap(im, valfmt="{x:.2f}")
	plt.title(title)
	fig.tight_layout()
	plt.savefig(file_name)

def plot_26dataset_bar(ctypes, aucs, is_seen, output_file):

	seen = np.where(is_seen==1)[0]
	unseen = np.where(is_seen==0)[0]
	seen_ind = np.argsort(aucs[seen] * -1)
	unseen_ind = np.argsort(aucs[unseen] * -1)
	seen = seen[seen_ind]
	unseen = unseen[unseen_ind]
	ind = np.concatenate((seen, unseen))
	ctypes = ctypes[ind]
	aucs = aucs[ind]
	nseen = len(seen)

	plt.clf()
	fig, ax = plt.subplots(figsize=(FIG_WIDTH*1.2,FIG_HEIGHT))
	y_pos = np.arange(len(ctypes))
	y_pos[nseen:] = y_pos[nseen:] + 1
	width = 0.85
	print (y_pos)
	bars = plt.bar(y_pos, aucs, align='edge', alpha=1, width=width, color='y')
	for i in range(nseen):
		bars[i].set_color('g')
	plt.plot([], c='g', label='Seen cell types')
	plt.plot([], c='y', label='Unseen cell types')
	plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.2), frameon=False, ncol=1, fontsize=4)
	plt.xticks(y_pos+width/2, ctypes, rotation=90)
	ax.set_ylabel('AUROC')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.ylim([0.8, 1.00])
	plt.yticks([0.80,0.85,0.90,0.95,1.])
	y0,y1 = ax.get_ylim()
	x0,x1 = ax.get_xlim()
	#ax.set_aspect(abs(x1-x0)/abs(y1-y0))
	plt.tight_layout()
	plt.savefig(output_file)


def plot_26dataset_more_data_bar(ctypes, aucs, errors, output_file):
	print (ctypes)
	aucs = np.array(aucs)
	errors = np.array(errors)
	fig, ax = plt.subplots(figsize=(FIG_WIDTH,FIG_HEIGHT))
	y_pos = np.arange(len(aucs))
	width = 0.9
	bars = plt.bar(y_pos, aucs, yerr = errors, align='edge', alpha=1, width=width, color='#2C7BB6')

	bars[0].set_color('#D7191C')
	print (ctypes)
	print (y_pos)
	plt.xticks(y_pos+width/2, ctypes, rotation=90)
	plt.ylabel('AUROC')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.ylim([0.6, 1.00])
	plt.yticks([0.6,0.7,0.8,0.9,1.0])
	plt.tight_layout()
	plt.savefig(output_file)


def plot_26dataset_auroc(labels, pred, keyword, fig_file, known_y= True):
	fpr,tpr = roc_curve(labels, pred)[:2]
	auc = roc_auc_score(labels, pred)
	MEDIUM_SIZE = 20
	BIGGER_SIZE = 20

	plt.rc('font', size=MEDIUM_SIZE)		  # controls default text sizes
	plt.rc('axes', titlesize=MEDIUM_SIZE)	 # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)	# fontsize of the x and y labels
	plt.rc('xtick', labelsize=MEDIUM_SIZE)	# fontsize of the tick labels
	plt.rc('ytick', labelsize=MEDIUM_SIZE)	# fontsize of the tick labels
	plt.rc('legend', fontsize=MEDIUM_SIZE)	# legend fontsize
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

	plt.xlabel('False Positive Rate',fontsize=20)
	plt.ylabel('True Positive Rate',fontsize=20)
	if known_y:
		suffix = 'in TMS'
	else:
		suffix = 'not in TMS'

	plt.title(keyword,fontsize=20)
	plt.legend(loc="lower right",fontsize=14,frameon=False)
	plt.tight_layout()
	plt.savefig(fig_file)#


def plot_auroc_curve(labels, pred, title, fig_file):
	fpr,tpr = roc_curve(labels, pred)[:2]
	auc = roc_auc_score(labels, pred)
	fig, ax = plt.subplots(figsize=(FIG_WIDTH,FIG_HEIGHT))
	lw = 1
	plt.plot(fpr, tpr, color='darkorange',
			 lw=lw, label='AUROC = %0.2f'%auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')

	plt.title(title)
	plt.legend(loc="lower right",frameon=False, fontsize=6)
	plt.tight_layout()
	plt.savefig(fig_file)#



def plot_auc_region_violin(datas, jitter_datas, fig_file, xticks,cutoff):
	plt.clf()
	fig, ax = plt.subplots(figsize=(FIG_WIDTH*1.5,FIG_HEIGHT))# fontsize of the f
	ax = sns.boxplot( data=datas, color=".8")
	#ax.violinplot(datas)
	ax = sns.stripplot(data=jitter_datas, jitter=True, size=1)
	ax.set_ylabel('AUROC')
	ax.set_xlabel('Number of seen cell types in the '+str(cutoff)+'-hop region')
	#ax.violinplot(datas)
	#ax.set_ylim([0,1])
	ax.set_xticklabels(xticks)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.tight_layout()
	plt.savefig(fig_file)

def plot_auc_shortest_distance_boxplot(datas, jitter_datas, fig_file, xticks):
	plt.clf()
	fig, ax = plt.subplots(figsize=(FIG_WIDTH*1.5,FIG_HEIGHT))# fontsize of the f
	#fig = plt.gcf()

	#ax1.set_title('Compare')
	#inner='box'
	ax = sns.boxplot( data=datas, color=".8")
	ax = sns.stripplot(data=jitter_datas, jitter=True, size=1)
	ax.set_ylabel('AUROC')
	ax.set_xlabel('Distance to the nearest seen cell type')
	#ax.violinplot(datas)
	#ax.set_ylim([0,1])
	ax.set_xticklabels(xticks)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.tight_layout()
	plt.savefig(fig_file)

def plot_nlp_shortest_distance_boxplot(datas, jitter_datas, fig_file, xticks):

	plt.clf()
	fig, ax = plt.subplots(figsize=(FIG_WIDTH*2,FIG_HEIGHT))# fontsize of the figure title

	ax = sns.violinplot( data=datas,
						inner='box', color=".8")
	ax = sns.stripplot(data=jitter_datas, jitter=True, size=2)
	ax.set_ylabel('Text-based\ncell type similarity')
	ax.set_xlabel('Shortest distance in the Cell Ontology')
	#ax.violinplot(datas)

	ax.set_xticklabels(xticks)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.tight_layout()
	plt.savefig(fig_file)

def plot_comparison_baselines_bar(ax, mean, error, group_l, method_l, fig_dir, title, lab2col, write_xlabel = True, write_title = True, ylabel='', xlabel='Ratio of unseen cell types in the test set'):


	#fig, ax = plt.subplots(figsize=(8,4))
	#fig, ax = plt.subplots()
	n_groups = len(group_l)
	nmethod = len(method_l)
	index = np.arange(n_groups)
	bar_width = 1. / nmethod * 0.8
	opacity = 0.8

	method2col = generate_colors(method_l)[0]

	#color_l = ['#F79821','#A2CF57','#7AAF41','black','pink','red','blue']
	#index1 = list(range(len(method_l)))
	#index1.reverse()
	for i in list(range(len(method_l))):
		#print len(mean[:,i]),index,np.shape(mean),np.shape(method_l)
		ax.bar(index+(nmethod-1-i)*bar_width, mean[:,i], yerr = error[:,i], width=bar_width,alpha=opacity,
						color=lab2col[method_l[i]],#,color_l[i],
						label=method_l[i])

	#ax.set_xlabel('Different time points',fontsize=20)
	ax.set_ylabel(ylabel)
	if write_xlabel:
		ax.set_xlabel(xlabel)
	ax.set_xticklabels(group_l)
	if write_title:
		ax.set_title(title, fontsize = BIGGER_SIZE)

	if nmethod==1:
		ax.set_xticks(index)
	else:
		ax.set_xticks(index + bar_width * (nmethod-0.5) *1. / 2 )

	print (index, nmethod, n_groups)
	print (index + bar_width* nmethod  *1. / 2 )

	#fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
	#xticks = mtick.FormatStrFormatter(fmt)
	#ax.xaxis.set_major_formatter(xticks)

	#x0,x1 = ax.get_xlim()
	#y0,y1 = ax.get_ylim()
	#ax.set_aspect(abs(x1-x0)/abs(y1-y0))
	max_y = min(np.ceil(np.max(mean*10))/10,1.0)
	min_y = max(np.floor(np.min(mean*10))/10,0)
	ax.set_ylim([min_y, max_y])
	if 'AUROC' in ylabel and min_y < 0.51:
		min_y = min(0.4,min_y)
	for step_size in [0.2,0.1,0.05]:
		ngap = (max_y-min_y) / step_size
		if ngap>3:
			break
	if step_size == 0.05:
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	else:
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	ax.yaxis.set_ticks(np.arange(min_y, max_y + 0.1-0.0000001, step_size))
	return ax


def plot_marker_comparison_baselines_bar(mean, error, group_l, method_l, lab2col, ylabel = '', output_file = ''):

	fig, ax = plt.subplots(figsize=(FIG_WIDTH*1,FIG_HEIGHT))
	nmethod, ngroup = np.shape(mean)

	n_groups = len(group_l)
	nmethod = len(method_l)
	index = np.arange(n_groups)
	bar_width = 1. / nmethod * 0.8
	opacity = 0.8

	method2col = generate_colors(method_l)[0]

	for i in list(range(len(method_l))):
		#print len(mean[:,i]),index,np.shape(mean),np.shape(method_l)
		ax.bar(index+(nmethod-1-i)*bar_width, mean[:,i], yerr = error[:,i], width=bar_width,alpha=opacity,
						color=lab2col[method_l[i]],#,color_l[i],
						label=method_l[i])

	#ax.set_xlabel('Different time points',fontsize=20)
	print (group_l)
	ax.set_ylabel(ylabel)
	ax.set_xticklabels(group_l)
	print (index, index + bar_width * (nmethod-0.5) *1. / 2 )
	if nmethod==1:
		ax.set_xticks(index)
	else:
		ax.set_xticks(index + bar_width * (nmethod-0.5) *1. / 2 - 0.1 )
	plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), frameon=False, ncol=1, fontsize=4)
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

	plt.setp(ax.get_xticklabels(), rotation=90, ha="right", va='center',
			 rotation_mode="anchor")

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	max_y = min(np.ceil(np.max(mean*10))/10,1.0)
	min_y = max(np.floor(np.min(mean*10))/10,0)
	ax.set_ylim([min_y, max_y])
	step_size = 0.1
	ax.yaxis.set_ticks(np.arange(min_y, max_y + 0.1-0.0000001, step_size))
	#ax.legend(method_l)
	fig.tight_layout()
	plt.savefig(output_file)


def plot_marker_comparison_prediction_accuracy_bar(mean, error, group_l, method_l, lab2col, ylabel = '', output_file = ''):
	fig, ax = plt.subplots(figsize=(FIG_WIDTH*1.2,FIG_HEIGHT))

	nmethod, ngroup = np.shape(mean)

	n_groups = len(group_l)
	nmethod = len(method_l)
	index = np.arange(n_groups)
	bar_width = 1. / nmethod * 0.8
	opacity = 0.8

	method2col = generate_colors(method_l)[0]

	for i in list(range(len(method_l))):
		#print len(mean[:,i]),index,np.shape(mean),np.shape(method_l)
		ax.bar(index+(nmethod-1-i)*bar_width, mean[:,i], yerr = error[:,i], width=bar_width,alpha=opacity,
						color=lab2col[method_l[i]],#,color_l[i],
						label=method_l[i])

	#ax.set_xlabel('Different time points',fontsize=20)
	ax.set_ylabel(ylabel)
	ax.set_xticklabels(group_l)

	if nmethod==1:
		ax.set_xticks(index)
	else:
		ax.set_xticks(index + bar_width * (nmethod-0.5) *1. / 2 - 0.1)
	plt.legend(loc='upper left',frameon=False, ncol=1, fontsize=4)
	#plt.legend(loc='upper left',bbox_to_anchor=(0.1, 1.1), frameon=False, ncol=1, fontsize=4)


	plt.setp(ax.get_xticklabels(), rotation=90,ha="right", va="center",
			 rotation_mode="anchor")

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	max_y = 1.0#min(np.ceil(np.max(mean*10))/10,1.0)
	min_y = max(np.floor(np.min(mean*10))/10,0)
	ax.set_ylim([min_y, max_y])
	if min_y<0.8:
		step_size = 0.1
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	else:
		step_size = 0.05
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	print (min_y, max_y)
	ax.yaxis.set_ticks(np.arange(min_y, max_y+step_size/2, step_size))
	#ax.legend(method_l)
	fig.tight_layout()
	plt.savefig(output_file)

def plot_comparison_baselines_bar_legend(ax, mean, error, group_l, method_l, fig_dir, title, lab2col, write_xlabel = True, write_title = True, ylabel='', xlabel='Ratio of unseen cell types\nin the test set'):
	mpl.rcParams['pdf.fonttype'] = 42
	SMALL_SIZE = 8
	MEDIUM_SIZE = 10
	BIGGER_SIZE = 25

	plt.rc('font', size=SMALL_SIZE)		  # controls default text sizes
	plt.rc('axes', titlesize=SMALL_SIZE)	 # fontsize of the axes title
	plt.rc('axes', labelsize=SMALL_SIZE)	# fontsize of the x and y labels
	#plt.rc('axes', fontsize=SMALL_SIZE)	# fontsize of the x and y labels
	#plt.rc('xtick', titlesize=SMALL_SIZE)	# fontsize of the tick labels
	plt.rc('xtick', labelsize=SMALL_SIZE)	# fontsize of the tick labels
	#plt.rc('xtick', fontsize=SMALL_SIZE)	# fontsize of the tick labels
	#plt.rc('ytick', titlesize=SMALL_SIZE)	# fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)	# fontsize of the tick labels
	#plt.rc('ytick', fontsize=SMALL_SIZE)	# fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)	# legend fontsize
	plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
	#plt.rc('title', titlesize=BIGGER_SIZE)  # fontsize of the figure title






	#fig, ax = plt.subplots(figsize=(8,4))
	#fig, ax = plt.subplots()
	n_groups = len(group_l)
	nmethod = len(method_l)
	index = np.arange(n_groups)
	bar_width = 1. / nmethod * 0.8
	opacity = 0.8

	method2col = generate_colors(method_l)[0]

	#color_l = ['#F79821','#A2CF57','#7AAF41','black','pink','red','blue']
	#index1 = list(range(len(method_l)))
	#index1.reverse()
	for i in list(range(len(method_l))):
		#print len(mean[:,i]),index,np.shape(mean),np.shape(method_l)
		ax.bar(index+(nmethod-1-i)*bar_width, mean[:,i], yerr = error[:,i], width=bar_width,alpha=opacity,
						color=lab2col[method_l[i]],#,color_l[i],
						label=method_l[i])

	#ax.set_xlabel('Different time points',fontsize=20)
	ax.set_ylabel(ylabel)
	if write_xlabel:
		ax.set_xlabel(xlabel)
	ax.set_xticklabels(group_l)
	if write_title:
		ax.set_title(title, fontsize = BIGGER_SIZE)

	if nmethod==1:
		ax.set_xticks(index)
	else:
		ax.set_xticks(index + bar_width * (nmethod-0.5) *1. / 2 )

	#print (index, nmethod, n_groups)
	#print (index + bar_width* nmethod  *1. / 2 )

	#fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
	#xticks = mtick.FormatStrFormatter(fmt)
	#ax.xaxis.set_major_formatter(xticks)
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	#x0,x1 = ax.get_xlim()
	#y0,y1 = ax.get_ylim()
	#ax.set_aspect(abs(x1-x0)/abs(y1-y0))
	max_y = min(np.ceil(np.max(mean*10))/10,1.0)
	min_y = max(np.floor(np.min(mean*10))/10,0)
	ax.set_ylim([min_y, max_y])
	if title == 'unseen_auc_macro':
		min_y = 0.4
	if min_y<0.4:
		step_size = 0.2
	else:
		step_size = 0.1
	ax.yaxis.set_ticks(np.arange(min_y, max_y + 0.1-0.0000001, step_size))

	#handles,labels = ax.get_legend_handles_labels()
	ax.set_legend(method_l)
	#fig_legend = plt.figure(figsize=(1,1))
	#axi = fig_legend.add_subplot(111)
	#
	#axi.xaxis.set_visible(False)
	#axi.yaxis.set_visible(False)
	return axi

def sankey(left, right, leftWeight=None, rightWeight=None, colorDict=None,
		   leftLabels=None, rightLabels=None, aspect=4, rightColor=False,
		   fontsize=14, figureName=None, closePlot=False):
	'''
	Make Sankey Diagram showing flow from left-->right
	Inputs:
		left = NumPy array of object labels on the left of the diagram
		right = NumPy array of corresponding labels on the right of the diagram
			len(right) == len(left)
		leftWeight = NumPy array of weights for each strip starting from the
			left of the diagram, if not specified 1 is assigned
		rightWeight = NumPy array of weights for each strip starting from the
			right of the diagram, if not specified the corresponding leftWeight
			is assigned
		colorDict = Dictionary of colors to use for each label
			{'label':'color'}
		leftLabels = order of the left labels in the diagram
		rightLabels = order of the right labels in the diagram
		aspect = vertical extent of the diagram in units of horizontal extent
		rightColor = If true, each strip in the diagram will be be colored
					according to its left label
	Ouput:
		None
	'''
	if leftWeight is None:
		leftWeight = []
	if rightWeight is None:
		rightWeight = []
	if leftLabels is None:
		leftLabels = []
	if rightLabels is None:
		rightLabels = []
	# Check weights
	if len(leftWeight) == 0:
		leftWeight = np.ones(len(left))

	if len(rightWeight) == 0:
		rightWeight = leftWeight

	#plt.figure()
	#plt.rc('text', usetex=False)
	#plt.rc('font', family='serif')
	plt.clf()
	fig, ax = plt.subplots(figsize=(FIG_WIDTH,FIG_HEIGHT))
	# Create Dataframe
	if isinstance(left, pd.Series):
		left = left.reset_index(drop=True)
	if isinstance(right, pd.Series):
		right = right.reset_index(drop=True)
	dataFrame = pd.DataFrame({'left': left, 'right': right, 'leftWeight': leftWeight,
							  'rightWeight': rightWeight}, index=range(len(left)))

	if len(dataFrame[(dataFrame.left.isnull()) | (dataFrame.right.isnull())]):
		raise NullsInFrame('Sankey graph does not support null values.')

	# Identify all labels that appear 'left' or 'right'
	allLabels = pd.Series(np.r_[dataFrame.left.unique(), dataFrame.right.unique()]).unique()

	# Identify left labels
	if len(leftLabels) == 0:
		leftLabels = pd.Series(dataFrame.left.unique()).unique()
	else:
		check_data_matches_labels(leftLabels, dataFrame['left'], 'left')

	# Identify right labels
	if len(rightLabels) == 0:
		rightLabels = pd.Series(dataFrame.right.unique()).unique()
	else:
		check_data_matches_labels(leftLabels, dataFrame['right'], 'right')
	# If no colorDict given, make one
	if colorDict is None:
		colorDict = {}
		palette = "hls"
		colorPalette = sns.color_palette(palette, len(allLabels))
		for i, label in enumerate(allLabels):
			colorDict[label] = colorPalette[i]
	else:
		missing = [label for label in allLabels if label not in colorDict.keys()]
		if missing:
			msg = "The colorDict parameter is missing values for the following labels : "
			msg += '{}'.format(', '.join(missing))
			raise ValueError(msg)

	# Determine widths of individual strips
	ns_l = defaultdict()
	ns_r = defaultdict()
	for leftLabel in leftLabels:
		leftDict = {}
		rightDict = {}
		for rightLabel in rightLabels:
			leftDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].leftWeight.sum()
			rightDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].rightWeight.sum()
		ns_l[leftLabel] = leftDict
		ns_r[leftLabel] = rightDict

	# Determine positions of left label patches and total widths
	leftWidths = defaultdict()
	for i, leftLabel in enumerate(leftLabels):
		myD = {}
		myD['left'] = dataFrame[dataFrame.left == leftLabel].leftWeight.sum()
		if i == 0:
			myD['bottom'] = 0
			myD['top'] = myD['left']
		else:
			myD['bottom'] = leftWidths[leftLabels[i - 1]]['top'] + 0.02 * dataFrame.leftWeight.sum()
			myD['top'] = myD['bottom'] + myD['left']
			topEdge = myD['top']
		leftWidths[leftLabel] = myD

	# Determine positions of right label patches and total widths
	rightWidths = defaultdict()
	for i, rightLabel in enumerate(rightLabels):
		myD = {}
		myD['right'] = dataFrame[dataFrame.right == rightLabel].rightWeight.sum()
		if i == 0:
			myD['bottom'] = 0
			myD['top'] = myD['right']
		else:
			myD['bottom'] = rightWidths[rightLabels[i - 1]]['top'] + 0.02 * dataFrame.rightWeight.sum()
			myD['top'] = myD['bottom'] + myD['right']
			topEdge = myD['top']
		rightWidths[rightLabel] = myD

	# Total vertical extent of diagram
	xMax = topEdge / aspect

	# Draw vertical bars on left and right of each  label's section & print label
	for leftLabel in leftLabels:
		plt.fill_between(
			[-0.02 * xMax, 0],
			2 * [leftWidths[leftLabel]['bottom']],
			2 * [leftWidths[leftLabel]['bottom'] + leftWidths[leftLabel]['left']],
			color=colorDict[leftLabel],
			alpha=0.99
		)
		'''
		plt.text(
			-0.05 * xMax,
			leftWidths[leftLabel]['bottom'] + 0.5 * leftWidths[leftLabel]['left'],
			leftLabel,
			{'ha': 'right', 'va': 'center'},
			fontsize=fontsize
		)
		'''
	for rightLabel in rightLabels:
		plt.fill_between(
			[xMax, 1.02 * xMax], 2 * [rightWidths[rightLabel]['bottom']],
			2 * [rightWidths[rightLabel]['bottom'] + rightWidths[rightLabel]['right']],
			color=colorDict[rightLabel],
			alpha=0.99
		)
		'''
		plt.text(
			1.05 * xMax,
			rightWidths[rightLabel]['bottom'] + 0.5 * rightWidths[rightLabel]['right'],
			rightLabel,
			{'ha': 'left', 'va': 'center'},
			fontsize=fontsize
		)
		'''


	# Plot strips
	for leftLabel in leftLabels:
		for rightLabel in rightLabels:
			labelColor = leftLabel
			if rightColor:
				labelColor = rightLabel
			if len(dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)]) > 0:
				# Create array of y values for each strip, half at left value,
				# half at right, convolve
				ys_d = np.array(50 * [leftWidths[leftLabel]['bottom']] + 50 * [rightWidths[rightLabel]['bottom']])
				ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
				ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
				ys_u = np.array(50 * [leftWidths[leftLabel]['bottom'] + ns_l[leftLabel][rightLabel]] + 50 * [rightWidths[rightLabel]['bottom'] + ns_r[leftLabel][rightLabel]])
				ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
				ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')

				# Update bottom edges at each label so next strip starts at the right place
				leftWidths[leftLabel]['bottom'] += ns_l[leftLabel][rightLabel]
				rightWidths[rightLabel]['bottom'] += ns_r[leftLabel][rightLabel]
				plt.fill_between(
					np.linspace(0, xMax, len(ys_d)), ys_d, ys_u, alpha=0.65,
					color=colorDict[labelColor]
				)
	plt.gca().axis('off')
	plt.gcf().set_size_inches(6, 6)
	if figureName != None:
		plt.savefig("{}.pdf".format(figureName), bbox_inches='tight', dpi=150)
	if closePlot:
		plt.close()

def plot_sankey_diagram(pred_Y, truth_Y, fig_file,colorDict=None):
	sankey(pred_Y, truth_Y, aspect=20,
	fontsize=10, figureName=fig_file,colorDict=colorDict)#, colorDict=colorDict
	return




def plot_comparison_bar(mean, group_l, method_l, fig_dir, title=''):
	xtickslabels = ['0','1-50','51-200','201-500','>500']
	fig, ax = plt.subplots(figsize=(4,4))
	n_groups = len(group_l)
	index = np.arange(n_groups)
	bar_width = 0.1
	opacity = 0.8
	color_l = ['#F79821','#A2CF57','#7AAF41','black','pink','red','blue']
	index1 = list(range(len(method_l)))
	index1.reverse()
	for i in index1:
		#print len(mean[:,i]),index,np.shape(mean),np.shape(method_l)
		ax.bar(index+(len(method_l)-1-i)*bar_width, mean[:,i], width=bar_width,alpha=opacity,
						color=cm.jet(1.*i/len(index1)),#,color_l[i],
						label=method_l[i])

	#ax.set_xlabel('Different time points',fontsize=20)
	ax.set_ylabel(title,fontsize=20)
	#ax.set_title(title,fontsize=18)
	ax.set_xticks(index + bar_width * (n_groups+1) / 2)
	ax.set_xticklabels(xtickslabels, rotation=90)
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	#x0,x1 = ax.get_xlim()
	#y0,y1 = ax.get_ylim()
	#ax.set_aspect(abs(x1-x0)/abs(y1-y0))
	max_y = min(np.ceil(np.max(mean*10))/10+0.05,1.0)
	min_y = max(np.floor(np.min(mean*10))/10-0.05,0)
	ax.set_ylim([min_y, max_y])

	fig.tight_layout()
	plt.savefig(fig_dir +'.png')
	plt.savefig(fig_dir +'.pdf')
	handles,labels = ax.get_legend_handles_labels()

	fig_legend = plt.figure(figsize=(10,30))
	axi = fig_legend.add_subplot(111)
	fig_legend.legend(handles, labels, loc='center', scatterpoints = 1, ncol=len(method_l), frameon=False)
	axi.xaxis.set_visible(False)
	axi.yaxis.set_visible(False)
	plt.savefig(fig_dir +'_legend.png')
	plt.savefig(fig_dir +'_legend.pdf')
	plt.gcf().clear()

def get_sparse_cat(s):
	if s==0:
		return 0
	if s<5:
		return 1
	if s<10:
		return 2
	if s<50:
		return 3
	return 4
