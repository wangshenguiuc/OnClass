'''
This scirpt is mostly adpated from Scanorama project which provides a very easy way to read and process single cell datasets.
We are very grateful for Scanorama authors to make these scripts and data available.
Please check the Scanorama project for more information.
'''

from scanorama import *

import gzip
import numpy as np
import os.path
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import sys

from scanorama import merge_datasets

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)		  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)	 # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)	# fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)	# fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)	# fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)	# legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

data_names_all = ['26-datasets/293t_jurkat/293t',
'26-datasets/293t_jurkat/jurkat',
'26-datasets/293t_jurkat/jurkat_293t_50_50',
'26-datasets/293t_jurkat/jurkat_293t_99_1',
'26-datasets/brain/neuron_9k',
'26-datasets/macrophage/infected',
'26-datasets/macrophage/mixed_infected',
'26-datasets/macrophage/uninfected',
'26-datasets/macrophage/uninfected_donor2',
'26-datasets/hsc/hsc_mars',
'26-datasets/hsc/hsc_ss2',
'26-datasets/pancreas/pancreas_inDrop',
'26-datasets/pancreas/pancreas_multi_celseq2_expression_matrix',
'26-datasets/pancreas/pancreas_multi_celseq_expression_matrix',
'26-datasets/pancreas/pancreas_multi_fluidigmc1_expression_matrix',
'26-datasets/pancreas/pancreas_multi_smartseq2_expression_matrix',
'26-datasets/pbmc/10x/68k_pbmc',
'26-datasets/pbmc/10x/b_cells',
'26-datasets/pbmc/10x/cd14_monocytes',
'26-datasets/pbmc/10x/cd4_t_helper',
'26-datasets/pbmc/10x/cd56_nk',
'26-datasets/pbmc/10x/cytotoxic_t',
'26-datasets/pbmc/10x/memory_t',
'26-datasets/pbmc/10x/regulatory_t',
'26-datasets/pbmc/pbmc_kang',
'26-datasets/pbmc/pbmc_10X']


def load_data(name):
	if os.path.isfile(name + '.h5.npz'):
		X = scipy.sparse.load_npz(name + '.h5.npz')
		with open(name + '.h5.genes.txt') as f:
			genes = np.array(f.read().rstrip().split())
	elif os.path.isfile(name + '.npz'):
		data = np.load(name + '.npz')
		X = data['X']
		genes = data['genes']
		data.close()
	elif os.path.isfile(name + '/tab.npz'):
		X = scipy.sparse.load_npz(name + '/tab.npz')
		with open(name + '/tab.genes.txt') as f:
			genes = np.array(f.read().rstrip().split())
	else:
		sys.stderr.write('Could not find: {}\n'.format(name))
		exit(1)
	genes = np.array([ gene.upper() for gene in genes ])
	return X, genes

def load_names(data_names, norm=True, log1p=False, verbose=True, DATA_DIR = ''):
	print (DATA_DIR)
	# Load datasets.
	datasets = []
	genes_list = []
	n_cells = 0
	for name in data_names:
		X_i, genes_i = load_data(DATA_DIR + name)
		if norm:
			X_i = normalize(X_i, axis=1)
		if log1p:
			X_i = np.log1p(X_i)
		X_i = csr_matrix(X_i)

		datasets.append(X_i)
		genes_list.append(genes_i)
		n_cells += X_i.shape[0]
		if verbose:
			print('Loaded {} with {} genes and {} cells'.
				  format(name, X_i.shape[1], X_i.shape[0]))
	if verbose:
		print('Found {} cells among all datasets'
			  .format(n_cells))

	return datasets, genes_list, n_cells

	

def my_assemble(datasets, verbose=VERBOSE, view_match=False, knn=KNN,
			 sigma=SIGMA, approx=APPROX, alpha=ALPHA, expr_datasets=None,
			 ds_names=None, batch_size=None,
			 geosketch=False, geosketch_max=20000, alignments=None, matches=None): # reimplement part of scanorama to return the corrected expression (instead of low-d vectors)
	#this code is copy and paste from scanorama in order to output the expression. Please check their tool and cite their paper if you used this function.
	if len(datasets) == 1:
		return datasets

	if alignments is None and matches is None:
		alignments, matches = find_alignments(
			datasets, knn=knn, approx=approx, alpha=alpha, verbose=verbose,
			geosketch=geosketch, geosketch_max=geosketch_max
		)

	ds_assembled = {}
	panoramas = []
	for i, j in alignments:
		if verbose:
			if ds_names is None:
				print('Processing datasets {}'.format((i, j)))
			else:
				print('Processing datasets {} <=> {}'.
					  format(ds_names[i], ds_names[j]))

		# Only consider a dataset a fixed amount of times.
		if not i in ds_assembled:
			ds_assembled[i] = 0
		ds_assembled[i] += 1
		if not j in ds_assembled:
			ds_assembled[j] = 0
		ds_assembled[j] += 1
		if ds_assembled[i] > 3 and ds_assembled[j] > 3:
			continue

		# See if datasets are involved in any current panoramas.
		panoramas_i = [ panoramas[p] for p in range(len(panoramas))
						if i in panoramas[p] ]
		assert(len(panoramas_i) <= 1)
		panoramas_j = [ panoramas[p] for p in range(len(panoramas))
						if j in panoramas[p] ]
		assert(len(panoramas_j) <= 1)

		if len(panoramas_i) == 0 and len(panoramas_j) == 0:
			if datasets[i].shape[0] < datasets[j].shape[0]:
				i, j = j, i
			panoramas.append([ i ])
			panoramas_i = [ panoramas[-1] ]

		# Map dataset i to panorama j.
		if len(panoramas_i) == 0:
			curr_ds = datasets[i]
			curr_ref = np.concatenate([ datasets[p] for p in panoramas_j[0] ])

			match = []
			base = 0
			for p in panoramas_j[0]:
				if i < p and (i, p) in matches:
					match.extend([ (a, b + base) for a, b in matches[(i, p)] ])
				elif i > p and (p, i) in matches:
					match.extend([ (b, a + base) for a, b in matches[(p, i)] ])
				base += datasets[p].shape[0]

			ds_ind = [ a for a, _ in match ]
			ref_ind = [ b for _, b in match ]

			bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma=sigma,
							 batch_size=batch_size)
			datasets[i] = curr_ds + bias

			if expr_datasets:
				curr_ds = expr_datasets[i]
				curr_ref = vstack([ expr_datasets[p]
									for p in panoramas_j[0] ])
				bias = transform(curr_ds, curr_ref, ds_ind, ref_ind,
								 sigma=sigma, cn=True, batch_size=batch_size)
				expr_datasets[i] = curr_ds + bias

			panoramas_j[0].append(i)

		# Map dataset j to panorama i.
		elif len(panoramas_j) == 0:
			curr_ds = datasets[j]
			curr_ref = np.concatenate([ datasets[p] for p in panoramas_i[0] ])

			match = []
			base = 0
			for p in panoramas_i[0]:
				if j < p and (j, p) in matches:
					match.extend([ (a, b + base) for a, b in matches[(j, p)] ])
				elif j > p and (p, j) in matches:
					match.extend([ (b, a + base) for a, b in matches[(p, j)] ])
				base += datasets[p].shape[0]

			ds_ind = [ a for a, _ in match ]
			ref_ind = [ b for _, b in match ]

			bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma=sigma,
							 batch_size=batch_size)
			datasets[j] = curr_ds + bias

			if expr_datasets:
				curr_ds = expr_datasets[j]
				curr_ref = vstack([ expr_datasets[p]
									for p in panoramas_i[0] ])
				bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma=sigma,
								 cn=True, batch_size=batch_size)
				expr_datasets[j] = curr_ds + bias

			panoramas_i[0].append(j)

		# Merge two panoramas together.
		else:
			curr_ds = np.concatenate([ datasets[p] for p in panoramas_i[0] ])
			curr_ref = np.concatenate([ datasets[p] for p in panoramas_j[0] ])

			# Find base indices into each panorama.
			base_i = 0
			for p in panoramas_i[0]:
				if p == i: break
				base_i += datasets[p].shape[0]
			base_j = 0
			for p in panoramas_j[0]:
				if p == j: break
				base_j += datasets[p].shape[0]

			# Find matching indices.
			match = []
			base = 0
			for p in panoramas_i[0]:
				if p == i and j < p and (j, p) in matches:
					match.extend([ (b + base, a + base_j)
								   for a, b in matches[(j, p)] ])
				elif p == i and j > p and (p, j) in matches:
					match.extend([ (a + base, b + base_j)
								   for a, b in matches[(p, j)] ])
				base += datasets[p].shape[0]
			base = 0
			for p in panoramas_j[0]:
				if p == j and i < p and (i, p) in matches:
					match.extend([ (a + base_i, b + base)
								   for a, b in matches[(i, p)] ])
				elif p == j and i > p and (p, i) in matches:
					match.extend([ (b + base_i, a + base)
								   for a, b in matches[(p, i)] ])
				base += datasets[p].shape[0]

			ds_ind = [ a for a, _ in match ]
			ref_ind = [ b for _, b in match ]

			# Apply transformation to entire panorama.
			bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma=sigma,
							 batch_size=batch_size)
			curr_ds += bias
			base = 0
			for p in panoramas_i[0]:
				n_cells = datasets[p].shape[0]
				datasets[p] = curr_ds[base:(base + n_cells), :]
				base += n_cells

			if not expr_datasets is None:
				curr_ds = vstack([ expr_datasets[p]
								   for p in panoramas_i[0] ])
				curr_ref = vstack([ expr_datasets[p]
									for p in panoramas_j[0] ])
				bias = transform(curr_ds, curr_ref, ds_ind, ref_ind,
								 sigma=sigma, cn=True, batch_size=batch_size)
				curr_ds += bias
				base = 0
				for p in panoramas_i[0]:
					n_cells = expr_datasets[p].shape[0]
					expr_datasets[p] = curr_ds[base:(base + n_cells), :]
					base += n_cells

			# Merge panoramas i and j and delete one.
			if panoramas_i[0] != panoramas_j[0]:
				panoramas_i[0] += panoramas_j[0]
				panoramas.remove(panoramas_j[0])

		# Visualize.
		if view_match:
			plot_mapping(curr_ds, curr_ref, ds_ind, ref_ind)

	return datasets, expr_datasets


def run_scanorama(test_X, test_genes, train_X, train_genes, scan_dim = 100):
	
	datasets = [train_X, test_X]
	genes_list = [train_genes, test_genes]
	datasets, genes = scanorama.merge_datasets(datasets, genes_list)
	datasets_dimred, genes = scanorama.process_data(datasets, genes, dimred=scan_dim)
	datasets_dimred, expr_datasets = my_assemble(datasets_dimred, expr_datasets = datasets, sigma=150)
	exp = expr_datasets[1]
	exp = exp.todense()
	ntest_cell = np.shape(test_X)[0]
	ntrain_gene = len(train_genes)
	new_test_x = np.zeros((ntest_cell, ntrain_gene))
	train_genes = list(train_genes)
	for i,g in enumerate(genes):
		ind = [train_genes.index(g)]
		new_test_x[:,ind] = exp[:,i]
	return new_test_x
