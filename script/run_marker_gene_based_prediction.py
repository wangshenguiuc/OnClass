import sys
from scipy import stats, sparse
import numpy as np
import os
from collections import Counter, defaultdict
from OnClass.OnClassModel import OnClassModel
from sklearn.metrics import roc_auc_score
from utils import read_ontology_file, ConvertLabels, MapLabel2CL, read_data, read_data_file, read_data, parse_pkl, SplitTrainTest, read_type2genes, corr2_coeff
from config import ontology_data_dir, scrna_data_dir, result_dir, intermediate_dir

dnames = ['muris_facs','muris_droplet','microcebusBernard','microcebusStumpy','microcebusMartine','microcebusAntoine']

marker_gene_file = scrna_data_dir + 'gene_marker_expert_curated.txt'
has_truth_our_auc_mat = defaultdict(dict)
has_truth_base_auc_mat = defaultdict(dict)
no_truth_our_auc_mat = defaultdict(dict)
for dnamei, dname1 in enumerate(dnames):
	cor_file = intermediate_dir + '/Marker_genes/' +dname1+ 'cor.released.npy'
	cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file(dname1, ontology_data_dir)
	OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
	feature_file, filter_key, label_key, label_file, gene_file = read_data_file(dname1, scrna_data_dir)
	if feature_file.endswith('.pkl'):
		feature, label, genes = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
	elif feature_file.endswith('.h5ad'):
		feature, genes, label, _, _ = read_data(feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
		exclude_non_leaf_ontology = True, tissue_key = 'tissue', filter_key = filter_key, AnnData_label_key=label_key,
		nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp)

	OnClass_train_obj.EmbedCellTypes(label)
	g2i1 = {g : i for i,g in enumerate(genes)}
	i2g1 = {i : g for i,g in enumerate(genes)}
	tp2genes_base = read_type2genes(g2i1, marker_gene_file, cl_obo_file)
	cor = np.load(cor_file)
	exist_y = np.unique(label)

	for dnamej, dname2 in enumerate(dnames):
		if dname1 == dname2:
			continue
		feature_file, filter_key, label_key, label_file, gene_file = read_data_file(dname2, scrna_data_dir)
		if feature_file.endswith('.pkl'):
			feature2, label2, genes2 = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
		elif feature_file.endswith('.h5ad'):
			feature2, genes2, label2, _, _ = read_data(feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
			exclude_non_leaf_ontology = True, tissue_key = 'tissue', filter_key = filter_key, AnnData_label_key=label_key,
			nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp)
		OnClass_test_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
		OnClass_test_obj.EmbedCellTypes(label2)
		nseen = OnClass_test_obj.nseen
		g2i2 = {g : i for i,g in enumerate(genes2)}
		common_genes = list(set(genes) & set(genes2))

		has_truth_our_auc = []
		has_truth_tp_base_auc = []
		no_truth_our_auc = []
		topk = 50
		thres = 0.4
		sub_cor = np.where(np.abs(cor)>thres, 1, 0)
		nmarkers = []
		label_mat2 = ConvertLabels(MapLabel2CL(label2, OnClass_test_obj.co2i), len(OnClass_test_obj.i2co))

		for i in range(nseen):
			tp = OnClass_test_obj.i2co[i]
			our_marker = np.where(sub_cor[OnClass_train_obj.co2i[tp],:]==1)[0]#np.argsort(cor[l2i1[tp],:]*-1)[:topk]
			our_marker_id = [g2i2[i2g1[gi]] for gi in our_marker if i2g1[gi] in g2i2]
			nmarkers.append(len(our_marker_id))
			if len(our_marker_id) ==0:
				our_marker = np.argsort(cor[OnClass_train_obj.co2i[tp],:]*-1)[:topk]
				our_marker_id = [g2i2[i2g1[gi]] for gi in our_marker if i2g1[gi] in g2i2]
			Y_true = label_mat2[:,i]
			Y_pred_our = np.sum(feature2[:, our_marker_id], axis=1)
			our_auc = roc_auc_score(Y_true, Y_pred_our)
			if tp in tp2genes_base and len([g2i2[g] for g in tp2genes_base[tp] if g in g2i2])!=0:
				base_marker = tp2genes_base[tp]
				base_marker_id = [g2i2[g] for g in base_marker if g in g2i2]
				if len(base_marker_id) == 0:
					continue
				Y_pred_base = np.sum(feature2[:, base_marker_id], axis=1)
				base_auc = roc_auc_score(Y_true, Y_pred_base)#roc_auc_score(Y_true, Y_pred_base)
				has_truth_our_auc.append(our_auc)
				has_truth_tp_base_auc.append(base_auc)
			else:
				no_truth_our_auc.append(our_auc)
		pv = stats.ttest_rel(has_truth_our_auc, has_truth_tp_base_auc)[1] / 2.
		print ('%f %s %s %f seen(our,base,length,pv):%f %f %d %d %e' % (thres, dname1,dname2,np.mean(no_truth_our_auc),
		np.mean(has_truth_our_auc), np.mean(has_truth_tp_base_auc), len(has_truth_tp_base_auc), np.median(nmarkers), pv))
		has_truth_our_auc_mat[dname1][dname2] = has_truth_our_auc
		has_truth_base_auc_mat[dname1][dname2] = has_truth_tp_base_auc
		no_truth_our_auc_mat[dname1][dname2] = no_truth_our_auc

'''
ndname = len(dnames)
heat_mat = np.zeros((ndname,ndname))
group_l = []
for i,dname1 in enumerate(dnames):
	group_l.append(dname2keyword[dname1])
	for j,dname2 in enumerate(dnames):
		if dname1 == dname2:
			continue
		heat_mat[i,j] = np.mean(no_truth_our_auc_mat[dname1][dname2])
plot_heatmap_cross_dataset(heat_mat, group_l, file_name = fig_dir + '.pdf', ylabel = 'AUROC', title='AUROC')
'''
