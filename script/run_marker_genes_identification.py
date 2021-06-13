import sys
from scipy import stats, sparse
import numpy as np
import os
from collections import Counter
from OnClass.OnClassModel import OnClassModel
from sklearn.metrics import roc_auc_score
from utils import read_ontology_file, read_data, read_data_file, read_data, parse_pkl, SplitTrainTest, read_type2genes, corr2_coeff
from config import ontology_data_dir, scrna_data_dir, result_dir, intermediate_dir
from plots import dname2keyword

dnames = ['muris_facs','muris_droplet','microcebusBernard','microcebusStumpy','microcebusMartine','microcebusAntoine']

marker_gene_file = scrna_data_dir + 'gene_marker_expert_curated.txt'
score = {}
for dname in dnames:
	score[dname] = {}
	for method in ['Seen cell types', 'Unseen cell types']:
		score[dname][method] = []

thres = 0.4
unseen_ratio = 0.5
for dname in dnames:
	pred_score_file = intermediate_dir + '/Marker_genes/' +dname + 'pred_Y_all.released.npy'
	cor_file = intermediate_dir + '/Marker_genes/' +dname+ 'cor.released.npy'
	cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file(dname, ontology_data_dir)
	OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
	feature_file, filter_key, label_key, label_file, gene_file = read_data_file(dname, scrna_data_dir)
	if feature_file.endswith('.pkl'):
		feature, label, genes = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
	elif feature_file.endswith('.h5ad'):
		feature, genes, label, _, _ = read_data(feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
		exclude_non_leaf_ontology = True, tissue_key = 'tissue', filter_key = filter_key, AnnData_label_key=label_key,
		nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp)
	train_feature, train_label, test_feature, test_label = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = 0)
	train_feature = np.vstack((train_feature, test_feature))
	train_label = np.concatenate((train_label, test_label))
	OnClass_train_obj.EmbedCellTypes(train_label)
	nseen = OnClass_train_obj.nseen
	train_genes = genes
	test_genes = genes
	train_feature, train_genes = OnClass_train_obj.ProcessTrainFeature(train_feature, train_label, train_genes)
	pred_Y_all = np.load(pred_score_file)

	g2i = {g : i for i,g in enumerate(genes)}
	tp2genes_base = read_type2genes(g2i, marker_gene_file, cl_obo_file)
	terms = [OnClass_train_obj.i2co[i] for i in range(len(OnClass_train_obj.i2co))]

	if not os.path.isfile(cor_file):
		pred_Y_all = np.load(pred_score_file)
		cor = corr2_coeff(pred_Y_all[:,:].T, train_feature[:,:].T)
		cor = np.nan_to_num(cor)
		np.save(cor_file, cor)
	else:
		cor = np.load(cor_file)

	sub_cor = np.where(np.abs(cor)>thres, 1, 0)
	seen_aucs = []
	unseen_aucs = []
	for tp in tp2genes_base:
		label = [g in tp2genes_base[tp] for g in genes]
		pred = cor[OnClass_train_obj.co2i[tp],:]
		auc = roc_auc_score(label, pred)
		if OnClass_train_obj.co2i[tp]<nseen:
			seen_aucs.append(auc)
		else:
			unseen_aucs.append(auc)
	print (np.mean(unseen_aucs), np.mean(seen_aucs))
	score[dname]['Seen cell types'] = seen_aucs
	score[dname]['Unseen cell types'] = unseen_aucs
#np.savez(npz_file, score = score)

metrics = ['Unseen cell types','Seen cell types']
ndname = len(dnames)
nmetric = len(metrics)
mean = np.zeros((ndname, nmetric))
error = np.zeros((ndname, nmetric))
group_l = []
for i in range(ndname):
	group_l.append(dname2keyword[dnames[i]])
	for j in range(nmetric):
		mean[i,j] = np.mean(score[dnames[i]][metrics[j]])
		error[i,j] = np.std(score[dnames[i]][metrics[j]]) / np.sqrt(len(score[dnames[i]][metrics[j]]))
print (group_l)
print (mean,error)
#plot_marker_comparison_baselines_bar(mean, error, group_l = group_l, method_l = metrics,  output_file = fig_dir + 'barplot_rename.pdf', ylabel='AUROC',lab2col={'Seen cell types':'g','Unseen cell types':'y'})
