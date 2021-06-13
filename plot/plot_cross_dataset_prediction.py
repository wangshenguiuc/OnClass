import sys
import numpy as np
import os
from utils import make_folder, read_ontology_file, read_data, read_data_file, evaluate, MapLabel2CL
from plots import *
from OnClass.OnClassModel import OnClassModel
from config import ontology_data_dir, scrna_data_dir, result_dir, figure_dir, intermediate_dir

if len(sys.argv) <= 2:
	pid = 0
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

figure_dir = make_folder(figure_dir+'/Crossdatasets/')

#dnames = ['muris_droplet','microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy']
result_dir = intermediate_dir+ '/Cross_dataset/'
metrics = ['AUROC(seen)','AUPRC(seen)','AUROC','AUPRC','AUROC(unseen)', 'AUPRC(unseen)','Accuracy@3','Accuracy@5']

pct = -1
for dname1 in dnames:
	cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file(dname1, ontology_data_dir)
	OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
	feature_file, filter_key, label_key, label_file, gene_file = read_data_file(dname1, scrna_data_dir)
	if feature_file.endswith('.pkl'):
		feature1, label1, genes1 = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
	elif feature_file.endswith('.h5ad'):
		feature1, genes1, label1, _, _ = read_data(feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
		exclude_non_leaf_ontology = True, tissue_key = 'tissue', filter_key = filter_key, AnnData_label_key=label_key,
		nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp)
	for dname2 in dnames:
		if dname1 == dname2:
			continue
		feature_file, filter_key, label_key, label_file, gene_file = read_data_file(dname2, scrna_data_dir)
		if feature_file.endswith('.pkl'):
			feature2, label2, genes2 = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
		elif feature_file.endswith('.h5ad'):
			feature2, genes2, label2, _, _ = read_data(feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
			exclude_non_leaf_ontology = True, tissue_key = 'tissue', filter_key = filter_key, AnnData_label_key=label_key,
			nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp)

			train_Y_str = label1
			test_Y_str = label2
			OnClass_train_obj.EmbedCellTypes(train_Y_str)
			nseen = OnClass_train_obj.nseen
			onto_net = OnClass_train_obj.ontology_dict
			unseen_l_str = OnClass_train_obj.unseen_co

			unseen_l = MapLabel2CL(unseen_l_str, OnClass_train_obj.co2i)
			train_Y = MapLabel2CL(train_Y_str, OnClass_train_obj.co2i)
			test_Y = MapLabel2CL(test_Y_str, OnClass_train_obj.co2i)

			test_Y_ind = np.sort(np.array(list(set(test_Y) |  set(train_Y))))
			print ('seen: %d, ntrainY: %d, ntestY: %d, unseen: %d' % (nseen, len(np.unique(train_Y)), len(np.unique(test_Y)), len(set(test_Y) - set(train_Y))))
			pred_Y_all = np.load(result_dir+ dname1+'.'+dname2+'pred_Y_all.released.npy')
			res = evaluate(pred_Y_all, test_Y, unseen_l, nseen, metrics = metrics
			, Y_ind = test_Y_ind, Y_net = onto_net, write_screen = True, prefix = dname1+'.'+dname2+'.'+str(len(set(test_Y) - set(train_Y)))+'.'+str(len(set(test_Y))))


ndame = len(dnames)
mat2i = {}
for metric in metrics:
	mat2i[metric] = np.empty((ndame, ndame))
	mat2i[metric][:] = np.NaN
mat2i['Ratio of unseen cell types'] = np.empty((ndame, ndame))
mat2i['Ratio of unseen cell types'][:] = np.NaN
dname2i = {}
i2dname = {}
for i in range(ndame):
	dname2i[dnames[i]] = i
	i2dname[i] = dnames[i]
print (dname2i)
fin = open(result_file)
for line in fin:
	w = line.strip().split('\t')
	d1,d2,nunseen,ntest = w[0].split('.')
	nunseen_ratio = int(nunseen) * 1. / int(ntest)
	d1i = dname2i[d1]
	d2i = dname2i[d2]
	mat2i['Ratio of unseen cell types'][d1i,d2i] = nunseen_ratio
	for i, metric in enumerate(metrics):
		mat2i[metric][d1i,d2i] = float(w[i+1])

methods = []
for dname in dnames:
	methods.append(dname2keyword[dname])
metrics = list(mat2i.keys())
metrics.reverse()
for metric in metrics:
	heat_mat = mat2i[metric]
	print (fig_dir + metric + '.pdf')
	plot_heatmap_cross_dataset(heat_mat, methods=methods, file_name =fig_dir + metric.replace('\n',' ') + '_resize0.pdf', title=metric, ylabel=metric)
