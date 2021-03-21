import sys
from scipy import stats, sparse
import numpy as np
import os
from collections import Counter

repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/OnClass/'
scrna_data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/'
sys.path.append(repo_dir)
os.chdir(repo_dir)
from utils import *
from package.OnClassModel import OnClassModel


if len(sys.argv) <= 2:
	pid = 0
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

output_dir = make_folder(result_dir+'/Crossdatasets/')
ontology_dir = make_folder(result_dir+'/Ontology/CellOntology/')
save_model_dir = make_folder(result_dir+'/PretrainedModel/')

dnames = ['muris_droplet','microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy','muris_facs']
#dnames = ['muris_facs','muris_droplet']
pct = -1
dname1 = dnames[3]
dname2 = dnames[3]
cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file(dname1, ontology_data_dir)
OnClass = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)
feature_file, filter_key, label_key, label_file, gene_file = read_data_file(dname1, scrna_data_dir)
if feature_file.endswith('.pkl'):
	feature1, label1, genes1 = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
elif feature_file.endswith('.h5ad'):
	feature1, genes1, label1, _, _ = read_data(feature_file, cell_ontology_ids = OnClass.cell_ontology_ids,
	exclude_non_leaf_ontology = True, tissue_key = 'tissue', filter_key = filter_key, AnnData_label_key=label_key,
	nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass.co2vec_nlp)

feature_file, filter_key, label_key, label_file, gene_file = read_data_file(dname2, scrna_data_dir)
if feature_file.endswith('.pkl'):
	feature2, label2, genes2 = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
elif feature_file.endswith('.h5ad'):
	feature2, genes2, label2, _, _ = read_data(feature_file, cell_ontology_ids = OnClass.cell_ontology_ids,
	exclude_non_leaf_ontology = True, tissue_key = 'tissue', filter_key = filter_key, AnnData_label_key=label_key,
	nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass.co2vec_nlp)


common_genes = np.array(list(set(genes1) & set(genes2)))
gid1 = find_gene_ind(genes1, common_genes)
gid2 = find_gene_ind(genes2, common_genes)

train_X = feature1[:, gid1]
test_X = feature2[:, gid2]
train_Y_str = label1
test_Y_str = label2

print (np.shape(feature1))

#folder = make_folder(output_dir +'/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio) + '/')
#save_model_dir = make_folder(result_dir+'/PretrainedModel/'+ dname + '/' + str(iter) + '/' + str(unseen_ratio) + '/')
#output_prefix = folder + pname
co2emb, co2i, i2co, ontology_mat = OnClass.EmbedCellTypes(train_Y_str)
train_X, test_X = OnClass.align_train_test_expression(train_X, test_X)
print ('train start')
#stage 1 seen prediction
OnClass.BuildModel(OnClass.co2emb, ngene = np.shape(train_X)[1])
OnClass.Train(train_X, train_Y_str, max_iter=1)
pred_Y_seen, pred_Y_all, pred_label = OnClass.Predict(test_X, use_normalize=True)
print (Counter(pred_label))
pred_Y_seen, pred_Y_all, pred_label = OnClass.Predict(test_X, use_normalize=False)
print (Counter(pred_label))
