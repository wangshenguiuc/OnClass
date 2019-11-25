import sys
from scanorama import *
import numpy as np
import os
from scipy import sparse
import sys
sys.path.append('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/OnClass/OnClass/')
#from OnClass.utils import *
#from OnClass.OnClassModel import OnClassModel
#from OnClass.other_datasets_utils import my_assemble, data_names_all, load_names
from utils import *
from OnClassModel import OnClassModel
from other_datasets_utils import my_assemble, data_names_all, load_names

OUTPUT_DIR = '../../OnClass_data/26-datasets/'
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)


OnClassModel = OnClassModel()
tp2emb, tp2i, i2tp = OnClassModel.EmbedCellTypes(dim=500,cell_type_network_file='../../../OnClass_data/cell_ontology/cl.ontology', use_pretrain='../../../OnClass_data/pretrain/tp2emb_500')
print (tp2emb)

data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-facs_cell_ontology.h5ad'
train_X, train_genes, train_Y = read_data(feature_file=data_file, tp2i = tp2i, AnnData_label='cell_ontology_class_reannotated')

print (train_Y)

#OnClassModel.train(train_X, train_Y, tp2emb, train_genes, nhidden=[2], max_iter=1, use_pretrain = None, save_model =  '../../../OnClass_data/pretrain/BilinearNN')
#test_label = OnClassModel.predict(train_X, train_genes)
#print (np.shape(test_label))
#print (test_label)
#print ('pretrain finished')


OnClassModel.train(train_X, train_Y, tp2emb, train_genes, nhidden=[500], log_transform = True, use_pretrain = None, save_model =  '../../../OnClass_data/pretrain/BilinearNN_500')
test_label = OnClassModel.predict(train_X, train_genes)
print (test_label)

cell2label = test_label[:,:5]
labels = [i2tp[i] for i in range(5)]
test_label = OnClassModel.predict_impute(cell2label, labels, tp2i, tp2emb)
print (test_label)
sdd
