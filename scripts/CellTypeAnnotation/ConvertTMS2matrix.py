import sys
from scanorama import *
import numpy as np
import os
from scipy import sparse
import sys
from anndata import AnnData
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

data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-facs.h5ad'
output_data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-facs_cell_ontology.h5ad'
train_X, train_Y_str, genes_list = read_data_TMS(filename=data_file, return_genes=True, cell_type_name_file='../../../OnClass_data/cell_ontology/cl.obo')
print (np.shape(train_X))
print (np.shape(train_Y_str))
print (genes_list.keys())
anndata = AnnData(train_X)
anndata.obs['cell_ontology_class_reannotated'] = train_Y_str
anndata.var.index = genes_list[list(genes_list.keys())[0]]
anndata.write(output_data_file)
