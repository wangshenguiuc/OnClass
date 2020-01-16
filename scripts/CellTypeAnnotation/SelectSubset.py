import sys
from scanorama import *
import numpy as np
import os
from scipy import sparse
import sys
from anndata import AnnData
#from OnClass.utils import *
#from OnClass.OnClassModel import OnClassModel
#from OnClass.other_datasets_utils import my_assemble, data_names_all, load_names
from utils import *
from OnClassModel import OnClassModel
from other_datasets_utils import my_assemble, data_names_all, load_names

OUTPUT_DIR = '../../OnClass_data/26-datasets/'
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)


test_data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-droplet.h5ad'
x = read_h5ad(test_data_file)

x_new = x[:1000,:]
#anndata.obs['cell_ontology_class_reannotated'] = train_Y_str
#anndata.var.index = genes_list[list(genes_list.keys())[0]]
output_data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-droplet_subset.h5ad'
x_new.write(output_data_file)
