from OnClass.utils import *
from OnClass.OnClassModel import OnClassModel
from OnClass.other_datasets_utils import my_assemble, data_names_all, load_names

OnClassModel = OnClassModel()
tp2emb, tp2i, i2tp = OnClassModel.EmbedCellTypes(dim=500,cell_type_network_file='../../../OnClass_data/cell_ontology/cl.ontology', use_pretrain='../../../OnClass_data/pretrain/tp2emb_500')
print ('compute cell type embedding finished')


data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-facs_cell_ontology.h5ad'
train_X, train_genes, train_Y = read_data(feature_file=data_file, tp2i = tp2i, AnnData_label='cell_ontology_class_reannotated')
print (np.shape(train_X))

test_data_file = '../../../OnClass_data/raw_data/tabula-muris-senis-droplet.h5ad'
test_X, test_genes, test_AnnData = read_data(feature_file=test_data_file, tp2i = tp2i, return_AnnData = True)
print (np.shape(test_X))

model_name = 'human'
model_path = '../../../OnClass_data/pretrain/' + model_name
#OnClassModel.train(train_X, train_Y, tp2emb, train_genes, nhidden=[500], max_iter=20, use_pretrain = None, save_model =  model_path) # this is the code to train your own model and save in '../../../OnClass_data/pretrain/BilinearNN'
OnClassModel.train(train_X, train_Y, tp2emb, train_genes, nhidden=[500], max_iter=20, use_pretrain = None, save_model =  model_path)
#OnClassModel.train(train_X, train_Y, tp2emb, train_genes, nhidden=[500], log_transform = True, use_pretrain = '../../../OnClass_data/pretrain/BilinearNN_50019', pretrain_expression='../../../OnClass_data/pretrain/BilinearNN_500')


test_label = OnClassModel.predict(test_X, test_genes,log_transform=True,correct_batch=False)
print (np.shape(test_label))

x = write_anndata_data(test_label, test_AnnData, i2tp, name_mapping_file='../../../OnClass_data/cell_ontology/cl.obo')#output_file is optional

print (x.obs['OnClass_annotation_ontology_ID'])
print (x.obs['OnClass_annotation_ontology_name'])
