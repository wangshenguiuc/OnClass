How to run OnClass
=========

To run OnClass, please first install OnClass, download datasets and then change file paths in `config.py <https://github.com/wangshenguiuc/OnClass/blob/master/run_OnClass_example.py>`__

We provide a `run_OnClass_example.py <https://github.com/wangshenguiuc/OnClass/blob/master/run_OnClass_example.py>`__ and Jupyter notebook as an example to run OnClass. This script trains an OnClass model on all cells from one Lemur dataset, saves that model to a model file, then use this model to classify cells from another Lemur dataset.

Run your own dataset for cell type annotation (`run_OnClass_example.py <https://github.com/wangshenguiuc/OnClass/blob/master/run_OnClass_example.py>`__)
----------------
You only need to modify line 9-13 in `run_OnClass_example.py <https://github.com/wangshenguiuc/OnClass/blob/master/run_OnClass_example.py>`__ by replacing train_file, test_file with your training and test file, and train_label and test_label with the cell ontology label key in your dataset.

One dataset cross-validation (Reproduce Figure 2 in OnClass paper)
----------------
`run_one_dataset_cross_validation.py <https://github.com/wangshenguiuc/OnClass/blob/master/script/run_one_dataset_cross_validation.py>`__ can be used to reproduce Figure 2 in our paper. All data are provided in figshare (please see Dataset and pretrained model)

Cross dataset prediction (Reproduce Figure 4 in OnClass paper)
----------------
`run_cross_dataset_prediction.py <https://github.com/wangshenguiuc/OnClass/blob/master/script/run_cross_dataset_prediction.py>`__ can be used to reproduce Figure 4 in our paper. All data are provided in figshare (please see Dataset and pretrained model)

Marker genes identification (Reproduce Figure 5 in OnClass paper)
----------------
Please first run `run_generate_pretrained_model.py <https://github.com/wangshenguiuc/OnClass/blob/master/script/run_generate_pretrained_model.py>`__ to generate the intermediate files for marker gene prediction. Then run `run_marker_genes_identification.py <https://github.com/wangshenguiuc/OnClass/blob/master/script/run_marker_genes_identification.py>`__ for marker gene identification (Figure 5c) and `run_marker_gene_based_prediction.py <https://github.com/wangshenguiuc/OnClass/blob/master/script/run_marker_gene_based_prediction.py>`__ for marker gene based prediction (Figure 5d,e,f, Extended Data Figure 7).
