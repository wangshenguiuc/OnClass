|PyPI| |Docs|

.. |PyPI| image:: https://img.shields.io/pypi/v/scanpy.svg
   :target: https://pypi.org/project/OnClass/
.. |Docs| image:: https://readthedocs.com/projects/icb-scanpy/badge/?version=latest
   :target: https://onclass.readthedocs.io/en/latest/introduction.html

How to use OnClass.
=========

They key idea of OnClass is using the Cell ontology to help cell type prediction. OnClass is the first tool that can classify cells into a new cell type that doesn't exist in the training data. To achieve this, the key components are:

1) **Embedding Cell ontology**
~~~~~~~~~

OnClass can use any form of cell type similarity, including hierarchical structure, directed acyclic graph, weighted network, unweighted network or a just a few lines of cell type similarity. An example can be found `here <https://github.com/wangshenguiuc/OnClass/tree/master/img/cell_type_similarity_example.txt>`__ .

In our paper, we use the hierarchical ontology structure from the `Cell Ontology <http://www.obofoundry.org/ontology/cl.html>`__. However, OnClass is very flexible and any forms of prior cell type similarity (i.e., label similarity) can be used. It could be a *weighted network*, where nodes are cell types and edge weights are cell type similarity (preferred to be normalized between 0 and 1, but not required). It could be a *unweighted network* where all edge weights are set to 0, which is the case for the Cell Ontology. It could be just *a few lines of cell type similarity*, where each line is a tab-spitted three Column file in the form of "CL:000001\tCL:000002\t0.8", representing cell type 1, cell type 2, and similarity. By default, all edges are undirected since they are similarity. But if you have directed similarities, please email us and we are happy to modify the code to support it.

We provide a precomputed cell ontology embeddings based on the `Cell Ontology <http://www.obofoundry.org/ontology/cl.html>`__. Please check the API section **Embedding Cell Ontology** for how to read and embed the Cell Ontology.

2) **Read the gene expression data**
~~~~~~~~~

The gene expression data is used as training data. It includes a cell by gene matrix and a label for each cell.

The label for each cell should be a cell ontology ID or cell ontology terms. If you training labels are not mapped to cell ontology ID, please use our natural language processing tool to map them to existing cell ontology terms. Our tool use a char-level LSTM siamese network and achieve 93.7% accuracy in mapping synonym of cell ontology terms and can deal with composition, misspelling, and abbreviations. For more information about this natural language processing tool, please see section Char-level LSTM for cell type term mapping.

OnClass is a highly flexible tool that can support a variety formats of gene expression data inputs, including scipy sparse matrix, numpy 2D array, tsv format of matrix, tsv format of "gene_name\tcell_name\tcount", scanpy AnnData, and sparse matrix format used by 10X Genomics. Some of these input implementation are adopted from Scanorama project.

3) **Predict cell types for new cells**
~~~~~~~~~


OnClass can predict cell types through any of the following four approaches:

a) Use the pretrained marker genes. OnClass found 20 marker genes for each cell type in the Cell Ontology. Marker genes are precomputed based on all FACS cells from Tabula Muris Senis.

b) Use the predicted score of any existing models and propagate to all cell types in the Cell Ontology.

c) Use the pretrained Blinear Neural Network model. This model is trained on all FACS cells from Tabula Muris Senis and all cell type terms from the Cell Ontology.

d) Train from new data. Train a new model on a new gene expression data and predicted on the cell types for a set of new cells.

The time and memory complexity of a) and b) are very small (e.g., less than 1 minute on any personal laptop). The time and memory complexity of c) is moderate (less than 1 mintues for 50K cells on GPU and less than 10 minutes for 50K on CPU.) The time and memory complexity of d) is large (about 1 hour for 50k cells on GPU and 4 hour for 50k cells on CPU). However, the expected performance is: d) > c) > b) > a). So please choose one of them according to your application.


Flowchart
~~~~~~~~~
Here is the flowchart of OnClass

.. image:: img/flowchart.png

OnClass is a joint work by `Altman lab <https://helix.stanford.edu/>`__ at stanford and `czbiohub <https://www.czbiohub.org/>`__.

For questions about the software, please contact `Sheng Wang <http://web.stanford.edu/~swang91/>`__ at swang91@stanford.edu.
