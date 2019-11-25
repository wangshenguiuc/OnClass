Installing OnClass
=========================

Use OnClass with GPU.
OnClass can be used to predict cell types based on the expression through any of the following five approaches
1) Use the pretrained marker genes. OnClass found 20 marker genes for each cell type in the Cell Ontology. Marker genes are precomputed based on all FACS cells from Tabula Muris Senis.
2) Use the predicted score of any existing models and propogate to all cell types in the Cell Ontology.
3) Use the pretrained Blinear Neural Network model. This model is trained on all FACS cells from Tabula Muris Senis and all cell type terms from the Cell Ontology.
4) Train from new data. Train a new model on a new gene expression data and predicted on the cell types for a set of new cells.

The



PyPI
~~~~~~~~~
OnClass is available through the `Python Package Index`_ and thus can be installed
using pip. To install OnClass using pip, run:

.. code:: bash

    pip install OnClass

.. _Python Package Index: https://pypi.python.org/pypi



Development Version
~~~~~~~~~
The lastest verion of OnClass is on `GitHub
<https://github.com/wangshenguiuc/OnClass/>`__

.. code:: bash

    git clone https://github.com/wangshenguiuc/OnClass.git
