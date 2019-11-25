|PyPI| |Docs|

.. |PyPI| image:: https://img.shields.io/pypi/v/scanpy.svg
   :target: https://pypi.org/project/OnClass/
.. |Docs| image:: https://readthedocs.com/projects/icb-scanpy/badge/?version=latest
   :target: https://onclass.readthedocs.io/en/latest/introduction.html

Introduction
=========
OnClass is a python package for single-cell cell type annotation. It uses the Cell Ontology to capture the cell type similarity. These similarities enable OnClass to annotate cell types that are never seen in the training data.

OnClass package is still under development. A preprint of OnClass paper is on `bioRxiv <https://www.biorxiv.org/content/10.1101/810234v1>`__
All datasets used in OnClass can be found in `figshare <https://figshare.com/projects/OnClass/70637>`__.
Currently, OnClass supports

1) annotate cell type


2) integrating different single-cell datasets based on the Cell Ontology


3) marker genes identification


How to use OnClass.
~~~~~~~~~
They key idea of OnClass is using the Cell ontology to help cell type prediction. OnClass is the first tool that can classify cells into a new cell type that doesn't exist in the training data. To achieve this, the key components are:

1) Embedding Cell ontology
OnClass can use any form of cell type similarity. In our paper, we use the **hierarchical ontology structure** (it is actually directed acyclic graph) from the `Cell Ontology <http://www.obofoundry.org/ontology/cl.html>`__. However, OnClass is very flexible and any forms of prior cell type similarity (i.e., label similarity) can be used. It could be a **weighted network**, where nodes are cell types and edge weights are cell type similarity (preferred to be normalized between 0 and 1, but not required). It could be a **unweighted network** where all edge weights are set to 0, which is the case for the Cell Ontology. It could be just **a few lines of cell type similarity**, where each line is a tab-spitted three Column file in the form of "CL:000001\tCL:000002\t0.8", representing cell type 1, cell type 2, and similarity. An example can be found `here <https://github.com/wangshenguiuc/OnClass/tree/master/img/cell_type_similarity_example.txt>`__ . By default, all edges are undirected since they are similarity. But if you have directed similarities, please email us and we are happy to modify the code to support it.

We provide a precomputed cell ontology embeddings based on the `Cell Ontology <http://www.obofoundry.org/ontology/cl.html>`__

Please check the API section **Embedding Cell Ontology** for how to read and embed the Cell Ontology.


Here is the flowchart of OnClass

.. image:: img/flowchart.png

OnClass is a joint work by `Altman lab <https://helix.stanford.edu/>`__ at stanford and `czbiohub <https://www.czbiohub.org/>`__.

For questions about the software, please contact `Sheng Wang <http://web.stanford.edu/~swang91/>`__ at swang91@stanford.edu.
