Quick start
=========
Here, we provide an introduction of how to use OnClass

Cell type annotation
----------------

A example script for cell type annotation is at our `GitHub <https://www.biorxiv.org/content/10.1101/810234v1>`__

Import OnClass as::

    import OnClass
	


Read the single cell data as::
    
	data_file = '../../OnClass_data/raw_data/tabula-muris-senis-facs'
	## read data
	X, Y = read_data(filename=data_file)
	
where `X` is a sample by gene gene expression matrix, `Y` is a label vector for each sample. The labels in `Y` should use the Cell Ontology Id (e.g., CL:1000398). The data (e.g., tabula muris raw gene expression matrix, the Cell Ontology obo file) can be downloaded from FigShare. Please change the path of data_file in the script.

