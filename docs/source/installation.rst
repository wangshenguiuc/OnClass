Installing OnClass
=========================
OnClass can be substantially accelerated by using GPU (tensorflow). However, there is only required when you want to train your own model.



Install from Conda environment file (Recommended)
~~~~~~~~~
This is the recommended way to install OnClass as it can install the most updated version of OnClass

1) Only use CPU


.. code:: bash

	git clone https://github.com/wangshenguiuc/OnClass.git
	conda env create -f environment.yml --name env_name
	conda activate env_name

..

2) Use GPU



.. code:: bash

	git clone https://github.com/wangshenguiuc/OnClass.git
	conda env create -f environment_gpu.yml --name env_name
	conda activate env_name

..



PyPI
~~~~~~~~~
OnClass is available through the `Python Package Index`_ and thus can be installed
using pip. Please use Python3.6. To install OnClass using pip, run:

1) Only use CPU


.. code:: bash

	pip install OnClass
	pip install tensorflow==1.14

.. _Python Package Index: https://pypi.python.org/pypi

2) Use GPU


.. code:: bash

	pip install OnClass
	pip install tensorflow-gpu==1.14

.. _Python Package Index: https://pypi.python.org/pypi




Development Version
~~~~~~~~~
The lastest verion of OnClass is on `GitHub
<https://github.com/wangshenguiuc/OnClass/>`__

.. code:: bash

	git clone https://github.com/wangshenguiuc/OnClass.git
