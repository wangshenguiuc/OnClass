Install OnClass
=========================
OnClass can be substantially accelerated by using GPU (tensorflow 2.0). However, this is only required when you want to train your own model. OnClass can also be used with only CPU.

OnClass package only has three scripts: the `OnClass class file
<https://github.com/wangshenguiuc/OnClass/blob/master/OnClass/OnClassModel.py>`__,
`the deep learning model file
<https://github.com/wangshenguiuc/OnClass/blob/master/OnClass/BilinearNN.py>`__, and
`the utility functions file
<https://github.com/wangshenguiuc/OnClass/blob/master/OnClass/OnClass_utils.py>`__.

You can simply get these three files and put it in the right directory (see how to use `run_OnClass_example.py
<https://github.com/wangshenguiuc/OnClass/blob/master/run_OnClass_example.py>`__ in Tutorial). You can also install OnClass using pip as following:


~~~~~~~~~
OnClass is available through the `Python Package Index`_ and thus can be installed
using pip. Please use Python3.6. To install OnClass using pip, run:

1) Only use CPU


.. code:: bash

	pip install OnClass==1.2
	pip install tensorflow==2.0

.. _Python Package Index: https://pypi.python.org/pypi

2) Use GPU


.. code:: bash

	pip install OnClass==1.2
	pip install tensorflow-gpu==2.0

.. _Python Package Index: https://pypi.python.org/pypi




Development Version
~~~~~~~~~
The lastest verion of OnClass is on `GitHub
<https://github.com/wangshenguiuc/OnClass/>`__

.. code:: bash

	git clone https://github.com/wangshenguiuc/OnClass.git
