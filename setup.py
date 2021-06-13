from setuptools import setup, find_packages

setup(
    name = "OnClass",
    version = "0.97",
    keywords = ("pip", "single cell", "OnClass"),
    description = "Single Cell Annotation based on the Cell Ontology",
    long_description = "Unifying single-cell annotations based on the Cell Ontology",
    license = "MIT Licence",

    url = "https://github.com/wangshenguiuc/OnClass",
    author = "Sheng Wang",
    author_email = "swang91@uw.edu",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires=[
		'sentence-transformers',
		'anndata>=0.6.4',
        'fbpca>=1.0',
        'umap-learn>=0.3.10',
        'matplotlib>=2.0.2',
        'numpy>=1.16.4',
        'scipy>=1.3.1',
        'scikit-learn>=0.21.3',
		'scanorama>=1.7.1'
    ]
)
