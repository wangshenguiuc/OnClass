from setuptools import setup, find_packages

setup(
    name = "OnClass",
    version = "0.0.1",
    keywords = ("pip", "single_cell", "OnClass", "swang"),
    description = "Single Cell Annotation",
    long_description = "Unifying single-cell annotations based on the Cell Ontology",
    license = "MIT Licence",

    url = "https://github.com/wangshenguiuc/OnClass",
    author = "swang",
    author_email = "swang91@stanford.edu",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["requests"]
)