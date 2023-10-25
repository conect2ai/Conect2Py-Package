from setuptools import setup, find_packages
from os.path import abspath, dirname, join

README_MD = open(join(dirname(abspath(__file__)), "README.md")).read()

setup(
    # Name 
    name="conect2py",

    # Version
    version="0.1.1",

    # Packages
    packages=find_packages(include=['conect2py', 'conect2py.*']),

    # Description
    description="A python library for data compression using TAC (Tiny Anomaly Compression)",

    # The content that will be shown for the project page.
    long_description=README_MD,
    long_description_content_type="text/markdown",

    # The url field - Link to a git repository
    url="https://github.com/conect2ai/Conect2Py-Package",

    # The author name and email 
    author="Conect2ai",
    author_email="conect2ai@gmail.com",

    # Classifiers
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"
    ],

    # Keywords are tags that identify your project and help searching for it
    # This field is OPTIONAL
    keywords="TEDA, TAC, Annomaly Detection, Data Compression, IoT, Eccentricity",

    install_requires=['numpy>=1.26', 'pandas>=2.1', 'matplotlib>=3.8', 'seaborn>=0.13', 'ipython>=8.16', 'scikit-learn>=1.3']
)