from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name='tac',
    packages = ['tac'],
    version='0.1.0',
    description='A python library for data compression using TAC (Tiny Anomaly Compression)',
    author='Conect2Ai',
    license='3-clause BSD',
    install_requires=['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'ipython'],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: 3-clause BSD",
    ],
)
