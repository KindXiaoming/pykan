<img width="600" alt="kan_plot" src="https://github.com/KindXiaoming/pykan/assets/23551623/a2d2d225-b4d2-4c1e-823e-bc45c7ea96f9">

# Kolmogorov-Arnold Newtworks (KANs)

This the github repo for the paper "KAN: Kolmogorov-Arnold Networks" [link]. Find the [documentaion here](https://kindxiaoming.github.io/pykan/).

Kolmogorov-Arnold Networks (KANs) are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations just like MLPs: MLPs are based on the [universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem), while KANs are based on [Kolmogorov-Arnold representation theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem). KANs and MLPs are dual: KANs have activation functions on edges, while MLPs have activation functions on nodes. This simple change makes KANs better (sometimes much better!) than MLPs in terms of both model accuracy and interpretability. 

<img width="1163" alt="mlp_kan_compare" src="https://github.com/KindXiaoming/pykan/assets/23551623/695adc2d-0d0b-4e4b-bcff-db2c8070f841">

## Installation
There are two ways to install pykan, through pypi or github.

**Installation via github**

```python
git clone https://github.com/KindXiaoming/pykan.git
cd pykan
pip install -e .
```

**Installation via pypi (soon)**

```python
pip install pykan
```

Requirements

```python
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
setuptools==65.5.0
sympy==1.11.1
torch==2.2.2
tqdm==4.66.2
```

To install requirements:
```python
pip install -r requirements.txt
```

## Documentation
The documenation can be found [here](https://kindxiaoming.github.io/pykan/).

## Tutorials

**Quickstart**

Get started with [hellokan.ipynb](./hellokan.ipynb) notebook.

**More demos**

More Notebook tutorials can be found in [tutorials](tutorials).


