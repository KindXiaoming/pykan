<img width="600" alt="kan_plot" src="https://github.com/KindXiaoming/pykan/assets/23551623/a2d2d225-b4d2-4c1e-823e-bc45c7ea96f9">

# Kolmogorov-Arnold Newtworks (KANs)

This the github repo for the paper "KAN: Kolmogorov-Arnold Networks" [link]. The documentation can be found here [link].

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


To install requirements:
```python
pip install -r requirements.txt
```

## Documentation
The documenation can be found here [].

## Tutorials

**Quickstart**

Get started with [hellokan.ipynb](./hellokan.ipynb) notebook

**More demos**
Jupyter Notebooks in ``docs/Examples`` and ``docs/API_demo`` are ready to play. You may also find these examples in documentation.


