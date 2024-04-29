.. kolmogorov-arnold-network documentation master file, created by
   sphinx-quickstart on Sun Apr 21 12:57:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Kolmogorov Aarnold Network (KAN) documentation!
==========================================================

.. image:: kan_plot.png

This documentation is for the paper "KAN: Kolmogorov-Arnold Networks" and the github repo can be found here.
Kolmogorov-Arnold Networks, inspired by the Kolmogorov-Arnold representation theorem, are promising alternatives
of Multi-Layer Preceptrons (MLPs). KANs have activation functions on edges, whereas MLPs have activation functions on nodes.
This simple change makes KAN better than MLPs in terms of both accuracy and interpretability.

Get the latest news at `CNN`_. (Add links to paper and github repo)

.. _CNN: http://cnn.com/

Installation
------------

Installation via github
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   git clone https://github.com/KindXiaoming/pykan.git
   cd pykan
   pip install -e .
   # pip install -r requirements.txt # install requirements


Installation via PyPI (soon)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   pip install pykan
   

Get started
-----------

* Quickstart: :ref:`hello-kan`
* KANs in Action: :ref:`api-demo`, :ref:`examples`
* API (advanced): :ref:`api`.



.. toctree::
   :maxdepth: 1
   :caption: Contents:

   intro.rst
   modules.rst
   demos.rst
   examples.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
