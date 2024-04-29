Demo 5: Initialization Hyperparamters
=====================================

Initialization is the first step to gaurantee good training. Each
activation function is initialized to be
:math:`\phi(x)={\rm scale\_base}*b(x) + {\rm scale\_sp}*{\rm spline}(x)`.
1. :math:`b(x)` is the base function, default: silu, can be set with
:math:`{\rm base\_fun}`

2. :math:`{\rm scale\_base}=1/\sqrt{n_{\rm in}}+e` where :math:`e` is
   drawn from
   :math:`U[-{\rm noise\_scale\_base},{\rm noise\_scale\_base}]`.
   Default: :math:`{\rm noise\_scale\_base}=0.1`.

3. :math:`{\rm scale\_sp}=1`

4. :math:`{\rm spline}(x)` is initialized by drawing coefficients
   independently from :math:`N(0,({\rm noise\_scale}/G)^2)` where
   :math:`G` is the grid number. Default:
   :math:`{\rm noise\_scale}=0.1`.

Default setup

.. code:: ipython3

    from kan import KAN, create_dataset
    import torch
    
    model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
    x = torch.normal(0,1,size=(100,2))
    model(x) # forward is needed to collect activations for plotting
    model.plot()



.. image:: API_5_initialization_hyperparameter_files/API_5_initialization_hyperparameter_3_0.png


Case 1: Initialize all activation functions to be exactly linear. We
need to set noise_scale_base = 0., base_fun = identity, noise_scale = 0.

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0, noise_scale_base = 0., base_fun = lambda x: x, noise_scale = 0.)
    x = torch.normal(0,1,size=(100,2))
    model(x) # forward is needed to collect activations for plotting
    model.plot()



.. image:: API_5_initialization_hyperparameter_files/API_5_initialization_hyperparameter_5_0.png


Case 2: Noisy spline initialization (not recommended, just for
illustration). Set noise_scale to be a large number.

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0, noise_scale=1.)
    x = torch.normal(0,1,size=(100,2))
    model(x) # forward is needed to collect activations for plotting
    model.plot()



.. image:: API_5_initialization_hyperparameter_files/API_5_initialization_hyperparameter_7_0.png


.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0, noise_scale=10.)
    x = torch.normal(0,1,size=(100,2))
    model(x) # forward is needed to collect activations for plotting
    model.plot()



.. image:: API_5_initialization_hyperparameter_files/API_5_initialization_hyperparameter_8_0.png


Case 3: Break Symmetry. When noise_scale_base is too small, nodes are
almost degenerate. Sometimes we want to break such symmetry even at
initialization. For an example, please see the PDE demo, where a
non-zero noise_scale_base is important for achieving fast convergence.

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0, noise_scale_base=0.0)
    x = torch.normal(0,1,size=(100,2))
    model(x) # forward is needed to collect activations for plotting
    model.plot()



.. image:: API_5_initialization_hyperparameter_files/API_5_initialization_hyperparameter_10_0.png


.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0, noise_scale_base=1.0)
    x = torch.normal(0,1,size=(100,2))
    model(x) # forward is needed to collect activations for plotting
    model.plot()



.. image:: API_5_initialization_hyperparameter_files/API_5_initialization_hyperparameter_11_0.png


