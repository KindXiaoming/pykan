API 4: Initialization
=====================

Initialization is the first step to gaurantee good training. Each
activation function is initialized to be
:math:`\phi(x)={\rm scale\_base}*b(x) + {\rm scale\_sp}*{\rm spline}(x)`.
1. :math:`b(x)` is the base function, default: ‘silu’, can be set with
:math:`{\rm base\_fun}`

2. scale_sp sample from N(0, noise_scale^2)

3. scale_base sampled from N(scale_base_mu, scale_base_sigma^2)

4. sparse initialization: if sparse_init = True, most scale_base and
   scale_sp will be set to zero

Default setup

.. code:: ipython3

    from kan import KAN, create_dataset
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model = KAN(width=[2,5,1], grid=5, k=3, seed=0, device=device)
    x = torch.normal(0,1,size=(100,2)).to(device)
    model(x) # forward is needed to collect activations for plotting
    model.plot()


.. parsed-literal::

    cuda
    checkpoint directory created: ./model
    saving model version 0.0



.. image:: API_4_initialization_files/API_4_initialization_3_1.png


Case 1: Initialize all activation functions to be exactly linear. We
need to set noise_scale_base = 0., base_fun = identity, noise_scale = 0.

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0, base_fun = 'identity', device=device)
    x = torch.normal(0,1,size=(100,2)).to(device)
    model(x) # forward is needed to collect activations for plotting
    model.plot()


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0



.. image:: API_4_initialization_files/API_4_initialization_5_1.png


Case 2: Noisy spline initialization (not recommended, just for
illustration). Set noise_scale to be a large number.

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0, noise_scale=0.3, device=device)
    x = torch.normal(0,1,size=(100,2)).to(device)
    model(x) # forward is needed to collect activations for plotting
    model.plot()


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0



.. image:: API_4_initialization_files/API_4_initialization_7_1.png


.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0, noise_scale=10., device=device)
    x = torch.normal(0,1,size=(100,2)).to(device)
    model(x) # forward is needed to collect activations for plotting
    model.plot()


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0



.. image:: API_4_initialization_files/API_4_initialization_8_1.png


Case 3: scale_base_mu and scale_base_sigma

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0, scale_base_mu=5, scale_base_sigma=0, device=device)
    x = torch.normal(0,1,size=(100,2)).to(device)
    model(x) # forward is needed to collect activations for plotting
    model.plot()


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0



.. image:: API_4_initialization_files/API_4_initialization_10_1.png


.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0, sparse_init=True, device=device)
    x = torch.normal(0,1,size=(100,2)).to(device)
    model(x) # forward is needed to collect activations for plotting
    model.plot()


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0



.. image:: API_4_initialization_files/API_4_initialization_11_1.png


