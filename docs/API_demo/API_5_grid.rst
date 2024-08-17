API 5: Grid
===========

One important feature of KANs is that they embed splines to neural
networks. However, splines are only valid for approximating functions in
known bounded regions, while the range of activations in neural networks
may be changing over training. So we have to update grids properly
according to that. Let’s first take a look at how we parametrize
splines.

.. code:: ipython3

    from kan.spline import B_batch
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from kan.spline import extend_grid
    
    # consider a 1D example.
    # Suppose we have grid in [-1,1] with G intervals, spline order k
    G = 5
    k = 3
    grid = torch.linspace(-1,1,steps=G+1)[None,:]
    grid = extend_grid(grid, k_extend=k)
    
    # and we have sample range in [-1,1]
    x = torch.linspace(-1,1,steps=1001)[None,:]
    
    basis = B_batch(x, grid, k=k)
    
    for i in range(G+k):
        plt.plot(x[0].detach().numpy(), basis[0,:,i].detach().numpy())
        
    plt.legend(['B_{}(x)'.format(i) for i in np.arange(G+k)])
    plt.xlabel('x')
    plt.ylabel('B_i(x)')




.. parsed-literal::

    Text(0, 0.5, 'B_i(x)')




.. image:: API_5_grid_files/API_5_grid_2_1.png


There are :math:`G+k` B-spline basis. The function is a linear
combination of these bases

.. math:: {\rm spline}(x)=\sum_{i=0}^{G+k-1} c_i B_i(x).

\ We don’t need worry about the implementation since it’s already built
in KAN. But let’s check if KAN is indeed implementing this. We
initialize a [1,1] KAN, which is simply a 1D spline.

.. code:: ipython3

    from kan import KAN
    
    model = KAN(width=[1,1], grid=G, k=k)
    # obtain coefficients c_i
    model.act_fun[0].coef
    assert(model.act_fun[0].coef[0].shape[1] == G+k)
    
    # the model forward
    model_output = model(x[0][:,None])
    
    # spline output
    spline_output = torch.einsum('j,ij->i',model.act_fun[0].coef[0][0], basis[0])[:,None]
    
    torch.mean((model_output - spline_output)**2)


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0




.. parsed-literal::

    tensor(0.0099, grad_fn=<MeanBackward0>)



They are not the same, what’s happening? We want to remind that we model
the activation function to have two additive parts, a residual function
:math:`b`\ (x) plus the spline function, i.e.,

.. math:: \phi(x)={\rm scale\_base}*b(x)+{\rm scale\_sp}*{\rm spline}(x),

\ and by default :math:`b(x)={\rm silu}(x)=x/(1+e^{-x})`.

.. code:: ipython3

    # residual output
    residual_output = torch.nn.SiLU()(x[0][:,None])
    scale_base = model.act_fun[0].scale_base
    scale_sp = model.act_fun[0].scale_sp
    torch.mean((model_output - (scale_base * residual_output + scale_sp * spline_output))**2)




.. parsed-literal::

    tensor(0., grad_fn=<MeanBackward0>)



What if my grid does not match my data? For example, my grid is in
[-1,1], but my data is in [10,10] or [-0.5,0.5]. Use
update_grid_from_sample to adjust grids to samples. This grid update
applies to all splines in all layers.

.. code:: ipython3

    model = KAN(width=[1,1], grid=G, k=k)
    print(model.act_fun[0].grid) # by default, the grid is in [-1,1]
    x = torch.linspace(-10,10,steps = 1001)[:,None]
    model.update_grid_from_samples(x)
    print(model.act_fun[0].grid) # now the grid becomes in [-10,10]. We add a 0.01 margin in case x have zero variance


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0
    Parameter containing:
    tensor([[-2.2000, -1.8000, -1.4000, -1.0000, -0.6000, -0.2000,  0.2000,  0.6000,
              1.0000,  1.4000,  1.8000,  2.2000]])
    Parameter containing:
    tensor([[-22., -18., -14., -10.,  -6.,  -2.,   2.,   6.,  10.,  14.,  18.,  22.]])


.. code:: ipython3

    model = KAN(width=[1,1], grid=G, k=k)
    print(model.act_fun[0].grid) # by default, the grid is in [-1,1]
    x = torch.linspace(-0.5,0.5,steps = 1001)[:,None]
    model.update_grid_from_samples(x)
    print(model.act_fun[0].grid) # now the grid becomes in [-10,10]. We add a 0.01 margin in case x have zero variance


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0
    Parameter containing:
    tensor([[-2.2000, -1.8000, -1.4000, -1.0000, -0.6000, -0.2000,  0.2000,  0.6000,
              1.0000,  1.4000,  1.8000,  2.2000]])
    Parameter containing:
    tensor([[-1.1000, -0.9000, -0.7000, -0.5000, -0.3000, -0.1000,  0.1000,  0.3000,
              0.5000,  0.7000,  0.9000,  1.1000]])


Uniform grid or non-uniform? We consider two options: (1) uniform grid;
(2) adaptive grid (based on sample distribution) such that there are
(rougly) same number of samples in each interval. We provide a parameter
grid_eps to interpolate between these two regimes. grid_eps = 1 gives
(1), and grid_eps = 0 gives (0). By default we set grid_eps = 1 (uniform
grid). There could be other options but it is out of our scope here.

.. code:: ipython3

    # uniform grid
    model = KAN(width=[1,1], grid=G, k=k)
    print(model.act_fun[0].grid) # by default, the grid is in [-1,1]
    x = torch.normal(0,1,size=(1000,1))
    model.update_grid_from_samples(x)
    print(model.act_fun[0].grid) # now the grid becomes in [-10,10]. We add a 0.01 margin in case x have zero variance


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0
    Parameter containing:
    tensor([[-2.2000, -1.8000, -1.4000, -1.0000, -0.6000, -0.2000,  0.2000,  0.6000,
              1.0000,  1.4000,  1.8000,  2.2000]])
    Parameter containing:
    tensor([[-8.3431, -6.8772, -5.4114, -3.9455, -2.4797, -1.0138,  0.4520,  1.9179,
              3.3837,  4.8496,  6.3154,  7.7813]])


.. code:: ipython3

    # adaptive grid based on sample distribution
    model = KAN(width=[1,1], grid=G, k=k, grid_eps = 0.)
    print(model.act_fun[0].grid) # by default, the grid is in [-1,1]
    x = torch.normal(0,1,size=(1000,1))
    model.update_grid_from_samples(x)
    print(model.act_fun[0].grid) # now the grid becomes in [-10,10]. We add a 0.01 margin in case x have zero variance


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0
    Parameter containing:
    tensor([[-2.2000, -1.8000, -1.4000, -1.0000, -0.6000, -0.2000,  0.2000,  0.6000,
              1.0000,  1.4000,  1.8000,  2.2000]])
    Parameter containing:
    tensor([[-8.3431, -6.8772, -5.4114, -3.9455, -0.8148, -0.2487,  0.2936,  0.8768,
              3.3837,  4.8496,  6.3154,  7.7813]])


