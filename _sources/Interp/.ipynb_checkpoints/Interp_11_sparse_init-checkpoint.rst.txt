Interpretability 11: sparse initialization
==========================================

.. code:: ipython3

    from kan import *
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model = KAN([5,5,5,1], sparse_init=False, device=device)
    x = torch.rand(100,5).to(device)
    model.get_act(x)
    model.plot()


.. parsed-literal::

    cuda
    checkpoint directory created: ./model
    saving model version 0.0



.. image:: Interp_11_sparse_init_files/Interp_11_sparse_init_1_1.png


.. code:: ipython3

    from kan import *
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model = KAN([5,5,5,1], sparse_init=True, device=device)
    x = torch.rand(100,5).to(device)
    model.get_act(x)
    model.plot()


.. parsed-literal::

    cuda
    checkpoint directory created: ./model
    saving model version 0.0



.. image:: Interp_11_sparse_init_files/Interp_11_sparse_init_2_1.png


