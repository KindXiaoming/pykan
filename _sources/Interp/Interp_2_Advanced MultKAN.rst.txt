Interpretability 2: Advanced MultKAN
====================================

In the last tutorial, we introduced multiplications to KANs which makes
interpretation easier in the case when multiplications are needed.
Multiplication nodes by default takes in two numbers, but can take more
variables specified by the user. This is done through the mult_arity
argument (by default mult_arity=2).

.. code:: ipython3

    from kan import *
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model = KAN(width=[2,[3,2],1], device=device)
    x = torch.randn(100,2).to(device)
    model(x)
    model.plot()


.. parsed-literal::

    cuda
    checkpoint directory created: ./model
    saving model version 0.0



.. image:: Interp_2_Advanced%20MultKAN_files/Interp_2_Advanced%20MultKAN_2_1.png


mult_arity=3

.. code:: ipython3

    model = KAN(width=[2,[3,2],1], mult_arity=3, device=device)
    model(x)
    model.plot()


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0



.. image:: Interp_2_Advanced%20MultKAN_files/Interp_2_Advanced%20MultKAN_4_1.png


mult_arity=4

.. code:: ipython3

    model = KAN(width=[2,[3,2],1], mult_arity=4, device=device)
    model(x)
    model.plot()


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0



.. image:: Interp_2_Advanced%20MultKAN_files/Interp_2_Advanced%20MultKAN_6_1.png


You may want different multiplication nodes to take in different number
of variables. This is also possible: pass in mult_arity as a list of
lists, specifying the arities in each layer, including input layer,
hidden layer(s), and output layer.

In the following example, we have 0 multiplications in the input or in
the output layer, corresponding to empty lists. In the hidden layer, we
have two multiplications with arity = 2 and arity = 3, so we have the
list [2,3] in the middle.

.. code:: ipython3

    model = KAN(width=[2,[3,2],1], mult_arity=[[],[2,3],[]], device=device)
    model(x)
    model.plot()


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0



.. image:: Interp_2_Advanced%20MultKAN_files/Interp_2_Advanced%20MultKAN_9_1.png


Make a deeper network

.. code:: ipython3

    model = KAN(width=[2,[2,2],[1,3],[3,2],[1,1]], mult_arity=2, device=device)
    model(x)
    model.plot()


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0



.. image:: Interp_2_Advanced%20MultKAN_files/Interp_2_Advanced%20MultKAN_11_1.png


