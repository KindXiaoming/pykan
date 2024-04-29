Demo 1: Indexing
================

.. code:: ipython3

    from kan import KAN
    import torch
    model = KAN(width=[2,3,2,1])
    x = torch.normal(0,1,size=(100,2))
    model(x);
    beta = 100
    model.plot(beta=beta)
    # [2,3,2,1] means 2 input nodes
    # 3 neurons in the first hidden layer,
    # 2 neurons in the second hidden layer,
    # 1 output node



.. image:: API_1_indexing_files/API_1_indexing_1_0.png


Indexing of edges (activation functions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each activation function is indexed by :math:`(l,i,j)` where :math:`l`
is the layer index, :math:`i` is the input neuron index, :math:`j` is
the output neuron index. All of them starts from 0. For example, the one
in the bottom left corner is (0, 0, 0). Let’s try to make it symbolic
and see it turns red.

.. code:: ipython3

    model.fix_symbolic(0,0,0,'sin')
    model.plot(beta=beta)
    model.unfix_symbolic(0,0,0)


.. parsed-literal::

    r2 is 0.9995602360489043



.. image:: API_1_indexing_files/API_1_indexing_4_1.png


.. code:: ipython3

    model.fix_symbolic(0,0,1,'sin')
    model.plot(beta=beta)
    model.unfix_symbolic(0,0,1)


.. parsed-literal::

    r2 is 0.9992399109543574



.. image:: API_1_indexing_files/API_1_indexing_5_1.png


.. code:: ipython3

    model.fix_symbolic(0,1,0,'sin')
    model.plot(beta=beta)
    model.unfix_symbolic(0,1,0)


.. parsed-literal::

    r2 is 0.9973507118333039



.. image:: API_1_indexing_files/API_1_indexing_6_1.png


.. code:: ipython3

    model.fix_symbolic(1,0,0,'sin')
    model.plot(beta=beta)
    model.unfix_symbolic(1,0,0)


.. parsed-literal::

    r2 is 0.9999506177136502



.. image:: API_1_indexing_files/API_1_indexing_7_1.png


.. code:: ipython3

    model.fix_symbolic(2,1,0,'sin')
    model.plot(beta=beta)
    model.unfix_symbolic(2,1,0)


.. parsed-literal::

    r2 is 0.9999411308602921



.. image:: API_1_indexing_files/API_1_indexing_8_1.png


Indexing of nodes (neurons)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each neuron (node) is indexed by :math:`(l,i)` where :math:`l` is the
layer index along depth, :math:`i` is the neuron index along width. In
the function remove_node, we use use :math:`(l,i)` to indicate which
node we want to remove.

.. code:: ipython3

    model.remove_node(1,0)

.. code:: ipython3

    model.plot(beta=beta)



.. image:: API_1_indexing_files/API_1_indexing_12_0.png


.. code:: ipython3

    model.remove_node(2,1)

.. code:: ipython3

    model.plot(beta=beta)



.. image:: API_1_indexing_files/API_1_indexing_14_0.png


.. code:: ipython3

    model.remove_node(1,2)

.. code:: ipython3

    model.plot(beta=beta)



.. image:: API_1_indexing_files/API_1_indexing_16_0.png


Indexing of layers
~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # KAN spline layers are refererred to as act_fun
    # KAN symbolic layers are referred to as symbolic_fun
    
    model = KAN(width=[2,3,2,1])
    
    i = 0
    model.act_fun[i] # => KAN Layer (Spline)
    model.symbolic_fun[i] # => KAN Layer (Symbolic)
    
    for i in range(3):
        print(model.act_fun[i].in_dim, model.act_fun[i].out_dim)
        print(model.symbolic_fun[i].in_dim, model.symbolic_fun[i].out_dim)


.. parsed-literal::

    2 3
    2 3
    3 2
    3 2
    2 1
    2 1


.. code:: ipython3

    # check model parameters
    model.act_fun[i].grid
    model.act_fun[i].coef
    model.symbolic_fun[i].funs_name
    model.symbolic_fun[i].mask




.. parsed-literal::

    Parameter containing:
    tensor([[0., 0.]])



