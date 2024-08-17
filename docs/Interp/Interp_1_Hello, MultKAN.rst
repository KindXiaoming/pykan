Interpretability 1: Hello, MultKAN!
===================================

Motivation: The original KAN has some level of interpretability, but
sometimes not fully interpretable (fully interpretable = convert the
network to a symbolic formula). The biggest limitation is the lack of
multiplications operators. The original KAN only has addition operators.
Although multiplication can be expressed as addition and single-variable
functions (which is the core idea of Kolmogorov-Arnold representation
theorem), we still hope to explicitly have multiplications in the KANs
so that multiplications can be more easily read out from KANs.

We first show how multiplications can be represented by addition and
single variable functions. Usually KAN would find solutions leveraging
linear functions and quadractic functions (the solutions are not
unique).

.. math:: xy=((x+y)^2-(x-y)^2)/4=((x+y)^2-x^2-y^2)/2=\cdots

.. code:: ipython3

    from kan import *
    torch.set_default_dtype(torch.float64)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model = KAN(width=[2,5,1], device=device)
    
    f = lambda x: x[:,0] * x[:,1]
    dataset = create_dataset(f, n_var=2, device=device)
    model.fit(dataset, steps=20, lamb=0.001);


.. parsed-literal::

    cuda
    checkpoint directory created: ./model
    saving model version 0.0


.. parsed-literal::

    | train_loss: 4.73e-03 | test_loss: 4.96e-03 | reg: 6.68e+00 | : 100%|█| 20/20 [00:04<00:00,  4.77it

.. parsed-literal::

    saving model version 0.1


.. parsed-literal::

    


.. code:: ipython3

    model.plot()



.. image:: Interp_1_Hello%2C%20MultKAN_files/Interp_1_Hello%2C%20MultKAN_4_0.png


This network seems to be using the equality
:math:`xy=((x+y)^2-(x-y)^2)/4` but not exactly.

Now we want to explicitly introduce multiplication operators, called
MultKAN. Note that MultKAN and KAN are actually the same class in
implementation, so you can use either class name. If you dig into
MultKAN.py, there is a line ‘KAN = MultKAN’. KAN is just a special case
of MultKAN. To inlcude multiplications, you only need to modify the
width parameter. For example, [2,5,1] KAN means 2 inputs, 5 hidden add
neurons, and 1 output; [2,[5,2],1] MultKAN means 2 inputs, 5 hidden add
neurons and 2 hidden mult neurons, and 1 output.

.. code:: ipython3

    model = KAN(width=[2,[5,2],1], base_fun='identity', device=device)
    model.get_act(dataset)
    model.plot()


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0



.. image:: Interp_1_Hello%2C%20MultKAN_files/Interp_1_Hello%2C%20MultKAN_7_1.png


.. code:: ipython3

    model.fit(dataset, steps=20, lamb=0.01, lamb_coef=1.0);


.. parsed-literal::

    | train_loss: 6.34e-02 | test_loss: 7.16e-02 | reg: 7.99e+00 | : 100%|█| 20/20 [00:04<00:00,  4.79it


.. parsed-literal::

    saving model version 0.1


.. code:: ipython3

    model.plot()



.. image:: Interp_1_Hello%2C%20MultKAN_files/Interp_1_Hello%2C%20MultKAN_9_0.png


.. code:: ipython3

    model = model.prune()


.. parsed-literal::

    saving model version 0.2


.. code:: ipython3

    model.plot()



.. image:: Interp_1_Hello%2C%20MultKAN_files/Interp_1_Hello%2C%20MultKAN_11_0.png


.. code:: ipython3

    model.fit(dataset, steps=20);


.. parsed-literal::

    | train_loss: 1.37e-07 | test_loss: 1.66e-07 | reg: 6.31e+00 | : 100%|█| 20/20 [00:02<00:00,  6.90it


.. parsed-literal::

    saving model version 0.3


.. code:: ipython3

    model.auto_symbolic()


.. parsed-literal::

    fixing (0,0,0) with x, r2=0.9999999997931204, c=1
    fixing (0,0,1) with 0
    fixing (0,1,0) with 0
    fixing (0,1,1) with x, r2=0.99999999995849, c=1
    fixing (1,0,0) with x, r2=0.9999999918922519, c=1
    saving model version 0.4


.. code:: ipython3

    model.fit(dataset, steps=20);


.. parsed-literal::

    | train_loss: 1.43e-16 | test_loss: 1.28e-16 | reg: 0.00e+00 | : 100%|█| 20/20 [00:00<00:00, 37.98it

.. parsed-literal::

    saving model version 0.5


.. parsed-literal::

    


.. code:: ipython3

    sf = model.symbolic_formula()[0][0]
    nsimplify(ex_round(ex_round(sf, 3),3))




.. math::

    \displaystyle x_{1} x_{2}



