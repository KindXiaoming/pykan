Interpretability 10B: swap
==========================

The multitask parity problem has 10 input bits
:math:`(x_1, x_2, \cdots, x_{10})`, :math:`x_i\in\{0,1\}`.

The are five output bits :math:`y_1, \cdots, y_5`, where
:math:`y_i = x_{2i-1} + x_{2i-1} ({\rm mod} 2)`

.. code:: ipython3

    from kan import *
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model = KAN(width=[10,10,5], seed=1, device=device)
    x = torch.normal(0,1,size=(100,2), device=device)
    
    #f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]])+x[:,[1]]**2)
    f = lambda x: torch.cat([x[:,[0]] + x[:,[1]], x[:,[2]] + x[:,[3]], x[:,[4]] + x[:,[5]], x[:,[6]] + x[:,[7]], x[:,[8]] + x[:,[9]]], dim=1)
    dataset = create_dataset(f, n_var=10, device=device)
    model.fit(dataset, steps=20, lamb=1e-2);



.. parsed-literal::

    cuda
    checkpoint directory created: ./model
    saving model version 0.0


.. parsed-literal::

    | train_loss: 8.26e-02 | test_loss: 7.72e-02 | reg: 1.66e+01 | : 100%|█| 20/20 [00:04<00:00,  4.93it

.. parsed-literal::

    saving model version 0.1


.. parsed-literal::

    


.. code:: ipython3

    model.plot()



.. image:: Interp_10B_swap_files/Interp_10B_swap_3_0.png


.. code:: ipython3

    model.auto_swap()


.. parsed-literal::

    saving model version 0.2


.. code:: ipython3

    model.plot()



.. image:: Interp_10B_swap_files/Interp_10B_swap_5_0.png


.. code:: ipython3

    # MLP
    from kan import *
    from kan.MLP import MLP
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    inputs = []
    for i in range(2**10):
        string = "{0:b}".format(i)
        sample = [int(string[i]) for i in range(len(string))]
        sample = (10 - len(sample)) * [0] + sample
        inputs.append(sample)
       
    inputs = np.array(inputs).astype(np.float32)
    labels = np.sum(inputs.reshape(2**10,5,2), axis=2) % 2
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    
    dataset = create_dataset_from_data(inputs, labels, device=device)
    
    model = MLP(width=[10,20,5], seed=5, device=device)
    model.fit(dataset, steps=100, lamb=2e-4, reg_metric='w');


.. parsed-literal::

    cuda


.. parsed-literal::

    | train_loss: 4.58e-03 | test_loss: 4.63e-03 | reg: 5.09e+01 | : 100%|█| 100/100 [00:04<00:00, 23.41


.. code:: ipython3

    model.plot(scale=1.5)



.. image:: Interp_10B_swap_files/Interp_10B_swap_7_0.png


.. code:: ipython3

    model.auto_swap()

.. code:: ipython3

    model.plot(scale=1.5)



.. image:: Interp_10B_swap_files/Interp_10B_swap_9_0.png


.. code:: ipython3

    model.auto_swap()

.. code:: ipython3

    model.plot(scale=1.5)



.. image:: Interp_10B_swap_files/Interp_10B_swap_11_0.png



