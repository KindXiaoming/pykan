Interprebility 6: Test symmetries of trained NN
===============================================

.. code:: ipython3

    from kan import *
    from kan.hypothesis import plot_tree
    
    f = lambda x: (x[:,[0]]**2 + x[:,[1]]**2) ** 2 + (x[:,[2]]**2 + x[:,[3]]**2) ** 2
    x = torch.rand(100,4) * 2 - 1
    plot_tree(f, x)



.. image:: Interp_6_test_symmetry_NN_files/Interp_6_test_symmetry_NN_1_0.png


.. code:: ipython3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    dataset = create_dataset(f, n_var=4, device=device)
    model = KAN(width=[4,5,5,1], seed=0, device=device)
    model.fit(dataset, steps=100);


.. parsed-literal::

    cuda
    checkpoint directory created: ./model
    saving model version 0.0


.. parsed-literal::

    | train_loss: 1.58e-03 | test_loss: 4.79e-03 | reg: 2.38e+01 | : 100%|█| 100/100 [00:20<00:00,  4.93

.. parsed-literal::

    saving model version 0.1


.. parsed-literal::

    


.. code:: ipython3

    model.tree(sym_th=1e-2, sep_th=5e-1)



.. image:: Interp_6_test_symmetry_NN_files/Interp_6_test_symmetry_NN_3_0.png


