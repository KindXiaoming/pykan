Demo 6: Training Hyperparamters
===============================

Regularization helps interpretability by making KANs sparser. This may
require some hyperparamter tuning. Let’s see how hyperparameters can
affect training

Load KAN and create_dataset

.. code:: ipython3

    from kan import KAN, create_dataset
    import torch
    
    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = create_dataset(f, n_var=2)
    dataset['train_input'].shape, dataset['train_label'].shape




.. parsed-literal::

    (torch.Size([1000, 2]), torch.Size([1000, 1]))



Default setup

.. code:: ipython3

    # train the model
    model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.1);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 1.69e-01 | test loss: 1.50e-01 | reg: 5.01e+00 : 100%|██| 20/20 [00:12<00:00,  1.59it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_4_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_4_2.png


Parameter 1: :math:`\lambda`, overall penalty strength.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously :math:`\lambda=0.1`, now we try different :math:`\lambda`.

:math:`\lambda=0`

.. code:: ipython3

    # train the model
    model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.00);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 4.16e-03 | test loss: 5.00e-03 | reg: 1.24e+01 : 100%|██| 20/20 [00:10<00:00,  1.86it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_7_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_7_2.png


:math:`\lambda=10^{-2}`

.. code:: ipython3

    # train the model
    model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.1, lamb_entropy=10.);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 6.01e-01 | test loss: 5.65e-01 | reg: 1.78e+01 : 100%|██| 20/20 [00:13<00:00,  1.51it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_9_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_9_2.png


:math:`\lambda=1`

.. code:: ipython3

    # train the model
    model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=1);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 1.09e+00 | test loss: 1.02e+00 | reg: 5.18e+00 : 100%|██| 20/20 [00:11<00:00,  1.67it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_11_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_11_2.png


Parameter 2: (relative) penalty strength of entropy :math:`\lambda_{\rm ent}`.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The absolute magnitude is :math:`\lambda\lambda_{\rm ent}`. Previously
we set :math:`\lambda=0.1` and :math:`\lambda_{\rm ent}=10.0`. Below we
fix :math:`\lambda=0.1` and vary :math:`\lambda_{\rm ent}`.

:math:`\lambda_{\rm ent}=0.0`

.. code:: ipython3

    # train the model
    model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.1, lamb_entropy=0.0);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 8.90e-02 | test loss: 8.40e-02 | reg: 1.68e+00 : 100%|██| 20/20 [00:12<00:00,  1.65it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_14_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_14_2.png


:math:`\lambda_{\rm ent}=10.`

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.1, lamb_entropy=10.0);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 6.03e-01 | test loss: 5.67e-01 | reg: 1.77e+01 : 100%|██| 20/20 [00:10<00:00,  1.89it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_16_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_16_2.png


:math:`\lambda_{\rm ent}=100.`

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.1, lamb_entropy=100.0);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 1.60e+00 | test loss: 1.54e+00 | reg: 2.69e+02 : 100%|██| 20/20 [00:11<00:00,  1.67it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_18_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_18_2.png


Parameter 3: Grid size :math:`G`.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously we set :math:`G=5`, we vary :math:`G` below.

:math:`G=1`

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=1, k=3, seed=0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=2.);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 1.41e-01 | test loss: 1.33e-01 | reg: 1.01e+01 : 100%|██| 20/20 [00:06<00:00,  2.95it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_21_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_21_2.png


:math:`G=3`

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=3, k=3, seed=0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=2.);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 6.18e-02 | test loss: 5.66e-02 | reg: 5.93e+00 : 100%|██| 20/20 [00:11<00:00,  1.76it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_23_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_23_2.png


:math:`G=5`

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=2.);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 7.47e-02 | test loss: 6.52e-02 | reg: 6.12e+00 : 100%|██| 20/20 [00:12<00:00,  1.58it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_25_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_25_2.png


:math:`G=10`

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=10, k=3, seed=0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=2.);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 8.08e-02 | test loss: 7.24e-02 | reg: 5.89e+00 : 100%|██| 20/20 [00:13<00:00,  1.44it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_27_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_27_2.png


:math:`G=20`

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=20, k=3, seed=0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=2.);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 5.14e-02 | test loss: 5.50e-02 | reg: 7.70e+00 : 100%|██| 20/20 [00:16<00:00,  1.23it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_29_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_29_2.png


Parameter 4: seed.
~~~~~~~~~~~~~~~~~~

Previously we use seed = 0. Below we vary seed.

:math:`{\rm seed} = 1`

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=1, noise_scale_base=0.0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 5.58e-02 | test loss: 5.50e-02 | reg: 8.48e+00 : 100%|██| 20/20 [00:13<00:00,  1.50it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_32_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_32_2.png


:math:`{\rm seed} = 42`

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=42, noise_scale_base=0.0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 1.43e-01 | test loss: 1.25e-01 | reg: 1.85e+01 : 100%|██| 20/20 [00:12<00:00,  1.65it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_34_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_34_2.png


:math:`{\rm seed} = 2024`

.. code:: ipython3

    model = KAN(width=[2,5,1], grid=5, k=3, seed=2024, noise_scale_base=0.0)
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);
    model.plot()
    model.prune()
    model.plot(mask=True)


.. parsed-literal::

    train loss: 1.50e-01 | test loss: 1.39e-01 | reg: 2.37e+01 : 100%|██| 20/20 [00:12<00:00,  1.57it/s]



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_36_1.png



.. image:: API_6_training_hyperparameter_files/API_6_training_hyperparameter_36_2.png


