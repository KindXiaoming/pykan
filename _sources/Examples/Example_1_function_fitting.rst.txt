Example 1: Function Fitting
===========================

In this example, we will cover how to leverage grid refinement to
maximimze KANs’ ability to fit functions

intialize model and create dataset

.. code:: ipython3

    from kan import *
    
    # initialize KAN with G=3
    model = KAN(width=[2,1,1], grid=3, k=3)
    
    # create dataset
    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = create_dataset(f, n_var=2)

Train KAN (grid=3)

.. code:: ipython3

    model.train(dataset, opt="LBFGS", steps=20);


.. parsed-literal::

    train loss: 1.54e-02 | test loss: 1.50e-02 | reg: 3.01e+00 : 100%|██| 20/20 [00:03<00:00,  6.45it/s]


The loss plateaus. we want a more fine-grained KAN!

.. code:: ipython3

    # initialize a more fine-grained KAN with G=10
    model2 = KAN(width=[2,1,1], grid=10, k=3)
    # initialize model2 from model
    model2.initialize_from_another_model(model, dataset['train_input']);

Train KAN (grid=10)

.. code:: ipython3

    model2.train(dataset, opt="LBFGS", steps=20);


.. parsed-literal::

    train loss: 3.18e-04 | test loss: 3.29e-04 | reg: 3.00e+00 : 100%|██| 20/20 [00:02<00:00,  6.87it/s]


The loss becomes lower. This is good! Now we can even iteratively making
grids finer.

.. code:: ipython3

    grids = np.array([5,10,20,50,100])
    
    train_losses = []
    test_losses = []
    steps = 50
    k = 3
    
    for i in range(grids.shape[0]):
        if i == 0:
            model = KAN(width=[2,1,1], grid=grids[i], k=k)
        if i != 0:
            model = KAN(width=[2,1,1], grid=grids[i], k=k).initialize_from_another_model(model, dataset['train_input'])
        results = model.train(dataset, opt="LBFGS", steps=steps, stop_grid_update_step=30)
        train_losses += results['train_loss']
        test_losses += results['test_loss']
        


.. parsed-literal::

    train loss: 6.73e-03 | test loss: 6.62e-03 | reg: 2.86e+00 : 100%|██| 50/50 [00:06<00:00,  7.28it/s]
    train loss: 4.32e-04 | test loss: 4.15e-04 | reg: 2.89e+00 : 100%|██| 50/50 [00:07<00:00,  6.93it/s]
    train loss: 4.59e-05 | test loss: 4.51e-05 | reg: 2.88e+00 : 100%|██| 50/50 [00:12<00:00,  4.01it/s]
    train loss: 4.19e-06 | test loss: 1.04e-05 | reg: 2.88e+00 : 100%|██| 50/50 [00:30<00:00,  1.63it/s]
    train loss: 1.62e-06 | test loss: 8.17e-06 | reg: 2.88e+00 : 100%|██| 50/50 [00:40<00:00,  1.24it/s]


Training dynamics of losses display staircase structures (loss suddenly
drops after grid refinement)

.. code:: ipython3

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(['train', 'test'])
    plt.ylabel('RMSE')
    plt.xlabel('step')
    plt.yscale('log')



.. image:: Example_1_function_fitting_files/Example_1_function_fitting_12_0.png


Neural scaling laws

.. code:: ipython3

    n_params = 3 * grids
    train_vs_G = train_losses[(steps-1)::steps]
    test_vs_G = test_losses[(steps-1)::steps]
    plt.plot(n_params, train_vs_G, marker="o")
    plt.plot(n_params, test_vs_G, marker="o")
    plt.plot(n_params, 100*n_params**(-4.), ls="--", color="black")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['train', 'test', r'$N^{-4}$'])
    plt.xlabel('number of params')
    plt.ylabel('RMSE')




.. parsed-literal::

    Text(0, 0.5, 'RMSE')




.. image:: Example_1_function_fitting_files/Example_1_function_fitting_14_1.png

