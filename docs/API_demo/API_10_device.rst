Demo 10: Device
===============

All other demos have by default used device = ‘cpu’. In case we want to
use cuda, we should pass the device argument to model and dataset.

.. code:: ipython3

    from kan import KAN, create_dataset
    import torch
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(device)


.. parsed-literal::

    cpu


.. code:: ipython3

    model = KAN(width=[4,100,100,100,1], grid=3, k=3, seed=0).to(device)
    f = lambda x: torch.exp((torch.sin(torch.pi*(x[:,[0]]**2+x[:,[1]]**2))+torch.sin(torch.pi*(x[:,[2]]**2+x[:,[3]]**2)))/2)
    dataset = create_dataset(f, n_var=4, train_num=1000, device=device)
    
    # train the model
    #model.train(dataset, opt="LBFGS", steps=20, lamb=1e-3, lamb_entropy=2.);
    model.fit(dataset, opt="Adam", lr=1e-3, steps=50, lamb=1e-3, lamb_entropy=5., update_grid=False);


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0


.. parsed-literal::

    | train_loss: 6.83e-01 | test_loss: 7.21e-01 | reg: 1.04e+03 | : 100%|█| 50/50 [00:19<00:00,  2.56it


.. parsed-literal::

    saving model version 0.1



.. code:: ipython3

    device = 'cuda'
    print(device)


.. parsed-literal::

    cuda


.. code:: ipython3

    model = KAN(width=[4,100,100,100,1], grid=3, k=3, seed=0).to(device)
    f = lambda x: torch.exp((torch.sin(torch.pi*(x[:,[0]]**2+x[:,[1]]**2))+torch.sin(torch.pi*(x[:,[2]]**2+x[:,[3]]**2)))/2)
    dataset = create_dataset(f, n_var=4, train_num=1000, device=device)
    
    # train the model
    #model.train(dataset, opt="LBFGS", steps=20, lamb=1e-3, lamb_entropy=2.);
    model.fit(dataset, opt="Adam", lr=1e-3, steps=50, lamb=1e-3, lamb_entropy=5., update_grid=False);


.. parsed-literal::

    checkpoint directory created: ./model
    saving model version 0.0


.. parsed-literal::

    | train_loss: 6.83e-01 | test_loss: 7.21e-01 | reg: 1.04e+03 | : 100%|█| 50/50 [00:01<00:00, 26.45it


.. parsed-literal::

    saving model version 0.1


