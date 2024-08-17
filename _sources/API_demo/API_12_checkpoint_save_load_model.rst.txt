API 12: Checkpoint, save & load model
=====================================

Whenever the KAN (model) is altered (e.g., fit, prune …), a new version
is saved to the model.ckpt folder (by default ‘model’). The version
number is ‘a.b’, where a is the round number (starting from zero, +1
when model.rewind() is called), b is the version number in each round.

the initialized model has version 0.0

.. code:: ipython3

    from kan import *
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = create_dataset(f, n_var=2, device=device)
    model = KAN(width=[2,5,1], grid=5, k=3, seed=1, auto_save=True, device=device)
    model.get_act(dataset)
    model.plot()


.. parsed-literal::

    cuda
    checkpoint directory created: ./model
    saving model version 0.0



.. image:: API_12_checkpoint_save_load_model_files/API_12_checkpoint_save_load_model_3_1.png


the auto_save is on (by default)

.. code:: ipython3

    model.auto_save




.. parsed-literal::

    True



After fitting, the version becomes 0.1

.. code:: ipython3

    model.fit(dataset, opt="LBFGS", steps=20, lamb=0.01);
    model.plot()


.. parsed-literal::

    | train_loss: 3.34e-02 | test_loss: 3.29e-02 | reg: 4.93e+00 | : 100%|█| 20/20 [00:03<00:00,  5.10it


.. parsed-literal::

    saving model version 0.1



.. image:: API_12_checkpoint_save_load_model_files/API_12_checkpoint_save_load_model_7_2.png


After pruning, the version becomes 0.2

.. code:: ipython3

    model = model.prune()
    model.plot()


.. parsed-literal::

    saving model version 0.2



.. image:: API_12_checkpoint_save_load_model_files/API_12_checkpoint_save_load_model_9_1.png


Suppose we want to revert back to version 0.1, use model =
model.rewind(‘0.1’). This starts a new round, meaning version 0.1
renamed to version 1.1.

.. code:: ipython3

    # revert to version 0.1 (if continuing)
    model = model.rewind('0.1')
    
    # revert to version 0.1 (if starting from scratch)
    #model = KAN.loadckpt('./model' + '0.1')
    #model.get_act(dataset)
    
    model.plot()


.. parsed-literal::

    rewind to model version 0.1, renamed as 1.1



.. image:: API_12_checkpoint_save_load_model_files/API_12_checkpoint_save_load_model_11_1.png


Suppose we do some more manipulation to version 1.1, we will roll
forward to version 1.2

.. code:: ipython3

    model.fit(dataset, opt="LBFGS", steps=2);
    model.plot()


.. parsed-literal::

    | train_loss: 2.06e-02 | test_loss: 2.18e-02 | reg: 5.48e+00 | : 100%|█| 2/2 [00:00<00:00,  5.83it/s


.. parsed-literal::

    saving model version 1.2



.. image:: API_12_checkpoint_save_load_model_files/API_12_checkpoint_save_load_model_13_2.png

