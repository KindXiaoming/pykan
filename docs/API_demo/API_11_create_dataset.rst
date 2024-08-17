API 11: Create dataset
======================

how to use create_dataset in kan.utils

Standard way

.. code:: ipython3

    from kan.utils import create_dataset
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    f = lambda x: x[:,[0]] * x[:,[1]]
    dataset = create_dataset(f, n_var=2, device=device)
    dataset['train_label'].shape


.. parsed-literal::

    cuda




.. parsed-literal::

    torch.Size([1000, 1])



Lazier way. We sometimes forget to add the bracket, i.e., write x[:,[0]]
as x[:,0], and this used to lead to an error in training (loss not going
down). Now the create_dataset can automatically detect this
simplification and produce the correct behavior.

.. code:: ipython3

    f = lambda x: x[:,0] * x[:,1]
    dataset = create_dataset(f, n_var=2, device=device)
    dataset['train_label'].shape




.. parsed-literal::

    torch.Size([1000, 1])



Laziest way. If you even want to get rid of the colon symbol, i.e., you
want to write x[;,0] as x[0], you can do that but need to pass in f_mode
= ‘row’.

.. code:: ipython3

    f = lambda x: x[0] * x[1]
    dataset = create_dataset(f, n_var=2, f_mode='row', device=device)
    dataset['train_label'].shape




.. parsed-literal::

    torch.Size([1000, 1])



if you already have x (inputs) and y (outputs), and you only want to
partition them into train/test, use create_dataset_from_data

.. code:: ipython3

    import torch
    from kan.utils import create_dataset_from_data
    
    x = torch.rand(100,2)
    y = torch.rand(100,1)
    dataset = create_dataset_from_data(x, y, device=device)

