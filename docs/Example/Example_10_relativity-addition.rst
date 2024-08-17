Example 10: Relativitistic Velocity Addition
============================================

In this example, we will symbolically regress
:math:`f(u,v)=\frac{u+v}{1+uv}`. In relavitity, we know the rapidity
trick :math:`f(u,v)={\rm tanh}({\rm arctanh}\ u+{\rm arctanh}\ v)`. Can
we rediscover rapidity trick with KAN?

Intialize model and create dataset

.. code:: ipython3

    from kan import *
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # initialize KAN with G=3
    model = KAN(width=[2,1,1], grid=10, k=3, device=device)
    
    # create dataset
    f = lambda x: (x[:,[0]]+x[:,[1]])/(1+x[:,[0]]*x[:,[1]])
    dataset = create_dataset(f, n_var=2, ranges=[-0.9,0.9], device=device)


.. parsed-literal::

    cuda
    checkpoint directory created: ./model
    saving model version 0.0


Train KAN and plot

.. code:: ipython3

    model.fit(dataset, opt="LBFGS", steps=20);


.. parsed-literal::

    | train_loss: 2.28e-03 | test_loss: 2.31e-03 | reg: 6.50e+00 | : 100%|█| 20/20 [00:03<00:00,  5.88it

.. parsed-literal::

    saving model version 0.1


.. parsed-literal::

    


.. code:: ipython3

    model.plot(beta=10)



.. image:: Example_10_relativity-addition_files/Example_10_relativity-addition_6_0.png


Retrain the model, the loss remains similar, meaning that the locking
does not degrade model behavior, justifying our hypothesis that these
two activation functions are the same. Let’s now determine what this
function is using :math:`\texttt{suggest_symbolic}`

.. code:: ipython3

    model.suggest_symbolic(0,1,0,weight_simple=0.0)


.. parsed-literal::

      function  fitting r2    r2 loss  complexity  complexity loss  total loss
    0  arctanh    0.999992 -15.786788           4                4  -15.786788
    1      tan    0.999825 -12.397871           3                3  -12.397871
    2   arccos    0.998852  -9.753944           4                4   -9.753944
    3   arcsin    0.998852  -9.753944           4                4   -9.753944
    4     sqrt    0.982166  -5.808383           2                2   -5.808383




.. parsed-literal::

    ('arctanh',
     (<function kan.utils.<lambda>(x)>,
      <function kan.utils.<lambda>(x)>,
      4,
      <function kan.utils.<lambda>(x, y_th)>),
     0.999992311000824,
     4)



We can see that :math:`{\rm arctanh}` is at the top of the suggestion
list! So we can set both to arctanh, retrain the model, and plot it.

.. code:: ipython3

    model.fix_symbolic(0,0,0,'arctanh')
    model.fix_symbolic(0,1,0,'arctanh')


.. parsed-literal::

    r2 is 0.9999759197235107
    saving model version 0.2
    r2 is 0.999992311000824
    saving model version 0.3




.. parsed-literal::

    tensor(1.0000, device='cuda:0')



.. code:: ipython3

    model.fit(dataset, opt="LBFGS", steps=20, update_grid=False);


.. parsed-literal::

    | train_loss: 7.94e-04 | test_loss: 9.43e-04 | reg: 4.12e+00 | : 100%|█| 20/20 [00:04<00:00,  4.34it

.. parsed-literal::

    saving model version 0.4


.. parsed-literal::

    


.. code:: ipython3

    model.plot(beta=10)



.. image:: Example_10_relativity-addition_files/Example_10_relativity-addition_12_0.png


We will see that :math:`{\rm tanh}` is at the top of the suggestion
list! So we can set it to :math:`{\rm tanh}`, retrain the model to
machine precision, plot it and finally get the symbolic formula.

.. code:: ipython3

    model.suggest_symbolic(1,0,0,weight_simple=0.)


.. parsed-literal::

       function  fitting r2    r2 loss  complexity  complexity loss  total loss
    0      tanh    0.999998 -16.336284           3                3  -16.336284
    1    arctan    0.999435 -10.764618           4                4  -10.764618
    2       cos    0.995899  -7.926177           2                2   -7.926177
    3       sin    0.995899  -7.926177           2                2   -7.926177
    4  gaussian    0.994457  -7.492519           3                3   -7.492519




.. parsed-literal::

    ('tanh',
     (<function kan.utils.<lambda>(x)>,
      <function kan.utils.<lambda>(x)>,
      3,
      <function kan.utils.<lambda>(x, y_th)>),
     0.9999979138374329,
     3)



.. code:: ipython3

    model.fix_symbolic(1,0,0,'tanh')


.. parsed-literal::

    r2 is 0.9999979138374329
    saving model version 0.5




.. parsed-literal::

    tensor(1.0000, device='cuda:0')



.. code:: ipython3

    model.fit(dataset, opt="Adam", lr=1e-3, steps=2000, update_grid=False, singularity_avoiding=True);


.. parsed-literal::

    | train_loss: 1.97e-06 | test_loss: 2.06e-06 | reg: 0.00e+00 | : 100%|█| 2000/2000 [00:21<00:00, 93.


.. parsed-literal::

    saving model version 0.6


.. code:: ipython3

    model.plot()



.. image:: Example_10_relativity-addition_files/Example_10_relativity-addition_17_0.png


.. code:: ipython3

    formula = model.symbolic_formula()[0][0]
    nsimplify(ex_round(formula, 4))




.. math::

    \displaystyle \tanh{\left(\operatorname{atanh}{\left(x_{1} \right)} + \operatorname{atanh}{\left(x_{2} \right)} \right)}


