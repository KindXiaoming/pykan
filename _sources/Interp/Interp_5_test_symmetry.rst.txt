Interprebility 5: Test symmetries
=================================

Figuring out the symbolic formula represented by a model is ideal but
sometimes too challenging. In this case, we might be content with simply
figuring out some modular structures or symmetries. These hypothesis
testing is partially inspired by AI Feynman.

.. code:: ipython3

    from kan.hypothesis import *
    import torch

Case 1: detect separability. \* Additive separability:
:math:`f(x_1, x_2, ...) = g_1(x_1,x_2) + g_2(x_3) + g_3(x_4,x_5,x_6) + ...`
\* Multiplicative separability:
:math:`f(x_1, x_2, ...) = g_1(x_1,x_2)g_2(x_3)g_3(x_4,x_5,x_6)...` \*
General separability:
:math:`f(x_1, x_2, x_3, ...) = h(p(x_1,x_2)+q(x_3,\cdots))`. (Note that
general additive separability = general multiplicative separability)

.. code:: ipython3

    f = lambda x: x[:,[0]] * x[:,[1]] + x[:,[2]] * x[:,[3]] + x[:,[4]] * x[:,[5]]
    x = torch.rand(100,6) * 2 - 1
    detect_separability(f, x, 'add')


.. parsed-literal::

    add separability detected




.. parsed-literal::

    {'hessian': tensor([[0.0000, 0.3147, 0.0000, 0.0000, 0.0000, 0.0000],
             [0.3147, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.3619, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.3619, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3358],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.3358, 0.0000]]),
     'n_groups': 3,
     'labels': [2, 2, 1, 1, 0, 0],
     'groups': [[4, 5], [2, 3], [0, 1]]}



.. code:: ipython3

    f = lambda x: (x[:,[0]] + x[:,[1]]) * (x[:,[2]] + x[:,[3]]) * (x[:,[4]] + x[:,[5]])
    x = torch.rand(100,6) * 2 - 1
    detect_separability(f, x, 'mul');


.. parsed-literal::

    mul separability detected


We could also test separability by providing a group partition as an
argument.

.. code:: ipython3

    f = lambda x: (x[:,[0]] + x[:,[1]]) * (x[:,[2]] + x[:,[3]]) * (x[:,[4]] + x[:,[5]])
    x = torch.rand(100,6) * 2 - 1
    groups = [[0,1],[2,3],[4,5]]
    test_separability(f, x, groups, 'mul')




.. parsed-literal::

    tensor(True)



.. code:: ipython3

    test_separability(f, x, [[0,1],[2,4],[3,5]], 'mul')




.. parsed-literal::

    tensor(False)



.. code:: ipython3

    f = lambda x: torch.sin((x[:,[0]] + x[:,[1]]) * (x[:,[2]] + x[:,[3]]) * (x[:,[4]] + x[:,[5]]))
    x = torch.rand(100,6) * 2 - 1
    test_separability(f, x, [[0,1],[2,3],[4,5]], 'mul')




.. parsed-literal::

    tensor(False)



.. code:: ipython3

    test_general_separability(f, x, [[0,1],[2,3],[4,5]])




.. parsed-literal::

    tensor(True)



Case 2: test symmetry. \* Symmetry means the output :math:`y` is only
dependent on a scalar function of a few variables, but otherwise does
not gain more infomration from knowing the individual values of these
variables. \* For example, we say a function has a symmetry
:math:`h(x_1, x_2)` if
:math:`f(x_1,x_2,x_3,\cdots)= g(h(x_1, x_2), x_3,\cdots)`. \* To
hypothesis test :math:`h`, use test_symmetry_var

.. code:: ipython3

    f = lambda x: (x[:,[0]] + x[:,[1]]) * (x[:,[2]] + x[:,[3]]) * (x[:,[4]] + x[:,[5]])
    x = torch.rand(100,6) * 2 - 1
    print('[0,1]:', test_symmetry(f, x, [0,1]))
    print('[0,2]:', test_symmetry(f, x, [0,2]))
    print('[2,3]:', test_symmetry(f, x, [2,3]))


.. parsed-literal::

    [0,1]: tensor(True)
    [0,2]: tensor(False)
    [2,3]: tensor(True)


.. code:: ipython3

    from sympy import *
    
    # the function is only dependent on b/c, but not on the individual values of b and c.
    f = lambda x: x[:,[0]] * torch.sqrt(1 + (x[:,[1]]/x[:,[2]])**2)
    input_vars = a, b, c = symbols('a b c')
    symmetry_var = b/c
    x = torch.rand(100,3) * 2 - 1
    test_symmetry_var(f, x, input_vars, symmetry_var);


.. parsed-literal::

    100.0% data have more than 0.9 cosine similarity
    suggesting symmetry


.. code:: ipython3

    not_symmetry_var = b * c
    test_symmetry_var(f, x, input_vars, not_symmetry_var);


.. parsed-literal::

    20.0% data have more than 0.9 cosine similarity
    not suggesting symmetry


Case 3: Plot tree graph. By applying the hypothesis testing above
iteratively, we are able to figure out the tree graph.

.. code:: ipython3

    f = lambda x: ((x[:,[0]]**2 + x[:,[1]]**2) ** 2 + (x[:,[2]]**2 + x[:,[3]]**2) ** 2) ** 2 + ((x[:,[4]]**2 + x[:,[5]]**2) ** 2 + (x[:,[6]]**2 + x[:,[7]]**2) ** 2) ** 2
    x = torch.rand(100,8) * 2 - 1
    plot_tree(f, x, style='tree') # by default, style = 'tree'



.. image:: Interp_5_test_symmetry_files/Interp_5_test_symmetry_16_0.png


.. code:: ipython3

    plot_tree(f, x, style='box')



.. image:: Interp_5_test_symmetry_files/Interp_5_test_symmetry_17_0.png


.. code:: ipython3

    f = lambda x: ((x[:,[0]]**2 + x[:,[1]]**2) ** 2 + (x[:,[2]]**2 + x[:,[3]]**2) ** 2) ** 2 + x[:,[4]]**2
    x = torch.rand(100,5) * 2 - 1
    plot_tree(f, x, style='tree') # by default, style = 'tree'



.. image:: Interp_5_test_symmetry_files/Interp_5_test_symmetry_18_0.png


.. code:: ipython3

    plot_tree(f, x, style='box')



.. image:: Interp_5_test_symmetry_files/Interp_5_test_symmetry_19_0.png


