Demo 9: Videos
==============

We have shown one can visualize KAN with the plot() method. If one wants
to save the training dynamics of KAN plots, one only needs to pass
argument save_video = True to train() method (and set some video related
parameters)

.. code:: ipython3

    from kan import KAN, create_dataset
    import torch
    
    # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
    model = KAN(width=[4,2,1,1], grid=3, k=3, seed=0)
    f = lambda x: torch.exp((torch.sin(torch.pi*(x[:,[0]]**2+x[:,[1]]**2))+torch.sin(torch.pi*(x[:,[2]]**2+x[:,[3]]**2)))/2)
    dataset = create_dataset(f, n_var=4, train_num=3000)
    
    # train the model
    #model.train(dataset, opt="LBFGS", steps=20, lamb=1e-3, lamb_entropy=2.);
    model.train(dataset, opt="LBFGS", steps=50, lamb=5e-5, lamb_entropy=2., save_video=True, beta=10, 
                in_vars=[r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$'],
                out_vars=[r'${\rm exp}({\rm sin}(x_1^2+x_2^2)+{\rm sin}(x_3^2+x_4^2))$'],
                video_name='video', fps=5);


.. parsed-literal::

    train loss: 6.39e-03 | test loss: 6.40e-03 | reg: 7.91e+00 : 100%|██| 50/50 [01:30<00:00,  1.81s/it]


.. parsed-literal::

    Moviepy - Building video video.mp4.
    Moviepy - Writing video video.mp4
    


.. parsed-literal::

                                                                                    

.. parsed-literal::

    Moviepy - Done !
    Moviepy - video ready video.mp4

