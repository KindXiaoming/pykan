Example 7: Solving Partial Differential Equation (PDE)
======================================================

We aim to solve a 2D poisson equation
:math:`\nabla^2 f(x,y) = -2\pi^2{\rm sin}(\pi x){\rm sin}(\pi y)`, with
boundary condition :math:`f(-1,y)=f(1,y)=f(x,-1)=f(x,1)=0`. The ground
truth solution is :math:`f(x,y)={\rm sin}(\pi x){\rm sin}(\pi y)`.

.. code:: ipython3

    from kan import KAN, LBFGS
    import torch
    import matplotlib.pyplot as plt
    from torch import autograd
    from tqdm import tqdm
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    
    dim = 2
    np_i = 51 # number of interior points (along each dimension)
    np_b = 51 # number of boundary points (along each dimension)
    ranges = [-1, 1]
    
    
    def batch_jacobian(func, x, create_graph=False):
        # x in shape (Batch, Length)
        def _func_sum(x):
            return func(x).sum(dim=0)
        return autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)
    
    # define solution
    sol_fun = lambda x: torch.sin(torch.pi*x[:,[0]])*torch.sin(torch.pi*x[:,[1]])
    source_fun = lambda x: -2*torch.pi**2 * torch.sin(torch.pi*x[:,[0]])*torch.sin(torch.pi*x[:,[1]])
    
    # interior
    sampling_mode = 'mesh' # 'radnom' or 'mesh'
    
    x_mesh = torch.linspace(ranges[0],ranges[1],steps=np_i)
    y_mesh = torch.linspace(ranges[0],ranges[1],steps=np_i)
    X, Y = torch.meshgrid(x_mesh, y_mesh, indexing="ij")
    if sampling_mode == 'mesh':
        #mesh
        x_i = torch.stack([X.reshape(-1,), Y.reshape(-1,)]).permute(1,0)
    else:
        #random
        x_i = torch.rand((np_i**2,2))*2-1
        
    x_i = x_i.to(device)
    
    # boundary, 4 sides
    helper = lambda X, Y: torch.stack([X.reshape(-1,), Y.reshape(-1,)]).permute(1,0)
    xb1 = helper(X[0], Y[0])
    xb2 = helper(X[-1], Y[0])
    xb3 = helper(X[:,0], Y[:,0])
    xb4 = helper(X[:,0], Y[:,-1])
    x_b = torch.cat([xb1, xb2, xb3, xb4], dim=0)
    
    x_b = x_b.to(device)
    
    alpha = 0.01
    log = 1
    
    
    grids = [5,10,20]
    steps = 50
    
    pde_losses = []
    bc_losses = []
    l2_losses = []
    
    for grid in grids:
        if grid == grids[0]:
            model = KAN(width=[2,2,1], grid=grid, k=3, seed=1, device=device)
            model = model.speed()
        else:
            model.save_act = True
            model.get_act(x_i)
            model = model.refine(grid)
            model = model.speed()
    
        def train():
            optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
    
            pbar = tqdm(range(steps), desc='description', ncols=100)
    
            for _ in pbar:
                def closure():
                    global pde_loss, bc_loss
                    optimizer.zero_grad()
                    # interior loss
                    sol = sol_fun(x_i)
                    sol_D1_fun = lambda x: batch_jacobian(model, x, create_graph=True)[:,0,:]
                    sol_D1 = sol_D1_fun(x_i)
                    sol_D2 = batch_jacobian(sol_D1_fun, x_i, create_graph=True)[:,:,:]
                    lap = torch.sum(torch.diagonal(sol_D2, dim1=1, dim2=2), dim=1, keepdim=True)
                    source = source_fun(x_i)
                    pde_loss = torch.mean((lap - source)**2)
    
                    # boundary loss
                    bc_true = sol_fun(x_b)
                    bc_pred = model(x_b)
                    bc_loss = torch.mean((bc_pred-bc_true)**2)
    
                    loss = alpha * pde_loss + bc_loss
                    loss.backward()
                    return loss
    
                if _ % 5 == 0 and _ < 20:
                    model.update_grid_from_samples(x_i)
    
                optimizer.step(closure)
                sol = sol_fun(x_i)
                loss = alpha * pde_loss + bc_loss
                l2 = torch.mean((model(x_i) - sol)**2)
    
                if _ % log == 0:
                    pbar.set_description("pde loss: %.2e | bc loss: %.2e | l2: %.2e " % (pde_loss.cpu().detach().numpy(), bc_loss.cpu().detach().numpy(), l2.cpu().detach().numpy()))
    
                pde_losses.append(pde_loss.cpu().detach().numpy())
                bc_losses.append(bc_loss.cpu().detach().numpy())
                l2_losses.append(l2.cpu().detach().numpy())
                
            
        train()


.. parsed-literal::

    cuda
    checkpoint directory created: ./model
    saving model version 0.0


.. parsed-literal::

    pde loss: 2.13e+00 | bc loss: 1.80e-03 | l2: 3.11e-03 : 100%|███████| 50/50 [00:35<00:00,  1.43it/s]
    pde loss: 5.68e-01 | bc loss: 5.30e-04 | l2: 1.03e-03 : 100%|███████| 50/50 [00:35<00:00,  1.43it/s]
    pde loss: 1.23e-01 | bc loss: 1.51e-04 | l2: 1.74e-04 : 100%|███████| 50/50 [00:35<00:00,  1.42it/s]


.. code:: ipython3

    plt.plot(pde_losses, marker='o')
    plt.plot(bc_losses, marker='o')
    plt.plot(l2_losses, marker='o')
    plt.yscale('log')
    plt.xlabel('steps')
    plt.legend(['PDE loss', 'BC loss', 'L2 squared'])




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f94ac3c5cd0>




.. image:: Example_7_PDE_accuracy_files/Example_7_PDE_accuracy_3_1.png


