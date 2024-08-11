import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import sympy
import yaml
from sympy.utilities.lambdify import lambdify

# sigmoid = sympy.Function('sigmoid')
# name: (torch implementation, sympy implementation)

# singularity protection functions
f_inv = lambda x, y_th: ((x_th := 1/y_th), y_th/x_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(1/x) * (torch.abs(x) >= x_th))
f_inv2 = lambda x, y_th: ((x_th := 1/y_th**(1/2)), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1/x**2) * (torch.abs(x) >= x_th))
f_inv3 = lambda x, y_th: ((x_th := 1/y_th**(1/3)), y_th/x_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(1/x**3) * (torch.abs(x) >= x_th))
f_inv4 = lambda x, y_th: ((x_th := 1/y_th**(1/4)), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1/x**4) * (torch.abs(x) >= x_th))
f_inv5 = lambda x, y_th: ((x_th := 1/y_th**(1/5)), y_th/x_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(1/x**5) * (torch.abs(x) >= x_th))
f_sqrt = lambda x, y_th: ((x_th := 1/y_th**2), x_th/y_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(torch.sqrt(torch.abs(x))*torch.sign(x)) * (torch.abs(x) >= x_th))
f_power1d5 = lambda x, y_th: torch.abs(x)**1.5
f_invsqrt = lambda x, y_th: ((x_th := 1/y_th**2), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1/torch.sqrt(torch.abs(x))) * (torch.abs(x) >= x_th))
f_log = lambda x, y_th: ((x_th := torch.e**(-y_th)), - y_th * (torch.abs(x) < x_th) + torch.nan_to_num(torch.log(torch.abs(x))) * (torch.abs(x) >= x_th))
f_tan = lambda x, y_th: ((clip := x % torch.pi), (delta := torch.pi/2-torch.arctan(y_th)), - y_th/delta * (clip - torch.pi/2) * (torch.abs(clip - torch.pi/2) < delta) + torch.nan_to_num(torch.tan(clip)) * (torch.abs(clip - torch.pi/2) >= delta))
f_arctanh = lambda x, y_th: ((delta := 1-torch.tanh(y_th) + 1e-4), y_th * torch.sign(x) * (torch.abs(x) > 1 - delta) + torch.nan_to_num(torch.arctanh(x)) * (torch.abs(x) <= 1 - delta))
f_arcsin = lambda x, y_th: ((), torch.pi/2 * torch.sign(x) * (torch.abs(x) > 1) + torch.nan_to_num(torch.arcsin(x)) * (torch.abs(x) <= 1))
f_arccos = lambda x, y_th: ((), torch.pi/2 * (1-torch.sign(x)) * (torch.abs(x) > 1) + torch.nan_to_num(torch.arccos(x)) * (torch.abs(x) <= 1))
f_exp = lambda x, y_th: ((x_th := torch.log(y_th)), y_th * (x > x_th) + torch.exp(x) * (x <= x_th))

SYMBOLIC_LIB = {'x': (lambda x: x, lambda x: x, 1, lambda x, y_th: ((), x)),
                 'x^2': (lambda x: x**2, lambda x: x**2, 2, lambda x, y_th: ((), x**2)),
                 'x^3': (lambda x: x**3, lambda x: x**3, 3, lambda x, y_th: ((), x**3)),
                 'x^4': (lambda x: x**4, lambda x: x**4, 3, lambda x, y_th: ((), x**4)),
                 'x^5': (lambda x: x**5, lambda x: x**5, 3, lambda x, y_th: ((), x**5)),
                 '1/x': (lambda x: 1/x, lambda x: 1/x, 2, f_inv),
                 '1/x^2': (lambda x: 1/x**2, lambda x: 1/x**2, 2, f_inv2),
                 '1/x^3': (lambda x: 1/x**3, lambda x: 1/x**3, 3, f_inv3),
                 '1/x^4': (lambda x: 1/x**4, lambda x: 1/x**4, 4, f_inv4),
                 '1/x^5': (lambda x: 1/x**5, lambda x: 1/x**5, 5, f_inv5),
                 'sqrt': (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
                 'x^0.5': (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
                 'x^1.5': (lambda x: torch.sqrt(x)**3, lambda x: sympy.sqrt(x)**3, 4, f_power1d5),
                 '1/sqrt(x)': (lambda x: 1/torch.sqrt(x), lambda x: 1/sympy.sqrt(x), 2, f_invsqrt),
                 '1/x^0.5': (lambda x: 1/torch.sqrt(x), lambda x: 1/sympy.sqrt(x), 2, f_invsqrt),
                 'exp': (lambda x: torch.exp(x), lambda x: sympy.exp(x), 2, f_exp),
                 'log': (lambda x: torch.log(x), lambda x: sympy.log(x), 2, f_log),
                 'abs': (lambda x: torch.abs(x), lambda x: sympy.Abs(x), 3, lambda x, y_th: ((), torch.abs(x))),
                 'sin': (lambda x: torch.sin(x), lambda x: sympy.sin(x), 2, lambda x, y_th: ((), torch.sin(x))),
                 'cos': (lambda x: torch.cos(x), lambda x: sympy.cos(x), 2, lambda x, y_th: ((), torch.cos(x))),
                 'tan': (lambda x: torch.tan(x), lambda x: sympy.tan(x), 3, f_tan),
                 'tanh': (lambda x: torch.tanh(x), lambda x: sympy.tanh(x), 3, lambda x, y_th: ((), torch.tanh(x))),
                 'sgn': (lambda x: torch.sign(x), lambda x: sympy.sign(x), 3, lambda x, y_th: ((), torch.sign(x))),
                 'arcsin': (lambda x: torch.arcsin(x), lambda x: sympy.asin(x), 4, f_arcsin),
                 'arccos': (lambda x: torch.arccos(x), lambda x: sympy.acos(x), 4, f_arccos),
                 'arctan': (lambda x: torch.arctan(x), lambda x: sympy.atan(x), 4, lambda x, y_th: ((), torch.arctan(x))),
                 'arctanh': (lambda x: torch.arctanh(x), lambda x: sympy.atanh(x), 4, f_arctanh),
                 '0': (lambda x: x*0, lambda x: x*0, 0, lambda x, y_th: ((), x*0)),
                 'gaussian': (lambda x: torch.exp(-x**2), lambda x: sympy.exp(-x**2), 3, lambda x, y_th: ((), torch.exp(-x**2))),
                 #'cosh': (lambda x: torch.cosh(x), lambda x: sympy.cosh(x), 5),
                 #'sigmoid': (lambda x: torch.sigmoid(x), sympy.Function('sigmoid'), 4),
                 #'relu': (lambda x: torch.relu(x), relu),
}

def create_dataset(f, 
                   n_var=2, 
                   f_mode = 'col',
                   ranges = [-1,1],
                   train_num=1000, 
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0):
    '''
    create dataset
    
    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.
        
    Returns:
    --------
        dataset : dic
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']
         
    Example
    -------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    torch.Size([100, 2])
    '''

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var,2)
    else:
        ranges = np.array(ranges)
        
    
    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:,i] = torch.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        test_input[:,i] = torch.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
                
    if f_mode == 'col':
        train_label = f(train_input)
        test_label = f(test_input)
    elif f_mode == 'row':
        train_label = f(train_input.T)
        test_label = f(test_input.T)
    else:
        print(f'f_mode {f_mode} not recognized')
        
    # if has only 1 dimension
    if len(train_label.shape) == 1:
        train_label = train_label.unsqueeze(dim=1)
        test_label = test_label.unsqueeze(dim=1)
        
    def normalize(data, mean, std):
            return (data-mean)/std
            
    if normalize_input == True:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
        
    if normalize_label == True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)

    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)

    return dataset



def fit_params(x, y, fun, a_range=(-10,10), b_range=(-10,10), grid_number=101, iteration=3, verbose=True, device='cpu'):
    '''
    fit a, b, c, d such that
    
    .. math::
        |y-(cf(ax+b)+d)|^2
        
    is minimized. Both x and y are 1D array. Sweep a and b, find the best fitted model.
    
    Args:
    -----
        x : 1D array
            x values
        y : 1D array
            y values
        fun : function
            symbolic function
        a_range : tuple
            sweeping range of a
        b_range : tuple
            sweeping range of b
        grid_num : int
            number of steps along a and b
        iteration : int
            number of zooming in
        verbose : bool
            print extra information if True
        device : str
            device
        
    Returns:
    --------
        a_best : float
            best fitted a
        b_best : float
            best fitted b
        c_best : float
            best fitted c
        d_best : float
            best fitted d
        r2_best : float
            best r2 (coefficient of determination)
    
    Example
    -------
    >>> num = 100
    >>> x = torch.linspace(-1,1,steps=num)
    >>> noises = torch.normal(0,1,(num,)) * 0.02
    >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
    >>> fit_params(x, y, torch.sin)
    r2 is 0.9999727010726929
    (tensor([2.9982, 1.9996, 5.0053, 0.7011]), tensor(1.0000))
    '''
    # fit a, b, c, d such that y=c*fun(a*x+b)+d; both x and y are 1D array.
    # sweep a and b, choose the best fitted model   
    for _ in range(iteration):
        a_ = torch.linspace(a_range[0], a_range[1], steps=grid_number, device=device)
        b_ = torch.linspace(b_range[0], b_range[1], steps=grid_number, device=device)
        a_grid, b_grid = torch.meshgrid(a_, b_, indexing='ij')
        post_fun = fun(a_grid[None,:,:] * x[:,None,None] + b_grid[None,:,:])
        x_mean = torch.mean(post_fun, dim=[0], keepdim=True)
        y_mean = torch.mean(y, dim=[0], keepdim=True)
        numerator = torch.sum((post_fun - x_mean)*(y-y_mean)[:,None,None], dim=0)**2
        denominator = torch.sum((post_fun - x_mean)**2, dim=0)*torch.sum((y - y_mean)[:,None,None]**2, dim=0)
        r2 = numerator/(denominator+1e-4)
        r2 = torch.nan_to_num(r2)
        
        
        best_id = torch.argmax(r2)
        a_id, b_id = torch.div(best_id, grid_number, rounding_mode='floor'), best_id % grid_number
        
        
        if a_id == 0 or a_id == grid_number - 1 or b_id == 0 or b_id == grid_number - 1:
            if _ == 0 and verbose==True:
                print('Best value at boundary.')
            if a_id == 0:
                a_range = [a_[0], a_[1]]
            if a_id == grid_number - 1:
                a_range = [a_[-2], a_[-1]]
            if b_id == 0:
                b_range = [b_[0], b_[1]]
            if b_id == grid_number - 1:
                b_range = [b_[-2], b_[-1]]
            
        else:
            a_range = [a_[a_id-1], a_[a_id+1]]
            b_range = [b_[b_id-1], b_[b_id+1]]
            
    a_best = a_[a_id]
    b_best = b_[b_id]
    post_fun = fun(a_best * x + b_best)
    r2_best = r2[a_id, b_id]
    
    if verbose == True:
        print(f"r2 is {r2_best}")
        if r2_best < 0.9:
            print(f'r2 is not very high, please double check if you are choosing the correct symbolic function.')

    post_fun = torch.nan_to_num(post_fun)
    reg = LinearRegression().fit(post_fun[:,None].detach().cpu().numpy(), y.detach().cpu().numpy())
    c_best = torch.from_numpy(reg.coef_)[0].to(device)
    d_best = torch.from_numpy(np.array(reg.intercept_)).to(device)
    return torch.stack([a_best, b_best, c_best, d_best]), r2_best


def sparse_mask(in_dim, out_dim):

    in_coord = torch.arange(in_dim) * 1/in_dim + 1/(2*in_dim)
    out_coord = torch.arange(out_dim) * 1/out_dim + 1/(2*out_dim)

    dist_mat = torch.abs(out_coord[:,None] - in_coord[None,:])
    in_nearest = torch.argmin(dist_mat, dim=0)
    in_connection = torch.stack([torch.arange(in_dim), in_nearest]).permute(1,0)
    out_nearest = torch.argmin(dist_mat, dim=1)
    out_connection = torch.stack([out_nearest, torch.arange(out_dim)]).permute(1,0)
    all_connection = torch.cat([in_connection, out_connection], dim=0)
    mask = torch.zeros(in_dim, out_dim)
    mask[all_connection[:,0], all_connection[:,1]] = 1.
    
    return mask


def add_symbolic(name, fun, c=1, fun_singularity=None):
    '''
    add a symbolic function to library
    
    Args:
    -----
        name : str
            name of the function
        fun : fun
            torch function or lambda function
    
    Returns:
    --------
        None
    
    Example
    -------
    >>> print(SYMBOLIC_LIB['Bessel'])
    KeyError: 'Bessel'
    >>> add_symbolic('Bessel', torch.special.bessel_j0)
    >>> print(SYMBOLIC_LIB['Bessel'])
    (<built-in function special_bessel_j0>, Bessel)
    '''
    exec(f"globals()['{name}'] = sympy.Function('{name}')")
    if fun_singularity==None:
        fun_singularity = fun
    SYMBOLIC_LIB[name] = (fun, globals()[name], c, fun_singularity)
    
  
def ex_round(ex1, n_digit):
    ex2 = ex1
    for a in sympy.preorder_traversal(ex1):
        if isinstance(a, sympy.Float):
            ex2 = ex2.subs(a, round(a, n_digit))
    return ex2


def augment_input(orig_vars, aux_vars, x):

    # if x is a tensor
    if isinstance(x, torch.Tensor):

        for aux_var in aux_vars:
            func = lambdify(orig_vars, aux_var,'numpy') # returns a numpy-ready function
            aux_value = torch.from_numpy(func(*[x[:,[i]].numpy() for i in range(len(orig_vars))]))
            x = torch.cat([x, aux_value], dim=1)

    # if x is a dataset
    elif isinstance(x, dict):
        x['train_input'] = augment_input(orig_vars, aux_vars, x['train_input'])
        x['test_input'] = augment_input(orig_vars, aux_vars, x['test_input'])
        
    return x


def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)
    return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph)[0]

def batch_hessian(model, x, create_graph=False):
    # x in shape (Batch, Length)
    jac = lambda x: batch_jacobian(model, x, create_graph=True)
    def _jac_sum(x):
        return jac(x).sum(dim=0)
    return torch.autograd.functional.jacobian(_jac_sum, x, create_graph=create_graph).permute(1,0,2)


def create_dataset_from_data(inputs, labels, train_ratio=0.8, device='cpu'):

    num = inputs.shape[0]
    train_id = np.random.choice(num, int(num*train_ratio), replace=False)
    test_id = list(set(np.arange(num)) - set(train_id))
    dataset = {}
    dataset['train_input'] = inputs[train_id].detach().to(device)
    dataset['test_input'] = inputs[test_id].detach().to(device)
    dataset['train_label'] = labels[train_id].detach().to(device)
    dataset['test_label'] = labels[test_id].detach().to(device)
    
    return dataset