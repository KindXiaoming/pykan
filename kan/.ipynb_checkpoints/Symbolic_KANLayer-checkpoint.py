import torch
import torch.nn as nn
import numpy as np
import sympy
from .utils import *



class Symbolic_KANLayer(nn.Module):
    '''
    KANLayer class

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        funs: 2D array of torch functions (or lambda functions)
            symbolic functions (torch)
        funs_name: 2D arry of str
            names of symbolic functions
        funs_sympy: 2D array of sympy functions (or lambda functions)
            symbolic functions (sympy)
        affine: 3D array of floats
            affine transformations of inputs and outputs
        
    Methods:
    --------
        __init__(): 
            initialize a Symbolic_KANLayer
        forward():
            forward
        get_subset():
            get subset of the KANLayer (used for pruning)
        fix_symbolic():
            fix an activation function to be symbolic
    '''
    def __init__(self, in_dim=3, out_dim=2, device='cpu'):
        '''
        initialize a Symbolic_KANLayer (activation functions are initialized to be identity functions)
        
        Args:
        -----
            in_dim : int
                input dimension
            out_dim : int
                output dimension
            device : str
                device
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=3)
        >>> len(sb.funs), len(sb.funs[0])
        (3, 3)
        '''
        super(Symbolic_KANLayer, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mask = torch.nn.Parameter(torch.zeros(out_dim, in_dim, device=device)).requires_grad_(False)
        # torch
        self.funs = [[lambda x: x*0. for i in range(self.in_dim)] for j in range(self.out_dim)]
        self.funs_avoid_singularity = [[lambda x, y_th: ((), x*0.) for i in range(self.in_dim)] for j in range(self.out_dim)]
        # name
        self.funs_name = [['0' for i in range(self.in_dim)] for j in range(self.out_dim)]
        # sympy
        self.funs_sympy = [[lambda x: x*0. for i in range(self.in_dim)] for j in range(self.out_dim)]
        ### make funs_name the only parameter, and make others as the properties of funs_name?
        
        self.affine = torch.nn.Parameter(torch.zeros(out_dim, in_dim, 4, device=device))
        # c*f(a*x+b)+d
        
        self.device = device
        self.to(device)
        
    def to(self, device):
        super(Symbolic_KANLayer, self).to(device)
        self.device = device    
        return self
    
    def forward(self, x, singularity_avoiding=False, y_th=10.):
        '''
        forward
        
        Args:
        -----
            x : 2D array
                inputs, shape (batch, input dimension)
            
        Returns:
        --------
            y : 2D array
                outputs, shape (batch, output dimension)
            postacts : 3D array
                activations after activation functions but before summing on nodes
        
        Example
        -------
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, postacts = sb(x)
        >>> y.shape, postacts.shape
        (torch.Size([100, 5]), torch.Size([100, 5, 3]))
        '''
        
        batch = x.shape[0]
        postacts = []

        for i in range(self.in_dim):
            postacts_ = []
            for j in range(self.out_dim):
                if singularity_avoiding:
                    xij = self.affine[j,i,2]*self.funs_avoid_singularity[j][i](self.affine[j,i,0]*x[:,[i]]+self.affine[j,i,1], torch.tensor(y_th))[1]+self.affine[j,i,3]
                else:
                    xij = self.affine[j,i,2]*self.funs[j][i](self.affine[j,i,0]*x[:,[i]]+self.affine[j,i,1])+self.affine[j,i,3]
                postacts_.append(self.mask[j][i]*xij)
            postacts.append(torch.stack(postacts_))

        postacts = torch.stack(postacts)
        postacts = postacts.permute(2,1,0,3)[:,:,:,0]
        y = torch.sum(postacts, dim=2)
        
        return y, postacts
        
        
    def get_subset(self, in_id, out_id):
        '''
        get a smaller Symbolic_KANLayer from a larger Symbolic_KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
            
        Returns:
        --------
            spb : Symbolic_KANLayer
         
        Example
        -------
        >>> sb_large = Symbolic_KANLayer(in_dim=10, out_dim=10)
        >>> sb_small = sb_large.get_subset([0,9],[1,2,3])
        >>> sb_small.in_dim, sb_small.out_dim
        (2, 3)
        '''
        sbb = Symbolic_KANLayer(self.in_dim, self.out_dim, device=self.device)
        sbb.in_dim = len(in_id)
        sbb.out_dim = len(out_id)
        sbb.mask.data = self.mask.data[out_id][:,in_id]
        sbb.funs = [[self.funs[j][i] for i in in_id] for j in out_id]
        sbb.funs_avoid_singularity = [[self.funs_avoid_singularity[j][i] for i in in_id] for j in out_id]
        sbb.funs_sympy = [[self.funs_sympy[j][i] for i in in_id] for j in out_id]
        sbb.funs_name = [[self.funs_name[j][i] for i in in_id] for j in out_id]
        sbb.affine.data = self.affine.data[out_id][:,in_id]
        return sbb
    
    
    def fix_symbolic(self, i, j, fun_name, x=None, y=None, random=False, a_range=(-10,10), b_range=(-10,10), verbose=True):
        '''
        fix an activation function to be symbolic
        
        Args:
        -----
            i : int
                the id of input neuron
            j : int 
                the id of output neuron
            fun_name : str
                the name of the symbolic functions
            x : 1D array
                preactivations
            y : 1D array
                postactivations
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of a
            verbose : bool
                print more information if True
            
        Returns:
        --------
            r2 (coefficient of determination)
            
        Example 1
        ---------
        >>> # when x & y are not provided. Affine parameters are set to a = 1, b = 0, c = 1, d = 0
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=2)
        >>> sb.fix_symbolic(2,1,'sin')
        >>> print(sb.funs_name)
        >>> print(sb.affine)
        [['', '', ''], ['', '', 'sin']]
        Parameter containing:
        tensor([[0., 0., 0., 0.],
                 [0., 0., 0., 0.],
                 [1., 0., 1., 0.]], requires_grad=True)
        Example 2
        ---------
        >>> # when x & y are provided, fit_params() is called to find the best fit coefficients
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=2)
        >>> batch = 100
        >>> x = torch.linspace(-1,1,steps=batch)
        >>> noises = torch.normal(0,1,(batch,)) * 0.02
        >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
        >>> sb.fix_symbolic(2,1,'sin',x,y)
        >>> print(sb.funs_name)
        >>> print(sb.affine[1,2,:].data)
        r2 is 0.9999701976776123
        [['', '', ''], ['', '', 'sin']]
        tensor([2.9981, 1.9997, 5.0039, 0.6978])
        '''
        if isinstance(fun_name,str):
            fun = SYMBOLIC_LIB[fun_name][0]
            fun_sympy = SYMBOLIC_LIB[fun_name][1]
            fun_avoid_singularity = SYMBOLIC_LIB[fun_name][3]
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = fun_name
            
            if x == None or y == None:
                #initialzie from just fun
                self.funs[j][i] = fun
                self.funs_avoid_singularity[j][i] = fun_avoid_singularity
                if random == False:
                    self.affine.data[j][i] = torch.tensor([1.,0.,1.,0.])
                else:
                    self.affine.data[j][i] = torch.rand(4,) * 2 - 1
                return None
            else:
                #initialize from x & y and fun
                params, r2 = fit_params(x,y,fun, a_range=a_range, b_range=b_range, verbose=verbose, device=self.device)
                self.funs[j][i] = fun
                self.funs_avoid_singularity[j][i] = fun_avoid_singularity
                self.affine.data[j][i] = params
                return r2
        else:
            # if fun_name itself is a function
            fun = fun_name
            fun_sympy = fun_name
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = "anonymous"

            self.funs[j][i] = fun
            self.funs_avoid_singularity[j][i] = fun
            if random == False:
                self.affine.data[j][i] = torch.tensor([1.,0.,1.,0.])
            else:
                self.affine.data[j][i] = torch.rand(4,) * 2 - 1
            return None
        
    def swap(self, i1, i2, mode='in'):

        with torch.no_grad():
            def swap_list_(data, i1, i2, mode='in'):

                if mode == 'in':
                    for j in range(self.out_dim):
                        data[j][i1], data[j][i2] = data[j][i2], data[j][i1]

                elif mode == 'out':
                    data[i1], data[i2] = data[i2], data[i1] 

            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[:,i1], data[:,i2] = data[:,i2].clone(), data[:,i1].clone()

                elif mode == 'out':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()

            swap_list_(self.funs_name,i1,i2,mode)
            swap_list_(self.funs_sympy,i1,i2,mode)
            swap_list_(self.funs_avoid_singularity,i1,i2,mode)
            swap_(self.affine.data,i1,i2,mode)
            swap_(self.mask.data,i1,i2,mode)
