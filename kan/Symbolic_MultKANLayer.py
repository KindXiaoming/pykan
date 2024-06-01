import torch
import torch.nn as nn
import numpy as np
import sympy
from .utils import *
from .Symbolic_KANLayer import Symbolic_KANLayer


class Symbolic_MultKANLayer(nn.Module):
    
    def __init__(self, in_dim=3, out_dim_sum=2, out_dim_mult=2, device='cpu'):
        
        super(Symbolic_MultKANLayer, self).__init__()
        self.out_dim_sum = out_dim_sum
        self.out_dim_mult = out_dim_mult
        self.symbolic_kanlayer = Symbolic_KANLayer(in_dim=in_dim, out_dim=out_dim_sum+2*out_dim_mult)
        self.device = device
    
    def forward(self, x):
        
        out_dim_sum = self.out_dim_sum
        y, postacts = self.symbolic_kanlayer(x)
        y_mult = y[:,out_dim_sum::2] * y[:,out_dim_sum+1::2]
        y = torch.cat([y[:,:out_dim_sum], y_mult], dim=1)
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
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = fun_name
            if x == None or y == None:
                #initialzie from just fun
                self.funs[j][i] = fun
                if random == False:
                    self.affine.data[j][i] = torch.tensor([1.,0.,1.,0.])
                else:
                    self.affine.data[j][i] = torch.rand(4,) * 2 - 1
                return None
            else:
                #initialize from x & y and fun
                params, r2 = fit_params(x,y,fun, a_range=a_range, b_range=b_range, verbose=verbose, device=self.device)
                self.funs[j][i] = fun
                self.affine.data[j][i] = params
                return r2
        else:
            # if fun_name itself is a function
            fun = fun_name
            fun_sympy = fun_name
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = "anonymous"

            self.funs[j][i] = fun
            if random == False:
                self.affine.data[j][i] = torch.tensor([1.,0.,1.,0.])
            else:
                self.affine.data[j][i] = torch.rand(4,) * 2 - 1
            return None
