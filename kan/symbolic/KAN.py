import torch
import torch.nn as nn
import numpy as np
import sympy
import copy
from .symbolic_KANLayer import SymbolicKANLayer
from ..utils import SYMBOLIC_LIB

from ..modeling.numeric_KAN import NumericKAN
from ..visualization.visualize import plot


class KAN(NumericKAN):
    '''
    KAN class
    
    Attributes:
    -----------
        biases: a list of nn.Linear()
            biases are added on nodes (in principle, biases can be absorbed into activation functions. However, we still have them for better optimization)
        act_fun: a list of KANLayer
            KANLayers
        depth: int
            depth of KAN
        width: list
            number of neurons in each layer. e.g., [2,5,5,3] means 2D inputs, 5D outputs, with 2 layers of 5 hidden neurons.
        grid: int
            the number of grid intervals
        k: int
            the order of piecewise polynomial
        base_fun: fun
            residual function b(x). an activation function phi(x) = sb_scale * b(x) + sp_scale * spline(x)
        symbolic_fun: a list of Symbolic_KANLayer
            Symbolic_KANLayers
        symbolic_enabled: bool
            If False, the symbolic front is not computed (to save time). Default: True.
    
    Methods:
    --------
        __init__():
            initialize a KAN
        initialize_from_another_model():
            initialize a KAN from another KAN (with the same shape, but potentially different grids)
        update_grid_from_samples():
            update spline grids based on samples
        initialize_grid_from_another_model():
            initalize KAN grids from another KAN
        forward():
            forward
        set_mode():
            set the mode of an activation function: 'n' for numeric, 's' for symbolic, 'ns' for combined (note they are visualized differently in plot(). 'n' as black, 's' as red, 'ns' as purple). 
        fix_symbolic():
            fix an activation function to be symbolic
        suggest_symbolic():
            suggest the symbolic candicates of a numeric spline-based activation function
        lock():
            lock activation functions to share parameters
        unlock():
            unlock locked activations
        get_range():
            get the input and output ranges of an activation function
        plot():
            plot the diagram of KAN
        train():
            train KAN
        prune():
            prune KAN
        remove_edge():
            remove some edge of KAN
        remove_node():
            remove some node of KAN
        auto_symbolic():
            automatically fit all splines to be symbolic functions
        symbolic_formula():
            obtain the symbolic formula of the KAN network
    '''

    def __init__(self, width=None, grid=3, k=3, noise_scale=0.1, noise_scale_base=0.1, base_fun=torch.nn.SiLU(), symbolic_enabled=True, bias_trainable=True, grid_eps=1.0, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True,
                 device='cpu', seed=0):
        '''
        initalize a KAN model
        
        Args:
        -----
            width : list of int
                :math:`[n_0, n_1, .., n_{L-1}]` specify the number of neurons in each layer (including inputs/outputs)
            grid : int
                number of grid intervals. Default: 3.
            k : int
                order of piecewise polynomial. Default: 3.
            noise_scale : float
                initial injected noise to spline. Default: 0.1.
            base_fun : fun
                the residual function b(x). Default: torch.nn.SiLU().
            symbolic_enabled : bool
                compute or skip symbolic computations (for efficiency). By default: True. 
            bias_trainable : bool
                bias parameters are updated or not. By default: True
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,))
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            seed : int
                random seed
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> model = KAN(width=[2,5,1], grid=5, k=3)
        >>> (model.act_fun[0].in_dim, model.act_fun[0].out_dim), (model.act_fun[1].in_dim, model.act_fun[1].out_dim)
        ((2, 5), (5, 1))
        '''
        super(KAN, self).__init__(width=width, grid=grid, k=k, noise_scale=noise_scale, noise_scale_base=noise_scale_base, base_fun=base_fun, bias_trainable=bias_trainable, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable, sb_trainable=sb_trainable,
                 device=device, seed=seed)
        self.symbolic_enabled=symbolic_enabled

        ### initializing the symbolic front ###
        self.symbolic_fun = []
        for l in range(self.depth):
            sb_batch = SymbolicKANLayer(in_dim=width[l], out_dim=width[l + 1])
            self.symbolic_fun.append(sb_batch)

        self.symbolic_fun = nn.ModuleList(self.symbolic_fun)
        self.symbolic_enabled = symbolic_enabled

    def initialize_from_another_model(self, another_model, x):
        '''
        initialize from a parent model. The parent has the same width as the current model but may have different grids.
        
        Args:
        -----
            another_model : KAN
                the parent model used to initialize the current model
            x : 2D torch.float
                inputs, shape (batch, input dimension)
        
        Returns:
        --------
            self : KAN
            
        Example
        -------
        >>> model_coarse = KAN(width=[2,5,1], grid=5, k=3)
        >>> model_fine = KAN(width=[2,5,1], grid=10, k=3)
        >>> print(model_fine.act_fun[0].coef[0][0].data)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model_fine.initialize_from_another_model(model_coarse, x);
        >>> print(model_fine.act_fun[0].coef[0][0].data)
        tensor(-0.0030)
        tensor(0.0506)
        '''
        self = super().initialize_from_another_model(another_model, x)

        for l in range(self.depth):
            self.symbolic_fun[l] = another_model.symbolic_fun[l]

        return self

    def forward(self, x):
        '''
        KAN forward
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (batch, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (batch, output dimension)
            
        Example
        -------
        >>> model = KAN(width=[2,5,3], grid=5, k=3)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model(x).shape
        torch.Size([100, 3])
        '''

        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_std = []
        # self.neurons_scale = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        for l in range(self.depth):

            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)

            x_symbolic, postacts_symbolic = self.symbolic_fun[l](x)

            x = x_numerical + x_symbolic
            postacts = postacts_numerical + postacts_symbolic

            # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
            grid_reshape = self.act_fun[l].grid.reshape(self.width[l + 1], self.width[l], -1)
            input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
            output_range = torch.mean(torch.abs(postacts), dim=0)
            self.acts_scale.append(output_range / input_range)
            self.acts_scale_std.append(torch.std(postacts, dim=0))
            self.spline_preacts.append(preacts.detach())
            self.spline_postacts.append(postacts.detach())
            self.spline_postsplines.append(postspline.detach())

            x = x + self.biases[l].weight
            self.acts.append(x)

        return x

    def lock(self, l, ids):
        '''
        lock ids in the l-th layer to be the same function
        
        Args:
        -----
            l : int
                layer index
            ids : 2D list
                :math:`[[i_1,j_1],[i_2,j_2],...]` set :math:`(l,i_i,j_1), (l,i_2,j_2), ...` to be the same function
        
        Returns:
        --------
            None
         
        Example
        -------
        >>> model = KAN(width=[2,3,1], grid=5, k=3, noise_scale=1.)
        >>> print(model.act_fun[0].weight_sharing.reshape(3,2))
        >>> model.lock(0,[[1,0],[1,1]])
        >>> print(model.act_fun[0].weight_sharing.reshape(3,2))
        tensor([[0, 1],
                [2, 3],
                [4, 5]])
        tensor([[0, 1],
                [2, 1],
                [4, 5]])
        '''
        self.act_fun[l].lock(ids)

    def unlock(self, l, ids):
        '''
        unlock ids in the l-th layer to be the same function
        
        Args:
        -----
            l : int
                layer index
            ids : 2D list)
                [[i1,j1],[i2,j2],...] set (l,ii,j1), (l,i2,j2), ... to be unlocked
            
        Example:
        --------
        >>> model = KAN(width=[2,3,1], grid=5, k=3, noise_scale=1.)
        >>> model.lock(0,[[1,0],[1,1]])
        >>> print(model.act_fun[0].weight_sharing.reshape(3,2))
        >>> model.unlock(0,[[1,0],[1,1]])
        >>> print(model.act_fun[0].weight_sharing.reshape(3,2))
        tensor([[0, 1],
                [2, 1],
                [4, 5]])
        tensor([[0, 1],
                [2, 3],
                [4, 5]])
        '''
        self.act_fun[l].unlock(ids)

    def set_mode(self, l, i, j, mode, mask_n=None):
        '''
        set (l,i,j) activation to have mode 
        
        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
            mode : str
                'n' (numeric) or 's' (symbolic) or 'ns' (combined)
            mask_n : None or float)
                magnitude of the numeric front
        
        Returns:
        --------
            None
        '''
        if mode == "s":
            mask_n = 0.;
            mask_s = 1.
        elif mode == "n":
            mask_n = 1.;
            mask_s = 0.
        elif mode == "sn" or mode == "ns":
            if mask_n == None:
                mask_n = 1.
            else:
                mask_n = mask_n
            mask_s = 1.
        else:
            mask_n = 0.;
            mask_s = 0.

        self.act_fun[l].mask.data[j * self.act_fun[l].in_dim + i] = mask_n
        self.symbolic_fun[l].mask.data[j, i] = mask_s

    def fix_symbolic(self, l, i, j, fun_name, fit_params_bool=True, a_range=(-10, 10), b_range=(-10, 10), verbose=True, random=False):
        '''
        set (l,i,j) activation to be symbolic (specified by fun_name)
        
        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
            fun_name : str
                function name
            fit_params_bool : bool
                obtaining affine parameters through fitting (True) or setting default values (False)
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of b
            verbose : bool
                If True, more information is printed.
            random : bool
                initialize affine parameteres randomly or as [1,0,1,0]
        
        Returns:
        --------
            None or r2 (coefficient of determination)
            
        Example 1 
        ---------
        >>> # when fit_params_bool = False
        >>> model = KAN(width=[2,5,1], grid=5, k=3)
        >>> model.fix_symbolic(0,1,3,'sin',fit_params_bool=False)
        >>> print(model.act_fun[0].mask.reshape(2,5))
        >>> print(model.symbolic_fun[0].mask.reshape(2,5))
        tensor([[1., 1., 1., 1., 1.],
                [1., 1., 0., 1., 1.]])
        tensor([[0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0.]])
                    
        Example 2
        ---------
        >>> # when fit_params_bool = True
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=1.)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model(x) # obtain activations (otherwise model does not have attributes acts)
        >>> model.fix_symbolic(0,1,3,'sin',fit_params_bool=True)
        >>> print(model.act_fun[0].mask.reshape(2,5))
        >>> print(model.symbolic_fun[0].mask.reshape(2,5))
        r2 is 0.8131332993507385
        r2 is not very high, please double check if you are choosing the correct symbolic function.
        tensor([[1., 1., 1., 1., 1.],
                [1., 1., 0., 1., 1.]])
        tensor([[0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0.]])
        '''
        self.set_mode(l, i, j, mode="s")
        if not fit_params_bool:
            self.symbolic_fun[l].fix_symbolic(i, j, fun_name, verbose=verbose, random=random)
            return None
        else:
            x = self.acts[l][:, i]
            y = self.spline_postacts[l][:, j, i]
            r2 = self.symbolic_fun[l].fix_symbolic(i, j, fun_name, x, y, a_range=a_range, b_range=b_range, verbose=verbose)
            return r2

    def unfix_symbolic(self, l, i, j):
        '''
        unfix the (l,i,j) activation function.
        '''
        self.set_mode(l, i, j, mode="n")

    def unfix_symbolic_all(self):
        '''
        unfix all activation functions.
        '''
        for l in range(len(self.width) - 1):
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    self.unfix_symbolic(l, i, j)

    def prune(self, threshold=1e-2, mode="auto", active_neurons_id=None):
        '''
        pruning KAN on the node level. If a node has small incoming or outgoing connection, it will be pruned away.
        
        Args:
        -----
            threshold : float
                the threshold used to determine whether a node is small enough
            mode : str
                "auto" or "manual". If "auto", the thresold will be used to automatically prune away nodes. If "manual", active_neuron_id is needed to specify which neurons are kept (others are thrown away).
            active_neuron_id : list of id lists
                For example, [[0,1],[0,2,3]] means keeping the 0/1 neuron in the 1st hidden layer and the 0/2/3 neuron in the 2nd hidden layer. Pruning input and output neurons is not supported yet.
            
        Returns:
        --------
            model2 : KAN
                pruned model
         
        Example
        -------
        >>> # for more interactive examples, please see demos
        >>> from utils import create_dataset
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.1, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.train(dataset, opt='LBFGS', steps=50, lamb=0.01);
        >>> model.prune()
        >>> model.plot(mask=True)
        '''
        mask = [torch.ones(self.width[0], )]
        active_neurons = [list(range(self.width[0]))]
        for i in range(len(self.acts_scale) - 1):
            if mode == "auto":
                in_important = torch.max(self.acts_scale[i], dim=1)[0] > threshold
                out_important = torch.max(self.acts_scale[i + 1], dim=0)[0] > threshold
                overall_important = in_important * out_important
            elif mode == "manual":
                overall_important = torch.zeros(self.width[i + 1], dtype=torch.bool)
                overall_important[active_neurons_id[i + 1]] = True
            mask.append(overall_important.float())
            active_neurons.append(torch.where(overall_important == True)[0])
        active_neurons.append(list(range(self.width[-1])))
        mask.append(torch.ones(self.width[-1], ))

        self.mask = mask  # this is neuron mask for the whole model

        # update act_fun[l].mask
        for l in range(len(self.acts_scale) - 1):
            for i in range(self.width[l + 1]):
                if i not in active_neurons[l + 1]:
                    self.remove_node(l + 1, i)

        model2 = KAN(copy.deepcopy(self.width), self.grid, self.k, base_fun=self.base_fun)
        model2.load_state_dict(self.state_dict())
        for i in range(len(self.acts_scale)):
            if i < len(self.acts_scale) - 1:
                model2.biases[i].weight.data = model2.biases[i].weight.data[:, active_neurons[i + 1]]

            model2.act_fun[i] = model2.act_fun[i].get_subset(active_neurons[i], active_neurons[i + 1])
            model2.width[i] = len(active_neurons[i])
            model2.symbolic_fun[i] = self.symbolic_fun[i].get_subset(active_neurons[i], active_neurons[i + 1])

        return model2

    def remove_edge(self, l, i, j):
        '''
        remove activtion phi(l,i,j) (set its mask to zero)
        
        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
        
        Returns:
        --------
            None
        '''
        self.act_fun[l].mask[j * self.width[l] + i] = 0.

    def remove_node(self, l, i):
        '''
        remove neuron (l,i) (set the masks of all incoming and outgoing activation functions to zero)
        
        Args:
        -----
            l : int
                layer index
            i : int
                neuron index
        
        Returns:
        --------
            None
        '''
        self.act_fun[l - 1].mask[i * self.width[l - 1] + torch.arange(self.width[l - 1])] = 0.
        self.act_fun[l].mask[torch.arange(self.width[l + 1]) * self.width[l] + i] = 0.
        self.symbolic_fun[l - 1].mask[i, :] *= 0.
        self.symbolic_fun[l].mask[:, i] *= 0.

    def suggest_symbolic(self, l, i, j, a_range=(-10, 10), b_range=(-10, 10), lib=None, topk=5, verbose=True):
        '''suggest the symbolic candidates of phi(l,i,j)
        
        Args:
        -----
            l : int
                layer index
            i : int 
                input neuron index
            j : int 
                output neuron index
            lib : dic
                library of symbolic bases. If lib = None, the global default library will be used. 
            topk : int
                display the top k symbolic functions (according to r2)
            verbose : bool
                If True, more information will be printed.
           
        Returns:
        --------
            None
            
        Example
        -------
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.1, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.train(dataset, opt='LBFGS', steps=50, lamb=0.01);
        >>> model = model.prune()
        >>> model(dataset['train_input'])
        >>> model.suggest_symbolic(0,0,0)
        function , r2
        sin , 0.9994412064552307
        gaussian , 0.9196369051933289
        tanh , 0.8608126044273376
        sigmoid , 0.8578218817710876
        arctan , 0.842217743396759
        '''
        r2s = []

        if lib == None:
            symbolic_lib = SYMBOLIC_LIB
        else:
            symbolic_lib = {}
            for item in lib:
                symbolic_lib[item] = SYMBOLIC_LIB[item]

        for (name, fun) in symbolic_lib.items():
            r2 = self.fix_symbolic(l, i, j, name, a_range=a_range, b_range=b_range, verbose=False)
            r2s.append(r2.item())

        self.unfix_symbolic(l, i, j)

        sorted_ids = np.argsort(r2s)[::-1][:topk]
        r2s = np.array(r2s)[sorted_ids][:topk]
        topk = np.minimum(topk, len(symbolic_lib))
        if verbose == True:
            print('function', ',', 'r2')
            for i in range(topk):
                print(list(symbolic_lib.items())[sorted_ids[i]][0], ',', r2s[i])

        best_name = list(symbolic_lib.items())[sorted_ids[0]][0]
        best_fun = list(symbolic_lib.items())[sorted_ids[0]][1]
        best_r2 = r2s[0]
        return best_name, best_fun, best_r2

    def auto_symbolic(self, a_range=(-10, 10), b_range=(-10, 10), lib=None, verbose=1):
        '''
        automatic symbolic regression: using top 1 suggestion from suggest_symbolic to replace splines with symbolic activations
        
        Args:
        -----
            lib : None or a list of function names
                the symbolic library 
            verbose : int
                verbosity
                
        Returns:
        --------
            None (print suggested symbolic formulas)
            
        Example 1
        ---------
        >>> # default library
        >>> from utils import create_dataset
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.1, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.train(dataset, opt='LBFGS', steps=50, lamb=0.01);
        >>> >>> model = model.prune()
        >>> model(dataset['train_input'])
        >>> model.auto_symbolic()
        fixing (0,0,0) with sin, r2=0.9994837045669556
        fixing (0,1,0) with cosh, r2=0.9978033900260925
        fixing (1,0,0) with arctan, r2=0.9997088313102722
        
        Example 2
        ---------
        >>> # customized library
        >>> from utils import create_dataset
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.1, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.train(dataset, opt='LBFGS', steps=50, lamb=0.01);
        >>> >>> model = model.prune()
        >>> model(dataset['train_input'])
        >>> model.auto_symbolic(lib=['exp','sin','x^2'])
        fixing (0,0,0) with sin, r2=0.999411404132843
        fixing (0,1,0) with x^2, r2=0.9962921738624573
        fixing (1,0,0) with exp, r2=0.9980258941650391
        '''
        for l in range(len(self.width) - 1):
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    if self.symbolic_fun[l].mask[j, i] > 0.:
                        print(f'skipping ({l},{i},{j}) since already symbolic')
                    else:
                        name, fun, r2 = self.suggest_symbolic(l, i, j, a_range=a_range, b_range=b_range, lib=lib, verbose=False)
                        self.fix_symbolic(l, i, j, name, verbose=verbose > 1)
                        if verbose >= 1:
                            print(f'fixing ({l},{i},{j}) with {name}, r2={r2}')

    def symbolic_formula(self, floating_digit=2, var=None, normalizer=None, simplify=False):
        '''
        obtain the symbolic formula
        
        Args:
        -----
            floating_digit : int
                the number of digits to display
            var : list of str
                the name of variables (if not provided, by default using ['x_1', 'x_2', ...])
            normalizer : [mean array (floats), varaince array (floats)]
                the normalization applied to inputs
            simplify : bool
                If True, simplify the equation at each step (usually quite slow), so set up False by default.
            
        Returns:
        --------
            symbolic formula : sympy function
        
        Example
        -------
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.1, seed=0, grid_eps=0.02)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.train(dataset, opt='LBFGS', steps=50, lamb=0.01);
        >>> model = model.prune()
        >>> model(dataset['train_input'])
        >>> model.auto_symbolic(lib=['exp','sin','x^2'])
        >>> model.train(dataset, opt='LBFGS', steps=50, lamb=0.00, update_grid=False);
        >>> model.symbolic_formula()
        '''
        symbolic_acts = []
        x = []

        def ex_round(ex1, floating_digit=floating_digit):
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, floating_digit))
            return ex2

        # define variables
        if var == None:
            for ii in range(1, self.width[0] + 1):
                exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                exec(f"x.append(x{ii})")
        else:
            x = [sympy.symbols(var_) for var_ in var]

        x0 = x

        if normalizer != None:
            mean = normalizer[0]
            std = normalizer[1]
            x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

        symbolic_acts.append(x)

        for l in range(len(self.width) - 1):
            y = []
            for j in range(self.width[l + 1]):
                yj = 0.
                for i in range(self.width[l]):
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]
                    sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]
                    try:
                        yj += c * sympy_fun(a * x[i] + b) + d
                    except:
                        print('make sure all activations need to be converted to symbolic formulas first!')
                        return
                if simplify == True:
                    y.append(sympy.simplify(yj + self.biases[l].weight.data[0, j]))
                else:
                    y.append(yj + self.biases[l].weight.data[0, j])

            x = y
            symbolic_acts.append(x)

        self.symbolic_acts = [[ex_round(symbolic_acts[l][i]) for i in range(len(symbolic_acts[l]))] for l in range(len(symbolic_acts))]

        out_dim = len(symbolic_acts[-1])
        return [ex_round(symbolic_acts[-1][i]) for i in range(len(symbolic_acts[-1]))], x0

    def plot(self, folder="./figures", beta=3, mask=False, mode="supervised", scale=0.5, tick=False, sample=False, in_vars=None, out_vars=None, title=None):
        '''
        plot KAN
        
        Args:
        -----
            folder : str
                the folder to store pngs
            beta : float
                positive number. control the transparency of each activation. transparency = tanh(beta*l1).
            mask : bool
                If True, plot with mask (need to run prune() first to obtain mask). If False (by default), plot all activation functions.
            mode : bool
                "supervised" or "unsupervised". If "supervised", l1 is measured by absolution value (not subtracting mean); if "unsupervised", l1 is measured by standard deviation (subtracting mean).
            scale : float
                control the size of the diagram
            in_vars: None or list of str
                the name(s) of input variables
            out_vars: None or list of str
                the name(s) of output variables
            title: None or str
                title
            
        Returns:
        --------
            Figure
            
        Example
        -------
        >>> # see more interactive examples in demos
        >>> model = KAN(width=[2,3,1], grid=3, k=3, noise_scale=1.0)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model(x) # do a forward pass to obtain model.acts
        >>> model.plot()
        '''
        plot(self, folder=folder, beta=beta, mask=mask, mode=mode, scale=scale, tick=tick, sample=sample, in_vars=in_vars, out_vars=out_vars, title=title)