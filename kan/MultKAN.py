import torch
import torch.nn as nn
import numpy as np
from .KANLayer import KANLayer
#from .Symbolic_MultKANLayer import *
from .Symbolic_KANLayer import Symbolic_KANLayer
from .LBFGS import *
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
#from .MultKANLayer import MultKANLayer
import pandas as pd
from sympy.printing import latex
from sympy import *
import sympy
import yaml
from .spline import curve2coef
from .utils import SYMBOLIC_LIB
from .hypothesis import plot_tree

class MultKAN(nn.Module):

    # include mult_ops = []
    def __init__(self, width=None, grid=3, k=3, mult_arity = 2, noise_scale=1.0, scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu', symbolic_enabled=True, affine_trainable=False, grid_eps=1.0, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, seed=1, save_act=True, sparse_init=False, auto_save=True, first_init=True, ckpt_path='./model', state_id=0, round=0):
        
        super(MultKAN, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ### initializeing the numerical front ###

        self.act_fun = []
        self.depth = len(width) - 1
        
        for i in range(len(width)):
            if type(width[i]) == int:
                width[i] = [width[i],0]
            
        self.width = width
        
        # if mult_arity is just a scalar, we extend it to a list of lists
        # e.g, mult_arity = [[2,3],[4]] means that in the first hidden layer, 2 mult ops have arity 2 and 3, respectively;
        # in the second hidden layer, 1 mult op has arity 4.
        if isinstance(mult_arity, int):
            self.mult_homo = True # when homo is True, parallelization is possible
        else:
            self.mult_homo = False # when home if False, for loop is required. 
        self.mult_arity = mult_arity

        width_in = self.width_in
        width_out = self.width_out
        
        self.base_fun_name = base_fun
        if base_fun == 'silu':
            base_fun = torch.nn.SiLU()
        elif base_fun == 'identity':
            base_fun = torch.nn.Identity()
            
        self.grid_eps = grid_eps
        self.grid_range = grid_range
            
        
        for l in range(self.depth):
            # splines
            scale_base = scale_base_mu * 1 / np.sqrt(width_in[l]) + \
                         scale_base_sigma * (torch.randn(width_in[l], width_out[l + 1]) * 2 - 1) * 1/np.sqrt(width_in[l])
            sp_batch = KANLayer(in_dim=width_in[l], out_dim=width_out[l+1], num=grid, k=k, noise_scale=noise_scale, scale_base=scale_base, scale_sp=1., base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable, sb_trainable=sb_trainable, sparse_init=sparse_init)
            self.act_fun.append(sp_batch)

        self.node_bias = []
        self.node_scale = []
        self.subnode_bias = []
        self.subnode_scale = []
        
        globals()['self.node_bias_0'] = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)
        #self.node_bias_0 = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)
        exec('self.node_bias_0' + " = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)")
        
        for l in range(self.depth):
            exec(f'self.node_bias_{l} = torch.nn.Parameter(torch.zeros(width_in[l+1],)).requires_grad_(affine_trainable)')
            exec(f'self.node_scale_{l} = torch.nn.Parameter(torch.ones(width_in[l+1],)).requires_grad_(affine_trainable)')
            exec(f'self.subnode_bias_{l} = torch.nn.Parameter(torch.zeros(width_out[l+1],)).requires_grad_(affine_trainable)')
            exec(f'self.subnode_scale_{l} = torch.nn.Parameter(torch.ones(width_out[l+1],)).requires_grad_(affine_trainable)')
            exec(f'self.node_bias.append(self.node_bias_{l})')
            exec(f'self.node_scale.append(self.node_scale_{l})')
            exec(f'self.subnode_bias.append(self.subnode_bias_{l})')
            exec(f'self.subnode_scale.append(self.subnode_scale_{l})')
            
        
        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid = grid
        self.k = k
        self.base_fun = base_fun

        ### initializing the symbolic front ###
        self.symbolic_fun = []
        for l in range(self.depth):
            sb_batch = Symbolic_KANLayer(in_dim=width_in[l], out_dim=width_out[l+1])
            self.symbolic_fun.append(sb_batch)

        self.symbolic_fun = nn.ModuleList(self.symbolic_fun)
        self.symbolic_enabled = symbolic_enabled
        self.affine_trainable = affine_trainable
        self.sp_trainable = sp_trainable
        self.sb_trainable = sb_trainable
        
        self.save_act = save_act
            
        self.node_scores = None
        self.edge_scores = None
        self.subnode_scores = None
        
        self.cache_data = None
        self.acts = None
        
        self.auto_save = auto_save
        self.state_id = 0
        self.ckpt_path = ckpt_path
        self.round = round
        
        if auto_save:
            if first_init:
                if not os.path.exists(ckpt_path):
                    # Create the directory
                    os.makedirs(ckpt_path)
                print(f"checkpoint directory created: {ckpt_path}")
                print('saving model version 0.0')

                history_path = self.ckpt_path+'/history.txt'
                with open(history_path, 'w') as file:
                    file.write(f'### Round {self.round} ###' + '\n')
                    file.write('init => 0.0' + '\n')
                self.saveckpt(path=self.ckpt_path+'/'+'0.0')
            else:
                self.state_id = state_id
            
        self.input_id = torch.arange(self.width_in[0],)

    def initialize_from_another_model(self, another_model, x):
        another_model(x)  # get activations
        batch = x.shape[0]

        self.initialize_grid_from_another_model(another_model, x)

        for l in range(self.depth):
            spb = self.act_fun[l]
            #spb_parent = another_model.act_fun[l]

            # spb = spb_parent
            preacts = another_model.spline_preacts[l]
            postsplines = another_model.spline_postsplines[l]
            #self.act_fun[l].coef.data = curve2coef(preacts[:,0,:], postsplines.permute(0,2,1), spb.grid, k=spb.k)
            self.act_fun[l].scale_base.data = another_model.act_fun[l].scale_base.data
            self.act_fun[l].scale_sp.data = another_model.act_fun[l].scale_sp.data
            self.act_fun[l].mask.data = another_model.act_fun[l].mask.data

        for l in range(self.depth):
            self.node_bias[l].data = another_model.node_bias[l].data
            self.node_scale[l].data = another_model.node_scale[l].data
            
            self.subnode_bias[l].data = another_model.subnode_bias[l].data
            self.subnode_scale[l].data = another_model.subnode_scale[l].data

        for l in range(self.depth):
            self.symbolic_fun[l] = another_model.symbolic_fun[l]

        return self
    
    def log_history(self, method_name): 

        if self.auto_save:

            # save to log file
            #print(func.__name__)
            with open(self.ckpt_path+'/history.txt', 'a') as file:
                file.write(str(self.round)+'.'+str(self.state_id)+' => '+ method_name + ' => ' + str(self.round)+'.'+str(self.state_id+1) + '\n')

            # update state_id
            self.state_id += 1

            # save to ckpt
            self.saveckpt(path=self.ckpt_path+'/'+str(self.round)+'.'+str(self.state_id))
            print('saving model version '+str(self.round)+'.'+str(self.state_id))

    
    def refine(self, new_grid):

        model_new = MultKAN(width=self.width, 
                     grid=new_grid, 
                     k=self.k, 
                     mult_arity=self.mult_arity, 
                     base_fun=self.base_fun_name, 
                     symbolic_enabled=self.symbolic_enabled, 
                     affine_trainable=self.affine_trainable, 
                     grid_eps=self.grid_eps, 
                     grid_range=self.grid_range, 
                     sp_trainable=self.sp_trainable,
                     sb_trainable=self.sb_trainable,
                     ckpt_path=self.ckpt_path,
                     auto_save=True,
                     first_init=False,
                     state_id=self.state_id,
                     round=self.round)
            
        model_new.initialize_from_another_model(self, self.cache_data)
        model_new.cache_data = self.cache_data
        model_new.grid = new_grid
        
        self.log_history('refine')
        model_new.state_id += 1
        
        return model_new
    
    
    def saveckpt(self, path='model'):
    
        model = self
        
        dic = dict(
            width = model.width,
            grid = model.grid,
            k = model.k,
            mult_arity = model.mult_arity,
            base_fun_name = model.base_fun_name,
            symbolic_enabled = model.symbolic_enabled,
            affine_trainable = model.affine_trainable,
            grid_eps = model.grid_eps,
            grid_range = model.grid_range,
            sp_trainable = model.sp_trainable,
            sb_trainable = model.sb_trainable,
            state_id = model.state_id,
            auto_save = model.auto_save,
            ckpt_path = model.ckpt_path,
            round = model.round
        )

        for i in range (model.depth):
            dic[f'symbolic.funs_name.{i}'] = model.symbolic_fun[i].funs_name

        with open(f'{path}_config.yml', 'w') as outfile:
            yaml.dump(dic, outfile, default_flow_style=False)

        torch.save(model.state_dict(), f'{path}_state')
        torch.save(model.cache_data, f'{path}_cache_data')
    
    @staticmethod
    def loadckpt(path='model'):
        with open(f'{path}_config.yml', 'r') as stream:
            config = yaml.safe_load(stream)

        state = torch.load(f'{path}_state')

        model_load = MultKAN(width=config['width'], 
                     grid=config['grid'], 
                     k=config['k'], 
                     mult_arity = config['mult_arity'], 
                     base_fun=config['base_fun_name'], 
                     symbolic_enabled=config['symbolic_enabled'], 
                     affine_trainable=config['affine_trainable'], 
                     grid_eps=config['grid_eps'], 
                     grid_range=config['grid_range'], 
                     sp_trainable=config['sp_trainable'],
                     sb_trainable=config['sb_trainable'],
                     state_id=config['state_id'],
                     auto_save=config['auto_save'],
                     first_init=False,
                     ckpt_path=config['ckpt_path'],
                     round = config['round']+1)

        model_load.load_state_dict(state)
        model_load.cache_data = torch.load(f'{path}_cache_data')
        
        depth = len(model_load.width) - 1
        for l in range(depth):
            out_dim = model_load.symbolic_fun[l].out_dim
            in_dim = model_load.symbolic_fun[l].in_dim
            funs_name = config[f'symbolic.funs_name.{l}']
            for j in range(out_dim):
                for i in range(in_dim):
                    fun_name = funs_name[j][i]
                    model_load.symbolic_fun[l].funs_name[j][i] = fun_name
                    model_load.symbolic_fun[l].funs[j][i] = SYMBOLIC_LIB[fun_name][0]
                    model_load.symbolic_fun[l].funs_sympy[j][i] = SYMBOLIC_LIB[fun_name][1]
                    model_load.symbolic_fun[l].funs_avoid_singularity[j][i] = SYMBOLIC_LIB[fun_name][3]
        return model_load
    
    def copy(self):
        path='copy_temp'
        self.saveckpt(path)
        return KAN.loadckpt(path)
    
    def rewind(self, model_id):
        
        self.round += 1
        self.state_id = model_id.split('.')[-1]
        
        history_path = self.ckpt_path+'/history.txt'
        with open(history_path, 'a') as file:
            file.write(f'### Round {self.round} ###' + '\n')

        self.saveckpt(path=self.ckpt_path+'/'+f'{self.round}.{self.state_id}')
        
        print('rewind to model version '+f'{self.round-1}.{self.state_id}'+', renamed as '+f'{self.round}.{self.state_id}')

        return MultKAN.loadckpt(path=self.ckpt_path+'/'+str(model_id))
    
    
    def checkout(self, model_id):
        return MultKAN.loadckpt(path=self.ckpt_path+'/'+str(model_id))

    @property
    def width_in(self):
        width = self.width
        width_in = [width[l][0]+width[l][1] for l in range(len(width))]
        return width_in
        
    @property
    def width_out(self):
        width = self.width
        if self.mult_homo == True:
            width_out = [width[l][0]+self.mult_arity*width[l][1] for l in range(len(width))]
        else:
            width_out = [width[l][0]+int(np.sum(self.mult_arity[l])) for l in range(len(width))]
        return width_out
    
    @property
    def n_sum(self):
        width = self.width
        n_sum = [width[l][0] for l in range(1,len(width)-1)]
        return n_sum
    
    @property
    def n_mult(self):
        width = self.width
        n_mult = [width[l][1] for l in range(1,len(width)-1)]
        return n_mult
    
    @property
    def feature_score(self):
        self.attribute()
        if self.node_scores == None:
            return None
        else:
            return self.node_scores[0]
    
    def update_grid_from_samples(self, x):
        for l in range(self.depth):
            self.get_act(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])
            
    def update_grid(self, x):
        self.update_grid_from_samples(x)

    def initialize_grid_from_another_model(self, model, x):
        model(x)
        for l in range(self.depth):
            self.act_fun[l].initialize_grid_from_parent(model.act_fun[l], model.acts[l])

    def forward(self, x, singularity_avoiding=False, y_th=10.):
        
        assert x.shape[1] == self.width_in[0]
        
        x = x[:,self.input_id.long()]
        
        # cache data
        self.cache_data = x
        
        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.acts_premult = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_spline = []
        self.subnode_actscale = []
        self.edge_actscale = []
        # self.neurons_scale = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        for l in range(self.depth):
            
            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)
            #print(preacts, postacts_numerical, postspline)
            
            if self.symbolic_enabled == True:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](x, singularity_avoiding=singularity_avoiding, y_th=y_th)
            else:
                x_symbolic = 0.
                postacts_symbolic = 0.

            x = x_numerical + x_symbolic
            
            if self.save_act:
                # save subnode_scale
                self.subnode_actscale.append(torch.std(x, dim=0).detach())
            
            # subnode affine transform
            x = self.subnode_scale[l][None,:] * x + self.subnode_bias[l][None,:]
            
            if self.save_act:
                postacts = postacts_numerical + postacts_symbolic

                # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
                #grid_reshape = self.act_fun[l].grid.reshape(self.width_out[l + 1], self.width_in[l], -1)
                input_range = torch.std(preacts, dim=0) + 0.1
                output_range_spline = torch.std(postacts_numerical, dim=0) # for training, only penalize the spline part
                output_range = torch.std(postacts, dim=0) # for visualization, include the contribution from both spline + symbolic
                # save edge_scale
                self.edge_actscale.append(output_range)
                
                self.acts_scale.append((output_range / input_range).detach())
                self.acts_scale_spline.append(output_range_spline / input_range)
                self.spline_preacts.append(preacts.detach())
                self.spline_postacts.append(postacts.detach())
                self.spline_postsplines.append(postspline.detach())

                self.acts_premult.append(x.detach())
            
            # multiplication
            dim_sum = self.width[l+1][0]
            dim_mult = self.width[l+1][1]
            
            if self.mult_homo == True:
                for i in range(self.mult_arity-1):
                    if i == 0:
                        x_mult = x[:,dim_sum::self.mult_arity] * x[:,dim_sum+1::self.mult_arity]
                    else:
                        x_mult = x_mult * x[:,dim_sum+i+1::self.mult_arity]
                        
            else:
                for j in range(dim_mult):
                    acml_id = dim_sum + np.sum(self.mult_arity[l+1][:j])
                    for i in range(self.mult_arity[l+1][j]-1):
                        if i == 0:
                            x_mult_j = x[:,[acml_id]] * x[:,[acml_id+1]]
                        else:
                            x_mult_j = x_mult_j * x[:,[acml_id+i+1]]
                            
                    if j == 0:
                        x_mult = x_mult_j
                    else:
                        x_mult = torch.cat([x_mult, x_mult_j], dim=1)
                
            if self.width[l+1][1] > 0:
                x = torch.cat([x[:,:dim_sum], x_mult], dim=1)
            
            # x = x + self.biases[l].weight
            # node affine transform
            x = self.node_scale[l][None,:] * x + self.node_bias[l][None,:]
            
            self.acts.append(x.detach())
            
        
        return x

    def set_mode(self, l, i, j, mode, mask_n=None):
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

        self.act_fun[l].mask.data[i][j] = mask_n
        self.symbolic_fun[l].mask.data[j,i] = mask_s

    def fix_symbolic(self, l, i, j, fun_name, fit_params_bool=True, a_range=(-10, 10), b_range=(-10, 10), verbose=True, random=False, log_history=True):
        if not fit_params_bool:
            self.symbolic_fun[l].fix_symbolic(i, j, fun_name, verbose=verbose, random=random)
            r2 = None
        else:
            x = self.acts[l][:, i]
            mask = self.act_fun[l].mask
            y = self.spline_postacts[l][:, j, i]
            #y = self.postacts[l][:, j, i]
            r2 = self.symbolic_fun[l].fix_symbolic(i, j, fun_name, x, y, a_range=a_range, b_range=b_range, verbose=verbose)
            if mask[i,j] == 0:
                r2 = - 1e8
        self.set_mode(l, i, j, mode="s")
        
        if log_history:
            self.log_history('fix_symbolic')
        return r2

    def unfix_symbolic(self, l, i, j, log_history=True):
        self.set_mode(l, i, j, mode="n")
        self.symbolic_fun[l].funs_name[j][i] = "0"
        if log_history:
            self.log_history('unfix_symbolic')

    def unfix_symbolic_all(self):
        for l in range(len(self.width) - 1):
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    self.unfix_symbolic(l, i, j)

    def get_range(self, l, i, j, verbose=True):
        x = self.spline_preacts[l][:, j, i]
        y = self.spline_postacts[l][:, j, i]
        x_min = torch.min(x)
        x_max = torch.max(x)
        y_min = torch.min(y)
        y_max = torch.max(y)
        if verbose:
            print('x range: [' + '%.2f' % x_min, ',', '%.2f' % x_max, ']')
            print('y range: [' + '%.2f' % y_min, ',', '%.2f' % y_max, ']')
        return x_min, x_max, y_min, y_max

    def plot(self, folder="./figures", beta=3, mask=False, metric='backward', scale=0.5, tick=False, sample=False, in_vars=None, out_vars=None, title=None, varscale=1.0):
        
        global Symbol
        
        if not self.save_act:
            print('cannot plot since data are not saved. Set save_act=True first.')
        
        # forward to obtain activations
        if self.acts == None:
            if self.cache_data == None:
                raise Exception('model hasn\'t seen any data yet.')
            self.forward(self.cache_data)
            
        if metric == 'backward':
            self.attribute()
            
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        # matplotlib.use('Agg')
        depth = len(self.width) - 1
        for l in range(depth):
            w_large = 2.0
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l+1]):
                    rank = torch.argsort(self.acts[l][:, i])
                    fig, ax = plt.subplots(figsize=(w_large, w_large))

                    num = rank.shape[0]

                    #print(self.width_in[l])
                    #print(self.width_out[l+1])
                    symbolic_mask = self.symbolic_fun[l].mask[j][i]
                    numeric_mask = self.act_fun[l].mask[i][j]
                    if symbolic_mask > 0. and numeric_mask > 0.:
                        color = 'purple'
                        alpha_mask = 1
                    if symbolic_mask > 0. and numeric_mask == 0.:
                        color = "red"
                        alpha_mask = 1
                    if symbolic_mask == 0. and numeric_mask > 0.:
                        color = "black"
                        alpha_mask = 1
                    if symbolic_mask == 0. and numeric_mask == 0.:
                        color = "white"
                        alpha_mask = 0
                        

                    if tick == True:
                        ax.tick_params(axis="y", direction="in", pad=-22, labelsize=50)
                        ax.tick_params(axis="x", direction="in", pad=-15, labelsize=50)
                        x_min, x_max, y_min, y_max = self.get_range(l, i, j, verbose=False)
                        plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                        plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    if alpha_mask == 1:
                        plt.gca().patch.set_edgecolor('black')
                    else:
                        plt.gca().patch.set_edgecolor('white')
                    plt.gca().patch.set_linewidth(1.5)
                    # plt.axis('off')

                    plt.plot(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5)
                    if sample == True:
                        plt.scatter(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, s=400 * scale ** 2)
                    plt.gca().spines[:].set_color(color)

                    '''lock_id = self.act_fun[l].lock_id[j * self.width[l] + i].long().item()
                    if lock_id > 0:
                        im = plt.imread(f'{folder}/lock.png')
                        newax = fig.add_axes([0.15, 0.7, 0.15, 0.15])
                        plt.text(500, 400, lock_id, fontsize=15)
                        newax.imshow(im)
                        newax.axis('off')'''

                    plt.savefig(f'{folder}/sp_{l}_{i}_{j}.png', bbox_inches="tight", dpi=400)
                    plt.close()

        def score2alpha(score):
            return np.tanh(beta * score)

        
        if metric == 'forward_n':
            scores = self.acts_scale
        elif metric == 'forward_u':
            scores = self.edge_actscale
        elif metric == 'backward':
            scores = self.edge_scores
        else:
            raise Exception(f'metric = \'{metric}\' not recognized')
        
        alpha = [score2alpha(score.cpu().detach().numpy()) for score in scores]
            
        # draw skeleton
        width = np.array(self.width)
        width_in = np.array(self.width_in)
        width_out = np.array(self.width_out)
        A = 1
        y0 = 0.3  # height: from input to pre-mult
        z0 = 0.1  # height: from pre-mult to post-mult (input of next layer)

        neuron_depth = len(width)
        min_spacing = A / np.maximum(np.max(width_out), 5)

        max_neuron = np.max(width_out)
        max_num_weights = np.max(width_in[:-1] * width_out[1:])
        y1 = 0.4 / np.maximum(max_num_weights, 5) # size (height/width) of 1D function diagrams
        y2 = 0.15 / np.maximum(max_neuron, 5) # size (height/width) of operations (sum and mult)

        fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale * (neuron_depth - 1) * (y0+z0)))
        # fig, ax = plt.subplots(figsize=(5,5*(neuron_depth-1)*y0))

        # -- Transformation functions
        DC_to_FC = ax.transData.transform
        FC_to_NFC = fig.transFigure.inverted().transform
        # -- Take data coordinates and transform them to normalized figure coordinates
        DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))
        
        # plot scatters and lines
        for l in range(neuron_depth):
            
            n = width_in[l]
            
            # scatters
            for i in range(n):
                plt.scatter(1 / (2 * n) + i / n, l * (y0+z0), s=min_spacing ** 2 * 10000 * scale ** 2, color='black')
                
            # plot connections (input to pre-mult)
            for i in range(n):
                if l < neuron_depth - 1:
                    n_next = width_out[l+1]
                    N = n * n_next
                    for j in range(n_next):
                        id_ = i * n_next + j

                        symbol_mask = self.symbolic_fun[l].mask[j][i]
                        numerical_mask = self.act_fun[l].mask[i][j]
                        if symbol_mask == 1. and numerical_mask > 0.:
                            color = 'purple'
                            alpha_mask = 1.
                        if symbol_mask == 1. and numerical_mask == 0.:
                            color = "red"
                            alpha_mask = 1.
                        if symbol_mask == 0. and numerical_mask == 1.:
                            color = "black"
                            alpha_mask = 1.
                        if symbol_mask == 0. and numerical_mask == 0.:
                            color = "white"
                            alpha_mask = 0.
                            
                            
                        if mask == True:
                            plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * (y0+z0), l * (y0+z0) + y0/2 - y1], color=color, lw=2 * scale, alpha=alpha[l][j][i] * self.mask[l][i].item() * self.mask[l + 1][j].item())
                            plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [l * (y0+z0) + y0/2 + y1, l * (y0+z0)+y0], color=color, lw=2 * scale, alpha=alpha[l][j][i] * self.mask[l][i].item() * self.mask[l + 1][j].item())
                        else:
                            plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * (y0+z0), l * (y0+z0) + y0/2 - y1], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)
                            plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [l * (y0+z0) + y0/2 + y1, l * (y0+z0)+y0], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)
                            
                            
            # plot connections (pre-mult to post-mult, post-mult = next-layer input)
            if l < neuron_depth - 1:
                n_in = width_out[l+1]
                n_out = width_in[l+1]
                mult_id = 0
                for i in range(n_in):
                    if i < width[l+1][0]:
                        j = i
                    else:
                        if i == width[l+1][0]:
                            if isinstance(self.mult_arity,int):
                                ma = self.mult_arity
                            else:
                                ma = self.mult_arity[l+1][mult_id]
                            current_mult_arity = ma
                        if current_mult_arity == 0:
                            mult_id += 1
                            if isinstance(self.mult_arity,int):
                                ma = self.mult_arity
                            else:
                                ma = self.mult_arity[l+1][mult_id]
                            current_mult_arity = ma
                        j = width[l+1][0] + mult_id
                        current_mult_arity -= 1
                        #j = (i-width[l+1][0])//self.mult_arity + width[l+1][0]
                    plt.plot([1 / (2 * n_in) + i / n_in, 1 / (2 * n_out) + j / n_out], [l * (y0+z0) + y0, (l+1) * (y0+z0)], color='black', lw=2 * scale)

                    
                    
            plt.xlim(0, 1)
            plt.ylim(-0.1 * (y0+z0), (neuron_depth - 1 + 0.1) * (y0+z0))


        plt.axis('off')

        for l in range(neuron_depth - 1):
            # plot splines
            n = width_in[l]
            for i in range(n):
                n_next = width_out[l + 1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j
                    im = plt.imread(f'{folder}/sp_{l}_{i}_{j}.png')
                    left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                    right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                    bottom = DC_to_NFC([0, l * (y0+z0) + y0/2 - y1])[1]
                    up = DC_to_NFC([0, l * (y0+z0) + y0/2 + y1])[1]
                    newax = fig.add_axes([left, bottom, right - left, up - bottom])
                    # newax = fig.add_axes([1/(2*N)+id_/N-y1, (l+1/2)*y0-y1, y1, y1], anchor='NE')
                    if mask == False:
                        newax.imshow(im, alpha=alpha[l][j][i])
                    else:
                        ### make sure to run model.prune_node() first to compute mask ###
                        newax.imshow(im, alpha=alpha[l][j][i] * self.mask[l][i].item() * self.mask[l + 1][j].item())
                    newax.axis('off')
                    
              
            # plot sum symbols
            N = n = width_out[l+1]
            for j in range(n):
                id_ = j
                path = os.path.dirname(os.path.abspath(__file__)) + "/assets/img/sum_symbol.png"
                im = plt.imread(path)
                left = DC_to_NFC([1 / (2 * N) + id_ / N - y2, 0])[0]
                right = DC_to_NFC([1 / (2 * N) + id_ / N + y2, 0])[0]
                bottom = DC_to_NFC([0, l * (y0+z0) + y0 - y2])[1]
                up = DC_to_NFC([0, l * (y0+z0) + y0 + y2])[1]
                newax = fig.add_axes([left, bottom, right - left, up - bottom])
                newax.imshow(im)
                newax.axis('off')
                
            # plot mult symbols
            N = n = width_in[l+1]
            n_sum = width[l+1][0]
            n_mult = width[l+1][1]
            for j in range(n_mult):
                id_ = j + n_sum
                path = os.path.dirname(os.path.abspath(__file__)) + "/assets/img/mult_symbol.png"
                im = plt.imread(path)
                left = DC_to_NFC([1 / (2 * N) + id_ / N - y2, 0])[0]
                right = DC_to_NFC([1 / (2 * N) + id_ / N + y2, 0])[0]
                bottom = DC_to_NFC([0, (l+1) * (y0+z0) - y2])[1]
                up = DC_to_NFC([0, (l+1) * (y0+z0) + y2])[1]
                newax = fig.add_axes([left, bottom, right - left, up - bottom])
                newax.imshow(im)
                newax.axis('off')

        if in_vars != None:
            n = self.width_in[0]
            for i in range(n):
                if isinstance(in_vars[i], sympy.Expr):
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, f'${latex(in_vars[i])}$', fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
                else:
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, in_vars[i], fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
                
                

        if out_vars != None:
            n = self.width_in[-1]
            for i in range(n):
                if isinstance(out_vars[i], sympy.Expr):
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0+z0) * (len(self.width) - 1) + 0.15, f'${latex(out_vars[i])}$', fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
                else:
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0+z0) * (len(self.width) - 1) + 0.15, out_vars[i], fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')

        if title != None:
            plt.gcf().get_axes()[0].text(0.5, (y0+z0) * (len(self.width) - 1) + 0.3, title, fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')

            
    def reg(self, reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff):

        if reg_metric == 'edge_forward_n':
            acts_scale = self.acts_scale_spline
            
        elif reg_metric == 'edge_forward_u':
            acts_scale = self.edge_actscale
            
        elif reg_metric == 'edge_backward':
            acts_scale = self.edge_scores
            
        elif reg_metric == 'node_backward':
            acts_scale = self.node_attribute_scores
            
        else:
            raise Exception(f'reg_metric = {reg_metric} not recognized!')
        
        reg_ = 0.
        for i in range(len(acts_scale)):
            vec = acts_scale[i]

            l1 = torch.sum(vec)
            p_row = vec / (torch.sum(vec, dim=1, keepdim=True) + 1)
            p_col = vec / (torch.sum(vec, dim=0, keepdim=True) + 1)
            entropy_row = - torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1))
            entropy_col = - torch.mean(torch.sum(p_col * torch.log2(p_col + 1e-4), dim=0))
            #entropy_row = torch.max(-torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1))
            #entropy_col = torch.max(-torch.sum(p_col * torch.log2(p_col + 1e-4), dim=0))
            reg_ += lamb_l1 * l1 + lamb_entropy * (entropy_row + entropy_col)  # both l1 and entropy
            '''vec = vec.reshape(-1,)
            p = vec / (torch.sum(vec) + 1e-4)
            entropy = - torch.sum(p * torch.log2(p + 1e-4))
            reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy'''

        # regularize coefficient to encourage spline to be zero
        for i in range(len(self.act_fun)):
            coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
            coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_
    
    def get_reg(self, reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff):
        return self.reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
    
    def disable_symbolic_in_fit(self):
                    
        # skip symbolic if no symbolic is turned on
        depth = len(self.symbolic_fun)
        no_symbolic = True
        for l in range(depth):
            no_symbolic *= torch.sum(torch.abs(self.symbolic_fun[l].mask)) == 0

        old_symbolic_enabled = self.symbolic_enabled

        if no_symbolic:
            self.symbolic_enabled = False
            
        return old_symbolic_enabled
    
    def disable_save_act_in_fit(self, lamb):
        
        old_save_act = self.save_act
        if lamb == 0.:
            self.save_act = False
                      
        return old_save_act
    
    def recover_symbolic_in_fit(self, old_symbolic_enabled):
        self.symbolic_enabled = old_symbolic_enabled
        
    def recover_save_act_in_fit(self, old_save_act):
        if old_save_act == True:
            self.save_act = True
    
    def get_params(self):
        return self.parameters()
        
            
    def fit(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1.,start_grid_update_step=-1, stop_grid_update_step=50, batch=-1,
              metrics=None, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', singularity_avoiding=False, y_th=1000., reg_metric='edge_backward', display_metrics=None):

        if lamb > 0. and not self.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')
            
        old_save_act = self.disable_save_act_in_fit(lamb)
        old_symbolic_enabled = self.disable_symbolic_in_fit()

        pbar = tqdm(range(steps), desc='description', ncols=100)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.get_params(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
            train_loss = loss_fn(pred, dataset['train_label'][train_id])
            if self.save_act:
                if reg_metric == 'edge_backward':
                    self.attribute()
                if reg_metric == 'node_backward':
                    self.node_attribute()
                reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            else:
                reg_ = torch.tensor(0.)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        for _ in pbar:
            
            if _ == steps-1:
                self.recover_save_act_in_fit(old_save_act)
            
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
                self.update_grid(dataset['train_input'][train_id])

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
                train_loss = loss_fn(pred, dataset['train_label'][train_id])
                if self.save_act:
                    if reg_metric == 'edge_backward':
                        self.attribute()
                    if reg_metric == 'node_backward':
                        self.node_attribute()
                    reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                else:
                    reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id])
            
            
            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if _ % log == 0:
                if display_metrics == None:
                    pbar.set_description("| train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        try:
                            results[metric]
                        except:
                            raise Exception(f'{metric} not recognized')
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)
                    
            
            if save_fig and _ % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()

        self.log_history('fit')
        # revert back to original state
        self.recover_symbolic_in_fit(old_symbolic_enabled)
        return results

    def prune_node(self, threshold=1e-2, mode="auto", active_neurons_id=None, log_history=True):

        if self.acts == None:
            self.get_act()
        
        mask_up = [torch.ones(self.width_in[0], )]
        mask_down = []
        active_neurons_up = [list(range(self.width_in[0]))]
        active_neurons_down = []
        num_sums = []
        num_mults = []
        mult_arities = [[]]
        
        if active_neurons_id != None:
            mode = "manual"

        for i in range(len(self.acts_scale) - 1):
            
            mult_arity = []
            
            if mode == "auto":
                self.attribute()
                overall_important_up = self.node_scores[i+1] > threshold
                
            elif mode == "manual":
                overall_important_up = torch.zeros(self.width_in[i + 1], dtype=torch.bool)
                overall_important_up[active_neurons_id[i]] = True
                
                
            num_sum = torch.sum(overall_important_up[:self.width[i+1][0]])
            num_mult = torch.sum(overall_important_up[self.width[i+1][0]:])
            if self.mult_homo == True:
                overall_important_down = torch.cat([overall_important_up[:self.width[i+1][0]], (overall_important_up[self.width[i+1][0]:][None,:].expand(self.mult_arity,-1)).T.reshape(-1,)], dim=0)
            else:
                overall_important_down = overall_important_up[:self.width[i+1][0]]
                for j in range(overall_important_up[self.width[i+1][0]:].shape[0]):
                    active_bool = overall_important_up[self.width[i+1][0]+j]
                    arity = self.mult_arity[i+1][j]
                    overall_important_down = torch.cat([overall_important_down, torch.tensor([active_bool]*arity)])
                    if active_bool:
                        mult_arity.append(arity)
            
            num_sums.append(num_sum.item())
            num_mults.append(num_mult.item())

            mask_up.append(overall_important_up.float())
            mask_down.append(overall_important_down.float())

            active_neurons_up.append(torch.where(overall_important_up == True)[0])
            active_neurons_down.append(torch.where(overall_important_down == True)[0])
            
            mult_arities.append(mult_arity)

        active_neurons_down.append(list(range(self.width_out[-1])))
        mask_down.append(torch.ones(self.width_out[-1], ))
        
        if self.mult_homo == False:
            mult_arities.append(self.mult_arity[-1])

        self.mask_up = mask_up
        self.mask_down = mask_down

        # update act_fun[l].mask up
        for l in range(len(self.acts_scale) - 1):
            for i in range(self.width_in[l + 1]):
                if i not in active_neurons_up[l + 1]:
                    self.remove_node(l + 1, i, mode='up',log_history=False)
                    
            for i in range(self.width_out[l + 1]):
                if i not in active_neurons_down[l]:
                    self.remove_node(l + 1, i, mode='down',log_history=False)

        model2 = MultKAN(copy.deepcopy(self.width), grid=self.grid, k=self.k, base_fun=self.base_fun_name, mult_arity=self.mult_arity, ckpt_path=self.ckpt_path, auto_save=True, first_init=False, state_id=self.state_id, round=self.round)
        model2.load_state_dict(self.state_dict())
        
        width_new = [self.width[0]]
        
        for i in range(len(self.acts_scale)):
            
            if i < len(self.acts_scale) - 1:
                num_sum = num_sums[i]
                num_mult = num_mults[i]
                model2.node_bias[i].data = model2.node_bias[i].data[active_neurons_up[i+1]]
                model2.node_scale[i].data = model2.node_scale[i].data[active_neurons_up[i+1]]
                model2.subnode_bias[i].data = model2.subnode_bias[i].data[active_neurons_down[i]]
                model2.subnode_scale[i].data = model2.subnode_scale[i].data[active_neurons_down[i]]
                model2.width[i+1] = [num_sum, num_mult]
                
                model2.act_fun[i].out_dim_sum = num_sum
                model2.act_fun[i].out_dim_mult = num_mult
                
                model2.symbolic_fun[i].out_dim_sum = num_sum
                model2.symbolic_fun[i].out_dim_mult = num_mult
                
                width_new.append([num_sum, num_mult])

            model2.act_fun[i] = model2.act_fun[i].get_subset(active_neurons_up[i], active_neurons_down[i])
            model2.symbolic_fun[i] = self.symbolic_fun[i].get_subset(active_neurons_up[i], active_neurons_down[i])
            
        model2.cache_data = self.cache_data
        model2.acts = None
        
        width_new.append(self.width[-1])
        model2.width = width_new
        
        if self.mult_homo == False:
            model2.mult_arity = mult_arities
        
        if log_history:
            self.log_history('prune_node')    
            model2.state_id += 1
        
        return model2
    
    def prune_edge(self, threshold=3e-2, log_history=True):
        
        if self.acts == None:
            self.get_act()
        
        for i in range(len(self.width)-1):
            #self.act_fun[i].mask.data = ((self.acts_scale[i] > threshold).permute(1,0)).float()
            old_mask = self.act_fun[i].mask.data
            self.act_fun[i].mask.data = ((self.edge_scores[i] > threshold).permute(1,0)*old_mask).float()
            
        if log_history:
            self.log_history('fix_symbolic')
    
    def prune(self, node_th=1e-2, edge_th=3e-2):
        
        if self.acts == None:
            self.get_act()
        
        self = self.prune_node(node_th, log_history=False)
        #self.prune_node(node_th, log_history=False)
        self.forward(self.cache_data)
        self.attribute()
        self.prune_edge(edge_th, log_history=False)
        self.log_history('prune')
        return self
    
    def prune_input(self, threshold=1e-2, active_inputs=None, log_history=True):
        
        if active_inputs == None:
            self.attribute()
            input_score = self.node_scores[0]
            input_mask = input_score > threshold
            print('keep:', input_mask.tolist())
            input_id = torch.where(input_mask==True)[0]
            
        else:
            input_id = torch.tensor(active_inputs, dtype=torch.long)
        
        model2 = MultKAN(copy.deepcopy(self.width), grid=self.grid, k=self.k, base_fun=self.base_fun, mult_arity=self.mult_arity, ckpt_path=self.ckpt_path, auto_save=True, first_init=False, state_id=self.state_id, round=self.round)
        model2.load_state_dict(self.state_dict())

        model2.act_fun[0] = model2.act_fun[0].get_subset(input_id, torch.arange(self.width_out[1]))
        model2.symbolic_fun[0] = self.symbolic_fun[0].get_subset(input_id, torch.arange(self.width_out[1]))

        model2.cache_data = self.cache_data
        model2.acts = None

        model2.width[0] = [len(input_id), 0]
        model2.input_id = input_id
        
        if log_history:
            self.log_history('prune_input')
            model2.state_id += 1
        
        return model2

    def remove_edge(self, l, i, j, log_history=True):
        self.act_fun[l].mask[i][j] = 0.
        if log_history:
            self.log_history('remove_edge')

    def remove_node(self, l ,i, mode='all', log_history=True):
        if mode == 'down':
            self.act_fun[l - 1].mask[:, i] = 0.
            self.symbolic_fun[l - 1].mask[i, :] *= 0.

        elif mode == 'up':
            self.act_fun[l].mask[i, :] = 0.
            self.symbolic_fun[l].mask[:, i] *= 0.
            
        else:
            self.remove_node(l, i, mode='up')
            self.remove_node(l, i, mode='down')
            
        if log_history:
            self.log_history('remove_node')
            
            
    def attribute(self, l=None, i=None, out_score=None, plot=True):
        
        # output (out_dim, in_dim)
        
        if l != None:
            self.attribute()
            out_score = self.node_scores[l]
       
        if self.acts == None:
            self.get_act()

        def score_node2subnode(node_score, width, mult_arity, out_dim):

            assert np.sum(width) == node_score.shape[1]
            if isinstance(mult_arity, int):
                n_subnode = width[0] + mult_arity * width[1]
            else:
                n_subnode = width[0] + int(np.sum(mult_arity))

            #subnode_score_leaf = torch.zeros(out_dim, n_subnode).requires_grad_(True)
            #subnode_score = subnode_score_leaf.clone()
            #subnode_score[:,:width[0]] = node_score[:,:width[0]]
            subnode_score = node_score[:,:width[0]]
            if isinstance(mult_arity, int):
                #subnode_score[:,width[0]:] = node_score[:,width[0]:][:,:,None].expand(out_dim, node_score[width[0]:].shape[0], mult_arity).reshape(out_dim,-1)
                subnode_score = torch.cat([subnode_score, node_score[:,width[0]:][:,:,None].expand(out_dim, node_score[:,width[0]:].shape[1], mult_arity).reshape(out_dim,-1)], dim=1)
            else:
                acml = width[0]
                for i in range(len(mult_arity)):
                    #subnode_score[:, acml:acml+mult_arity[i]] = node_score[:, width[0]+i]
                    
                    subnode_score = torch.cat([subnode_score, node_score[:, width[0]+i].expand(out_dim, mult_arity[i])], dim=1)
                    acml += mult_arity[i]
            return subnode_score


        node_scores = []
        subnode_scores = []
        edge_scores = []
        
        l_query = l
        if l == None:
            l_end = self.depth
        else:
            l_end = l

        # back propagate from the queried layer
        out_dim = self.width_in[l_end]
        if out_score == None:
            node_score = torch.eye(out_dim).requires_grad_(True)
        else:
            node_score = torch.diag(out_score).requires_grad_(True)
        node_scores.append(node_score)
        
        device = self.act_fun[0].grid.device

        for l in range(l_end,0,-1):

            # node to subnode 
            if isinstance(self.mult_arity, int):
                subnode_score = score_node2subnode(node_score, self.width[l], self.mult_arity, out_dim=out_dim)
            else:
                mult_arity = self.mult_arity[l]
                subnode_score = score_node2subnode(node_score, self.width[l], mult_arity, out_dim=out_dim)

            subnode_scores.append(subnode_score)
            # subnode to edge
            #print(self.edge_actscale[l-1].device, subnode_score.device, self.subnode_actscale[l-1].device)
            edge_score = torch.einsum('ij,ki,i->kij', self.edge_actscale[l-1], subnode_score.to(device), 1/(self.subnode_actscale[l-1]+1e-4))
            edge_scores.append(edge_score)

            # edge to node
            node_score = torch.sum(edge_score, dim=1)
            node_scores.append(node_score)

        self.node_scores_all = list(reversed(node_scores))
        self.edge_scores_all = list(reversed(edge_scores))
        self.subnode_scores_all = list(reversed(subnode_scores))

        self.node_scores = [torch.mean(l, dim=0) for l in self.node_scores_all]
        self.edge_scores = [torch.mean(l, dim=0) for l in self.edge_scores_all]
        self.subnode_scores = [torch.mean(l, dim=0) for l in self.subnode_scores_all]

        # return
        if l_query != None:
            if i == None:
                return self.node_scores_all[0]
            else:
                
                # plot
                if plot:
                    in_dim = self.width_in[0]
                    plt.figure(figsize=(1*in_dim, 3))
                    plt.bar(range(in_dim),self.node_scores_all[0][i].detach().numpy())
                    plt.xticks(range(in_dim));

                return self.node_scores_all[0][i]
            
    def node_attribute(self):
        self.node_attribute_scores = []
        for l in range(1, self.depth+1):
            node_attr = self.attribute(l)
            self.node_attribute_scores.append(node_attr)
            
    def feature_interaction(self, l, neuron_th = 1e-2, feature_th = 1e-2):
        
        dic = {}
        width = self.width_in[l]

        for i in range(width):
            score = self.attribute(l,i,plot=False)

            if torch.max(score) > neuron_th:
                features = tuple(torch.where(score > torch.max(score) * feature_th)[0].detach().numpy())
                if features in dic.keys():
                    dic[features] += 1
                else:
                    dic[features] = 1

        return dic

    def suggest_symbolic(self, l, i, j, a_range=(-10, 10), b_range=(-10, 10), lib=None, topk=5, verbose=True, r2_loss_fun=lambda x: np.log2(1+1e-5-x), c_loss_fun=lambda x: x, weight_simple = 0.8):
        
        r2s = []
        cs = []
        
        if lib == None:
            symbolic_lib = SYMBOLIC_LIB
        else:
            symbolic_lib = {}
            for item in lib:
                symbolic_lib[item] = SYMBOLIC_LIB[item]

        # getting r2 and complexities
        for (name, content) in symbolic_lib.items():
            r2 = self.fix_symbolic(l, i, j, name, a_range=a_range, b_range=b_range, verbose=False, log_history=False)
            if r2 == -1e8: # zero function
                r2s.append(-1e8)
            else:
                r2s.append(r2.item())
                self.unfix_symbolic(l, i, j, log_history=False)
            c = content[2]
            cs.append(c)

        r2s = np.array(r2s)
        cs = np.array(cs)
        r2_loss = r2_loss_fun(r2s).astype('float')
        cs_loss = c_loss_fun(cs)
        
        loss = weight_simple * cs_loss + (1-weight_simple) * r2_loss
            
        sorted_ids = np.argsort(loss)[:topk]
        r2s = r2s[sorted_ids][:topk]
        cs = cs[sorted_ids][:topk]
        r2_loss = r2_loss[sorted_ids][:topk]
        cs_loss = cs_loss[sorted_ids][:topk]
        loss = loss[sorted_ids][:topk]
        
        topk = np.minimum(topk, len(symbolic_lib))
        
        if verbose == True:
            # print results in a dataframe
            results = {}
            results['function'] = [list(symbolic_lib.items())[sorted_ids[i]][0] for i in range(topk)]
            results['fitting r2'] = r2s[:topk]
            results['r2 loss'] = r2_loss[:topk]
            results['complexity'] = cs[:topk]
            results['complexity loss'] = cs_loss[:topk]
            results['total loss'] = loss[:topk]

            df = pd.DataFrame(results)
            print(df)

        '''if verbose == True:
            print('function', ',', 'r2', ',', 'c', ',', 'r2 loss', ',', 'c loss', ',', 'total loss')
            for i in range(topk):
                print(list(symbolic_lib.items())[sorted_ids[i]][0], ',', r2s[i], ',', cs[i], ',', r2_loss[i], ',', cs_loss[i], ',', loss[i])'''

        best_name = list(symbolic_lib.items())[sorted_ids[0]][0]
        best_fun = list(symbolic_lib.items())[sorted_ids[0]][1]
        best_r2 = r2s[0]
        best_c = cs[0]
            
        '''if best_r2 < 1e-3:
            # zero function
            zero_id = list(SYMBOLIC_LIB).index('0')
            best_r2 = 0.0
            best_name = '0'
            best_fun = list(symbolic_lib.items())[zero_id][1]
            best_c = 0.0
            print('behave like a zero function')'''
        
        return best_name, best_fun, best_r2, best_c;

    def auto_symbolic(self, a_range=(-10, 10), b_range=(-10, 10), lib=None, verbose=1):
        for l in range(len(self.width_in) - 1):
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l + 1]):
                    #if self.symbolic_fun[l].mask[j, i] > 0. and self.act_fun[l].mask[i][j] == 0.:
                    if self.symbolic_fun[l].mask[j, i] > 0. and self.act_fun[l].mask[i][j] == 0.:
                        print(f'skipping ({l},{i},{j}) since already symbolic')
                    elif self.symbolic_fun[l].mask[j, i] == 0. and self.act_fun[l].mask[i][j] == 0.:
                        self.fix_symbolic(l, i, j, '0', verbose=verbose > 1, log_history=False)
                        print(f'fixing ({l},{i},{j}) with 0')
                    else:
                        name, fun, r2, c = self.suggest_symbolic(l, i, j, a_range=a_range, b_range=b_range, lib=lib, verbose=False)
                        self.fix_symbolic(l, i, j, name, verbose=verbose > 1, log_history=False)
                        if verbose >= 1:
                            print(f'fixing ({l},{i},{j}) with {name}, r2={r2}, c={c}')
                            
        self.log_history('auto_symbolic')

    def symbolic_formula(self, compute_digit=5, display_digit=3, var=None, normalizer=None, simplify=False, output_normalizer = None):
        
        symbolic_acts = []
        symbolic_acts_premult = []
        x = []

        def ex_round(ex1, n_digit):
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, n_digit))
            return ex2

        # define variables
        if var == None:
            for ii in range(1, self.width[0][0] + 1):
                exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                exec(f"x.append(x{ii})")
        elif type(var[0]) == Symbol:
            x = var
        else:
            x = [sympy.symbols(var_) for var_ in var]

        x0 = x

        if normalizer != None:
            mean = normalizer[0]
            std = normalizer[1]
            x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

        symbolic_acts.append(x)

        for l in range(len(self.width_in) - 1):
            num_sum = self.width[l + 1][0]
            num_mult = self.width[l + 1][1]
            y = []
            for j in range(self.width_out[l + 1]):
                yj = 0.
                for i in range(self.width_in[l]):
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]
                    sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]
                    try:
                        yj += c * sympy_fun(a * x[i] + b) + d
                    except:
                        print('make sure all activations need to be converted to symbolic formulas first!')
                        return
                yj = self.subnode_scale[l][j] * yj + self.subnode_bias[l][j]
                if simplify == True:
                    y.append(sympy.simplify(yj))
                else:
                    y.append(yj)
                    
            symbolic_acts_premult.append(y)
                  
            mult = []
            for k in range(num_mult):
                if isinstance(self.mult_arity, int):
                    mult_arity = self.mult_arity
                else:
                    mult_arity = self.mult_arity[l+1][k]
                for i in range(mult_arity-1):
                    if i == 0:
                        mult_k = y[num_sum+2*k] * y[num_sum+2*k+1]
                    else:
                        mult_k = mult_k * y[num_sum+2*k+i+1]
                mult.append(mult_k)
                
            y = y[:num_sum] + mult
            
            for j in range(self.width_in[l+1]):
                y[j] = self.node_scale[l][j] * y[j] + self.node_bias[l][j]
            
            x = y
            symbolic_acts.append(x)

        if output_normalizer != None:
            output_layer = symbolic_acts[-1]
            means = output_normalizer[0]
            stds = output_normalizer[1]

            assert len(output_layer) == len(means), 'output_normalizer does not match the output layer'
            assert len(output_layer) == len(stds), 'output_normalizer does not match the output layer'
            
            output_layer = [(output_layer[i] * stds[i] + means[i]) for i in range(len(output_layer))]
            symbolic_acts[-1] = output_layer


        self.symbolic_acts = [[symbolic_acts[l][i] for i in range(len(symbolic_acts[l]))] for l in range(len(symbolic_acts))]
        self.symbolic_acts_premult = [[symbolic_acts_premult[l][i] for i in range(len(symbolic_acts_premult[l]))] for l in range(len(symbolic_acts_premult))]

        out_dim = len(symbolic_acts[-1])
        #return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
        
        if simplify:
            return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
        else:
            return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
        
        
    def expand_depth(self):
        
        self.depth += 1

        # add kanlayer, set mask to zero
        dim_out = self.width_in[-1]
        layer = KANLayer(dim_out, dim_out, num=self.grid, k=self.k)
        layer.mask *= 0.
        self.act_fun.append(layer)

        self.width.append([dim_out, 0])
        self.mult_arity.append([])

        # add symbolic_kanlayer set mask to one. fun = identity on diagonal and zero for off-diagonal
        layer = Symbolic_KANLayer(dim_out, dim_out)
        layer.mask += 1.

        for j in range(dim_out):
            for i in range(dim_out):
                if i == j:
                    layer.fix_symbolic(i,j,'x')
                else:
                    layer.fix_symbolic(i,j,'0')

        self.symbolic_fun.append(layer)

        self.node_bias.append(torch.nn.Parameter(torch.zeros(dim_out,)).requires_grad_(self.affine_trainable))
        self.node_scale.append(torch.nn.Parameter(torch.ones(dim_out,)).requires_grad_(self.affine_trainable))
        self.subnode_bias.append(torch.nn.Parameter(torch.zeros(dim_out,)).requires_grad_(self.affine_trainable))
        self.subnode_scale.append(torch.nn.Parameter(torch.ones(dim_out,)).requires_grad_(self.affine_trainable))
        
        self.log_history('expand_depth')

    def expand_width(self, layer_id, n_added_nodes, sum_bool=True, mult_arity=2):
        
        def _expand(layer_id, n_added_nodes, sum_bool=True, mult_arity=2, added_dim='out'):
            l = layer_id
            in_dim = self.symbolic_fun[l].in_dim
            out_dim = self.symbolic_fun[l].out_dim
            if sum_bool:

                if added_dim == 'out':
                    new = Symbolic_KANLayer(in_dim, out_dim + n_added_nodes)
                    old = self.symbolic_fun[l]
                    in_id = np.arange(in_dim)
                    out_id = np.arange(out_dim + n_added_nodes) 

                    for j in out_id:
                        for i in in_id:
                            new.fix_symbolic(i,j,'0')
                    new.mask += 1.

                    for j in out_id:
                        for i in in_id:
                            if j > n_added_nodes-1:
                                new.funs[j][i] = old.funs[j-n_added_nodes][i]
                                new.funs_avoid_singularity[j][i] = old.funs_avoid_singularity[j-n_added_nodes][i]
                                new.funs_sympy[j][i] = old.funs_sympy[j-n_added_nodes][i]
                                new.funs_name[j][i] = old.funs_name[j-n_added_nodes][i]
                                new.affine.data[j][i] = old.affine.data[j-n_added_nodes][i]

                    self.symbolic_fun[l] = new
                    self.act_fun[l] = KANLayer(in_dim, out_dim + n_added_nodes, num=self.grid, k=self.k)
                    self.act_fun[l].mask *= 0.

                    self.node_scale[l].data = torch.cat([torch.ones(n_added_nodes), self.node_scale[l].data])
                    self.node_bias[l].data = torch.cat([torch.zeros(n_added_nodes), self.node_bias[l].data])
                    self.subnode_scale[l].data = torch.cat([torch.ones(n_added_nodes), self.subnode_scale[l].data])
                    self.subnode_bias[l].data = torch.cat([torch.zeros(n_added_nodes), self.subnode_bias[l].data])



                if added_dim == 'in':
                    new = Symbolic_KANLayer(in_dim + n_added_nodes, out_dim)
                    old = self.symbolic_fun[l]
                    in_id = np.arange(in_dim + n_added_nodes)
                    out_id = np.arange(out_dim) 

                    for j in out_id:
                        for i in in_id:
                            new.fix_symbolic(i,j,'0')
                    new.mask += 1.

                    for j in out_id:
                        for i in in_id:
                            if i > n_added_nodes-1:
                                new.funs[j][i] = old.funs[j][i-n_added_nodes]
                                new.funs_avoid_singularity[j][i] = old.funs_avoid_singularity[j][i-n_added_nodes]
                                new.funs_sympy[j][i] = old.funs_sympy[j][i-n_added_nodes]
                                new.funs_name[j][i] = old.funs_name[j][i-n_added_nodes]
                                new.affine.data[j][i] = old.affine.data[j][i-n_added_nodes]

                    self.symbolic_fun[l] = new
                    self.act_fun[l] = KANLayer(in_dim + n_added_nodes, out_dim, num=self.grid, k=self.k)
                    self.act_fun[l].mask *= 0.


            else:

                if isinstance(mult_arity, int):
                    mult_arity = [mult_arity] * n_added_nodes

                if added_dim == 'out':
                    n_added_subnodes = np.sum(mult_arity)
                    new = Symbolic_KANLayer(in_dim, out_dim + n_added_subnodes)
                    old = self.symbolic_fun[l]
                    in_id = np.arange(in_dim)
                    out_id = np.arange(out_dim + n_added_nodes)

                    for j in out_id:
                        for i in in_id:
                            new.fix_symbolic(i,j,'0')
                    new.mask += 1.

                    for j in out_id:
                        for i in in_id:
                            if j < out_dim:
                                new.funs[j][i] = old.funs[j][i]
                                new.funs_avoid_singularity[j][i] = old.funs_avoid_singularity[j][i]
                                new.funs_sympy[j][i] = old.funs_sympy[j][i]
                                new.funs_name[j][i] = old.funs_name[j][i]
                                new.affine.data[j][i] = old.affine.data[j][i]

                    self.symbolic_fun[l] = new
                    self.act_fun[l] = KANLayer(in_dim, out_dim + n_added_subnodes, num=self.grid, k=self.k)
                    self.act_fun[l].mask *= 0.

                    self.node_scale[l].data = torch.cat([self.node_scale[l].data, torch.ones(n_added_nodes)])
                    self.node_bias[l].data = torch.cat([self.node_bias[l].data, torch.zeros(n_added_nodes)])
                    self.subnode_scale[l].data = torch.cat([self.subnode_scale[l].data, torch.ones(n_added_subnodes)])
                    self.subnode_bias[l].data = torch.cat([self.subnode_bias[l].data, torch.zeros(n_added_subnodes)])

                if added_dim == 'in':
                    new = Symbolic_KANLayer(in_dim + n_added_nodes, out_dim)
                    old = self.symbolic_fun[l]
                    in_id = np.arange(in_dim + n_added_nodes)
                    out_id = np.arange(out_dim) 

                    for j in out_id:
                        for i in in_id:
                            new.fix_symbolic(i,j,'0')
                    new.mask += 1.

                    for j in out_id:
                        for i in in_id:
                            if i < in_dim:
                                new.funs[j][i] = old.funs[j][i]
                                new.funs_avoid_singularity[j][i] = old.funs_avoid_singularity[j][i]
                                new.funs_sympy[j][i] = old.funs_sympy[j][i]
                                new.funs_name[j][i] = old.funs_name[j][i]
                                new.affine.data[j][i] = old.affine.data[j][i]

                    self.symbolic_fun[l] = new
                    self.act_fun[l] = KANLayer(in_dim + n_added_nodes, out_dim, num=self.grid, k=self.k)
                    self.act_fun[l].mask *= 0.

        _expand(layer_id-1, n_added_nodes, sum_bool, mult_arity, added_dim='out')
        _expand(layer_id, n_added_nodes, sum_bool, mult_arity, added_dim='in')
        if sum_bool:
            self.width[layer_id][0] += n_added_nodes
        else:
            if isinstance(mult_arity, int):
                mult_arity = [mult_arity] * n_added_nodes

            self.width[layer_id][1] += n_added_nodes
            self.mult_arity[layer_id] += mult_arity
            
        self.log_history('expand_width')
            
    def perturb(self, mag=0.02, mode='all'):
        if mode == 'all':
            for i in range(self.depth):
                self.act_fun[i].mask += self.act_fun[i].mask*0. + mag
                
        if mode == 'minimal':
            for l in range(self.depth):
                funs_name = self.symbolic_fun[l].funs_name
                for j in range(self.width_out[l+1]):
                    for i in range(self.width_in[l]):
                        if funs_name[j][i] != '0':
                            self.act_fun[l].mask.data[i][j] = mag
        
        self.log_history('perturb')
                            
                            
    def module(self, start_layer, chain):
        #chain = '[-1]->[-1,-2]->[-1]->[-1]'
        groups = chain.split('->')
        n_total_layers = len(groups)//2
        #start_layer = 0

        for l in range(n_total_layers):
            current_layer = cl = start_layer + l
            id_in = [int(i) for i in groups[2*l][1:-1].split(',')]
            id_out = [int(i) for i in groups[2*l+1][1:-1].split(',')]

            in_dim = self.width_in[cl]
            out_dim = self.width_out[cl+1]
            id_in_other = list(set(range(in_dim)) - set(id_in))
            id_out_other = list(set(range(out_dim)) - set(id_out))
            self.act_fun[cl].mask.data[np.ix_(id_in_other,id_out)] = 0.
            self.act_fun[cl].mask.data[np.ix_(id_in,id_out_other)] = 0.
            self.symbolic_fun[cl].mask.data[np.ix_(id_out,id_in_other)] = 0.
            self.symbolic_fun[cl].mask.data[np.ix_(id_out_other,id_in)] = 0.
            
        self.log_history('module')
        
    def tree(self, x=None, in_var=None, style='tree', sym_th=1e-3, sep_th=1e-1, skip_sep_test=False, verbose=False):
        if x == None:
            x = self.cache_data
        plot_tree(self, x, in_var=in_var, style=style, sym_th=sym_th, sep_th=sep_th, skip_sep_test=skip_sep_test, verbose=verbose)
        
        
    def speed(self, compile=False):
        self.symbolic_enabled=False
        self.save_act=False
        self.auto_save=False
        if compile == True:
            return torch.compile(self)
        else:
            return self
        
    def get_act(self, x=None):
        if isinstance(x, dict):
            x = x['train_input']
        if x == None:
            if self.cache_data != None:
                x = self.cache_data
            else:
                raise Exception("missing input data x")
        save_act = self.save_act
        self.save_act = True
        self.forward(x)
        self.save_act = save_act
        
    def get_fun(self, l, i, j):
        inputs = self.spline_preacts[l][:,j,i]
        outputs = self.spline_postacts[l][:,j,i]
        # they are not ordered yet
        rank = np.argsort(inputs)
        inputs = inputs[rank]
        outputs = outputs[rank]
        plt.figure(figsize=(3,3))
        plt.plot(inputs, outputs, marker="o")
        return inputs, outputs
        
        
    def history(self, k='all'):
        
        with open(self.ckpt_path+'/history.txt', 'r') as f:
            data = f.readlines()
            n_line = len(data)
            if k == 'all':
                k = n_line

            data = data[-k:]
            for line in data:
                print(line[:-1])
    @property
    def n_edge(self):
        depth = len(self.act_fun)
        complexity = 0
        for l in range(depth):
            complexity += torch.sum(self.act_fun[l].mask > 0.)
        return complexity.item()
    
    def evaluate(self, dataset):
        evaluation = {}
        evaluation['test_loss'] = torch.sqrt(torch.mean((self.forward(dataset['test_input']) - dataset['test_label'])**2)).item()
        evaluation['n_edge'] = self.n_edge
        evaluation['n_grid'] = self.grid
        # add other metrics (maybe accuracy)
        return evaluation
    
    def swap(self, l, i1, i2, log_history=True):
        
        self.act_fun[l-1].swap(i1,i2,mode='out')
        self.symbolic_fun[l-1].swap(i1,i2,mode='out')
        self.act_fun[l].swap(i1,i2,mode='in')
        self.symbolic_fun[l].swap(i1,i2,mode='in')
        
        def swap_(data, i1, i2):
            data[i1], data[i2] = data[i2], data[i1]
            
        swap_(self.node_scale[l-1].data, i1, i2)
        swap_(self.node_bias[l-1].data, i1, i2)
        swap_(self.subnode_scale[l-1].data, i1, i2)
        swap_(self.subnode_bias[l-1].data, i1, i2)
        
        if log_history:
            self.log_history('swap')
            
    @property
    def connection_cost(self):
        
        cc = 0.
        for t in self.edge_scores:
            
            def get_coordinate(n):
                return torch.linspace(0,1,steps=n+1)[:n] + 1/(2*n)

            in_dim = t.shape[0]
            x_in = get_coordinate(in_dim)

            out_dim = t.shape[1]
            x_out = get_coordinate(out_dim)

            dist = torch.abs(x_in[:,None] - x_out[None,:])
            cc += torch.sum(dist * t)

        return cc
    
    def auto_swap_l(self, l):

        num = self.width_in[1]
        for i in range(num):
            ccs = []
            for j in range(num):
                self.swap(l,i,j,log_history=False)
                self.get_act()
                self.attribute()
                cc = self.connection_cost.detach().clone()
                ccs.append(cc)
                self.swap(l,i,j,log_history=False)
            j = torch.argmin(torch.tensor(ccs))
            self.swap(l,i,j,log_history=False)

    def auto_swap(self):
        depth = self.depth
        for l in range(1, depth):
            self.auto_swap_l(l)
            
        self.log_history('auto_swap')

KAN = MultKAN
