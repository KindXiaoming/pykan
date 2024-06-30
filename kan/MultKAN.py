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

class MultKAN(nn.Module):

    # include mult_ops = []
    def __init__(self, width=None, grid=3, k=3, mult_arity = 2, noise_scale=0.1, scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu', symbolic_enabled=True, affine_trainable=False, grid_eps=1.0, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, device='cpu', seed=0, save_plot_data=True):
        
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
            sp_batch = KANLayer(in_dim=width_in[l], out_dim=width_out[l+1], num=grid, k=k, noise_scale=noise_scale, scale_base=scale_base, scale_sp=1., base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable, sb_trainable=sb_trainable, device=device)
            self.act_fun.append(sp_batch)

        
        #self.node_bias = [torch.nn.Parameter(torch.zeros(width_in[l+1],)).requires_grad_(affine_trainable).to(device) for l in range(self.depth)]
        #self.node_scale = [torch.nn.Parameter(torch.ones(width_in[l+1],)).requires_grad_(affine_trainable).to(device) for l in range(self.depth)]
        #self.subnode_bias = [torch.nn.Parameter(torch.zeros(width_out[l+1],)).requires_grad_(affine_trainable).to(device) for l in range(self.depth)]
        #self.subnode_scale = [torch.nn.Parameter(torch.ones(width_out[l+1],)).requires_grad_(affine_trainable).to(device) for l in range(self.depth)]
        
        self.node_bias = []
        self.node_scale = []
        self.subnode_bias = []
        self.subnode_scale = []
        
        globals()['self.node_bias_0'] = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)
        #self.node_bias_0 = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)
        exec('self.node_bias_0' + " = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)")
        
        for l in range(self.depth):
            exec(f'self.node_bias_{l} = torch.nn.Parameter(torch.zeros(width_in[l+1],)).requires_grad_(affine_trainable).to(device)')
            exec(f'self.node_scale_{l} = torch.nn.Parameter(torch.ones(width_in[l+1],)).requires_grad_(affine_trainable).to(device)')
            exec(f'self.subnode_bias_{l} = torch.nn.Parameter(torch.zeros(width_out[l+1],)).requires_grad_(affine_trainable).to(device)')
            exec(f'self.subnode_scale_{l} = torch.nn.Parameter(torch.ones(width_out[l+1],)).requires_grad_(affine_trainable).to(device)')
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
            sb_batch = Symbolic_KANLayer(in_dim=width_in[l], out_dim=width_out[l+1], device=device)
            self.symbolic_fun.append(sb_batch)

        self.symbolic_fun = nn.ModuleList(self.symbolic_fun)
        self.symbolic_enabled = symbolic_enabled
        self.affine_trainable = affine_trainable
        self.sp_trainable = sp_trainable
        self.sb_trainable = sb_trainable
        
        self.device = device
        self.save_plot_data = save_plot_data
        
        self.cache_data = None

    def initialize_from_another_model(self, another_model, x):
        another_model(x.to(another_model.device))  # get activations
        batch = x.shape[0]

        self.initialize_grid_from_another_model(another_model, x.to(another_model.device))

        for l in range(self.depth):
            spb = self.act_fun[l]
            spb_parent = another_model.act_fun[l]

            # spb = spb_parent
            preacts = another_model.spline_preacts[l]
            postsplines = another_model.spline_postsplines[l]
            self.act_fun[l].coef.data = curve2coef(preacts[:,0,:], postsplines.permute(0,2,1), spb.grid, k=spb.k, device=self.device)
            spb.scale_base.data = spb_parent.scale_base.data
            spb.scale_sp.data = spb_parent.scale_sp.data
            spb.mask.data = spb_parent.mask.data

        for l in range(self.depth):
            self.node_bias[l].data = another_model.node_bias[l].data
            self.node_scale[l].data = another_model.node_scale[l].data
            
            self.subnode_bias[l].data = another_model.subnode_bias[l].data
            self.subnode_scale[l].data = another_model.subnode_scale[l].data

        for l in range(self.depth):
            self.symbolic_fun[l] = another_model.symbolic_fun[l]

        return self

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
    
    def update_grid_from_samples(self, x):
        for l in range(self.depth):
            self.forward(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])

    def initialize_grid_from_another_model(self, model, x):
        model(x)
        for l in range(self.depth):
            self.act_fun[l].initialize_grid_from_parent(model.act_fun[l], model.acts[l])

    def forward(self, x, singularity_avoiding=False, y_th=10.):
        
        # cache data
        self.cache_data = x
        
        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.acts_premult = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_spline = []
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
            # subnode affine transform
            x = self.subnode_scale[l][None,:] * x + self.subnode_bias[l][None,:]
            
            if self.save_plot_data:
                postacts = postacts_numerical + postacts_symbolic

                # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
                #grid_reshape = self.act_fun[l].grid.reshape(self.width_out[l + 1], self.width_in[l], -1)
                input_range = torch.std(preacts, dim=0) + 0.1
                output_range_spline = torch.std(postacts_numerical, dim=0) # for training, only penalize the spline part
                output_range = torch.std(postacts, dim=0) # for visualization, include the contribution from both spline + symbolic
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

    def fix_symbolic(self, l, i, j, fun_name, fit_params_bool=True, a_range=(-10, 10), b_range=(-10, 10), verbose=True, random=False):
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
        return r2

    def unfix_symbolic(self, l, i, j):
        self.set_mode(l, i, j, mode="n")
        self.symbolic_fun[l].funs_name[j][i] = ""

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

    def plot(self, folder="./figures", beta=3, mask=False, mode="supervised", scale=0.5, tick=False, sample=False, in_vars=None, out_vars=None, title=None):
        
        global Symbol
        
        if not self.save_plot_data:
            print('cannot plot since data are not saved. Set save_plot_data=True first.')
        
        # forward to obtain activations
        self.forward(self.cache_data)
        
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

        if mode == "supervised":
            alpha = [score2alpha(score.cpu().detach().numpy()) for score in self.acts_scale]
        elif mode == "unsupervised":
            alpha = [score2alpha(score.cpu().detach().numpy()) for score in self.acts_scale_std]

            
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
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, f'${latex(in_vars[i])}$', fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')
                else:
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, in_vars[i], fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')
                
                

        if out_vars != None:
            n = self.width_in[-1]
            for i in range(n):
                if isinstance(out_vars[i], sympy.Expr):
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0+z0) * (len(self.width) - 1) + 0.15, f'${latex(out_vars[i])}$', fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')
                else:
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0+z0) * (len(self.width) - 1) + 0.15, out_vars[i], fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')

        if title != None:
            plt.gcf().get_axes()[0].text(0.5, (y0+z0) * (len(self.width) - 1) + 0.3, title, fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')

    def train(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., stop_grid_update_step=50, batch=-1,
              small_mag_threshold=1e-16, small_reg_factor=1., metrics=None, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', device='cpu', singularity_avoiding=False, y_th=10.):

        if lamb > 0. and not self.save_plot_data:
            print('setting lamb=0. If you want to set lamb > 0, set self.save_plot_data=True')
        
        def reg(acts_scale):

            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

            reg_ = 0.
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(-1, )

                p = vec / (torch.sum(vec) + 1e-4)
                l1 = torch.sum(nonlinear(vec))
                entropy = - torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

            # regularize coefficient to encourage spline to be zero
            for i in range(len(self.act_fun)):
                coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
                coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_

        pbar = tqdm(range(steps), desc='description', ncols=100)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
            #optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
            #optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, debug=True)

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
            pred = self.forward(dataset['train_input'][train_id].to(device), singularity_avoiding=singularity_avoiding, y_th=y_th)
            train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
            if self.save_plot_data:
                reg_ = reg(self.acts_scale_spline)
            else:
                reg_ = torch.tensor(0.)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        for _ in pbar:

            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid:
                self.update_grid_from_samples(dataset['train_input'][train_id].to(device))

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id].to(device))
                if sglr_avoid == True:
                    id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                    train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
                else:
                    train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
                if self.save_plot_data:
                    reg_ = reg(self.acts_scale_spline)
                else:
                    reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id].to(device)), dataset['test_label'][test_id].to(device))

            if _ % log == 0:
                pbar.set_description("train loss: %.2e | test loss: %.2e | reg: %.2e " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))

            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if save_fig and _ % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()

        return results

    def prune_node(self, threshold=1e-2, mode="auto", active_neurons_id=None):

        mask_up = [torch.ones(self.width_in[0], )]
        mask_down = []
        active_neurons_up = [list(range(self.width_in[0]))]
        active_neurons_down = []

        for i in range(len(self.acts_scale) - 1):
            if mode == "auto":
                in_important = torch.max(self.acts_scale[i], dim=1)[0] > threshold # num_sum + 2 * num_mult
                in_important_sum = in_important[:self.width[i+1][0]]
                in_important_mult = in_important[self.width[i+1][0]:].reshape(self.width[i+1][1], 2)
                in_important_mult = torch.prod(in_important_mult, dim=1)
                in_important = torch.cat([in_important_sum, in_important_mult], dim=0)

                out_important = torch.max(self.acts_scale[i + 1], dim=0)[0] > threshold # num_sum + num_mult
                overall_important_up = in_important * out_important # num_sum + num_mult
                overall_important_down = torch.cat([overall_important_up[:self.width[i+1][0]], (overall_important_up[self.width[i+1][0]:][None,:].expand(2,-1)).T.reshape(-1,)], dim=0) # num_sum + 2 * num_mult

            elif mode == "manual":
                overall_important_up = torch.zeros(self.width_in[i + 1], dtype=torch.bool)
                overall_important_up[active_neurons_up[i]] = True
                overall_important_down = torch.cat([overall_important_up[:self.width[i+1][0]], (overall_important_up[self.width[i+1][0]:][None,:].expand(2,-1)).T.reshape(-1,)], dim=0) # num_sum + 2 * num_mult

            mask_up.append(overall_important_up.float())
            mask_down.append(overall_important_down.float())

            active_neurons_up.append(torch.where(overall_important_up == True)[0])
            active_neurons_down.append(torch.where(overall_important_down == True)[0])

        active_neurons_down.append(list(range(self.width_in[-1])))
        mask_down.append(torch.ones(self.width_in[-1], ))

        self.mask_up = mask_up
        self.mask_down = mask_down

        # update act_fun[l].mask up
        for l in range(len(self.acts_scale) - 1):
            for i in range(self.width_in[l + 1]):
                if i not in active_neurons_up[l + 1]:
                    self.remove_node(l + 1, i, mode='up')
                    
            for i in range(self.width_out[l + 1]):
                if i not in active_neurons_down[l]:
                    self.remove_node(l + 1, i, mode='down')

        model2 = MultKAN(copy.deepcopy(self.width), self.grid, self.k, base_fun=self.base_fun, device=self.device)
        model2.load_state_dict(self.state_dict())
        
        width_new = [self.width[0]]
        
        for i in range(len(self.acts_scale)):
            
            if i < len(self.acts_scale) - 1:
                num_mult = len(active_neurons_down[i]) - len(active_neurons_up[i+1])
                num_sum = len(active_neurons_down[i]) - 2 * num_mult
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
        
        width_new.append(self.width[-1])
        model2.width = width_new
        
            
        return model2
    
    def prune_edge(self, threshold=3e-2):
        for i in range(len(self.width)-1):
            self.act_fun[i].mask.data = ((self.acts_scale[i] > threshold).permute(1,0)).float()
    
    def prune(self, node_th=1e-2, edge_th=3e-2):
        self = self.prune_node(node_th)
        self.forward(self.cache_data)
        self.prune_edge(edge_th)
        return self
    
    def remove_edge(self, l, i, j):
        self.act_fun[l].mask[i][j] = 0.

    def remove_node(self, l ,i, mode='down'):
        if mode == 'down':
            self.act_fun[l - 1].mask[:, i] = 0.
            self.symbolic_fun[l - 1].mask[i, :] *= 0.

        elif mode == 'up':
            self.act_fun[l].mask[i, :] = 0.
            self.symbolic_fun[l].mask[:, i] *= 0.

    def suggest_symbolic(self, l, i, j, a_range=(-10, 10), b_range=(-10, 10), lib=None, topk=5, verbose=True, r2_loss_fun=lambda x: 1 - x, c_loss_fun=lambda x: x, weight_simple = 0.02):
        
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
            r2 = self.fix_symbolic(l, i, j, name, a_range=a_range, b_range=b_range, verbose=False)
            if r2 == -1e8: # zero function
                r2s.append(-1e8)
            else:
                r2s.append(r2.item())
                self.unfix_symbolic(l, i, j)
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
                    if self.symbolic_fun[l].mask[j, i] > 0. and self.act_fun[l].mask[i][j] == 0.:
                        print(f'skipping ({l},{i},{j}) since already symbolic')
                    else:
                        name, fun, r2, c = self.suggest_symbolic(l, i, j, a_range=a_range, b_range=b_range, lib=lib, verbose=False)
                        self.fix_symbolic(l, i, j, name, verbose=verbose > 1)
                        if verbose >= 1:
                            print(f'fixing ({l},{i},{j}) with {name}, r2={r2}, c={c}')

    def symbolic_formula(self, n_digit=2, var=None, normalizer=None, simplify=False, output_normalizer = None):
        
        symbolic_acts = []
        symbolic_acts_premult = []
        x = []

        def ex_round(ex1, n_digit=n_digit):
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
                for i in range(self.mult_arity-1):
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


        self.symbolic_acts = [[ex_round(symbolic_acts[l][i]) for i in range(len(symbolic_acts[l]))] for l in range(len(symbolic_acts))]
        self.symbolic_acts_premult = [[ex_round(symbolic_acts_premult[l][i]) for i in range(len(symbolic_acts_premult[l]))] for l in range(len(symbolic_acts_premult))]

        out_dim = len(symbolic_acts[-1])
        #return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
        if simplify:
            return [sympy.simplify(ex_round(ex_round(symbolic_acts[-1][i]))) for i in range(len(symbolic_acts[-1]))], x0
        else:
            return [ex_round(ex_round(symbolic_acts[-1][i])) for i in range(len(symbolic_acts[-1]))], x0
        
        
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

        self.node_bias.append(torch.nn.Parameter(torch.zeros(dim_out,)).requires_grad_(self.affine_trainable).to(self.device))
        self.node_scale.append(torch.nn.Parameter(torch.ones(dim_out,)).requires_grad_(self.affine_trainable).to(self.device))
        self.subnode_bias.append(torch.nn.Parameter(torch.zeros(dim_out,)).requires_grad_(self.affine_trainable).to(self.device))
        self.subnode_scale.append(torch.nn.Parameter(torch.ones(dim_out,)).requires_grad_(self.affine_trainable).to(self.device))

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
                            