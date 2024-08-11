import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sympy.utilities.lambdify import lambdify
from sklearn.cluster import AgglomerativeClustering
from .utils import batch_jacobian, batch_hessian
from functools import reduce
from kan.utils import batch_jacobian, batch_hessian
import copy 
import matplotlib.pyplot as plt
import sympy 
from sympy.printing import latex


def detect_separability(model, x, mode='add', score_th=1e-2, res_th=1e-2, n_clusters=None, bias=0, verbose=False):
    
    results = {}
    
    if mode == 'add':
        hessian = batch_hessian(model, x)
    elif mode == 'mul':
        compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)
        hessian = batch_hessian(compose(torch.log, torch.abs, lambda x: x+bias, model), x)
        
    std = torch.std(x, dim=0)
    hessian_normalized = hessian * std[None,:] * std[:,None]
    score_mat = torch.median(torch.abs(hessian_normalized), dim=0)[0]
    results['hessian'] = score_mat

    dist_hard = (score_mat < score_th).float()
    
    if isinstance(n_clusters, int):
        n_cluster_try = [n_clusters, n_clusters]
    elif isinstance(n_clusters, list):
        n_cluster_try = n_clusters
    else:
        n_cluster_try = [1,x.shape[1]]
        
    n_cluster_try = list(range(n_cluster_try[0], n_cluster_try[1]+1))
    
    for n_cluster in n_cluster_try:
        
        clustering = AgglomerativeClustering(
          metric='precomputed',
          n_clusters=n_cluster,
          linkage='complete',
        ).fit(dist_hard)

        labels = clustering.labels_

        groups = [list(np.where(labels == i)[0]) for i in range(n_cluster)]
        blocks = [torch.sum(score_mat[groups[i]][:,groups[i]]) for i in range(n_cluster)]
        block_sum = torch.sum(torch.stack(blocks))
        total_sum = torch.sum(score_mat)
        residual_sum = total_sum - block_sum
        residual_ratio = residual_sum / total_sum
        
        if verbose == True:
            print(f'n_group={n_cluster}, residual_ratio={residual_ratio}')
        
        if residual_ratio < res_th:
            results['n_groups'] = n_cluster
            results['labels'] = list(labels)
            results['groups'] = groups
            
    if results['n_groups'] > 1:
        print(f'{mode} separability detected')
    else:
        print(f'{mode} separability not detected')
            
    return results


def batch_grad_normgrad(model, x, group, create_graph=False):
    # x in shape (Batch, Length)
    group_A = group
    group_B = list(set(range(x.shape[1])) - set(group))

    def jac(x):
        input_grad = batch_jacobian(model, x, create_graph=True)
        input_grad_A = input_grad[:,group_A]
        norm = torch.norm(input_grad_A, dim=1, keepdim=True) + 1e-6
        input_grad_A_normalized = input_grad_A/norm
        return input_grad_A_normalized
    
    def _jac_sum(x):
        return jac(x).sum(dim=0)
    
    return torch.autograd.functional.jacobian(_jac_sum, x, create_graph=create_graph).permute(1,0,2)[:,:,group_B]


def get_dependence(model, x, group):
    group_A = group
    group_B = list(set(range(x.shape[1])) - set(group))
    grad_normgrad = batch_grad_normgrad(model, x, group=group)
    std = torch.std(x, dim=0)
    dependence = grad_normgrad * std[None,group_A,None] * std[None,None,group_B]
    dependence = torch.median(torch.abs(dependence), dim=0)[0]
    return dependence

def test_symmetry(model, x, group, dependence_th=1e-3):
    
    if len(group) == x.shape[1] or len(group) == 0:
        return True
    
    dependence = get_dependence(model, x, group)
    max_dependence = torch.max(dependence)
    return max_dependence < dependence_th


def test_separability(model, x, groups, mode='add', threshold=1e-2, bias=0):

    if mode == 'add':
        hessian = batch_hessian(model, x)
    elif mode == 'mul':
        compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)
        hessian = batch_hessian(compose(torch.log, torch.abs, lambda x: x+bias, model), x)

    std = torch.std(x, dim=0)
    hessian_normalized = hessian * std[None,:] * std[:,None]
    score_mat = torch.median(torch.abs(hessian_normalized), dim=0)[0]
    
    sep_bool = True
    
    # internal test
    n_groups = len(groups)
    for i in range(n_groups):
        for j in range(i+1, n_groups):
            sep_bool *= torch.max(score_mat[groups[i]][:,groups[j]]) < threshold
            
    # external test
    group_id = [x for xs in groups for x in xs]
    nongroup_id = list(set(range(x.shape[1])) - set(group_id))
    if len(nongroup_id) > 0 and len(group_id) > 0:
        sep_bool *= torch.max(score_mat[group_id][:,nongroup_id]) < threshold 

    return sep_bool

def test_general_separability(model, x, groups, threshold=1e-2):

    grad = batch_jacobian(model, x)

    gensep_bool = True

    n_groups = len(groups)
    for i in range(n_groups):
        for j in range(i+1,n_groups):
            group_A = groups[i]
            group_B = groups[j]
            for member_A in group_A:
                for member_B in group_B:
                    def func(x):
                        grad = batch_jacobian(model, x, create_graph=True)
                        return grad[:,[member_B]]/grad[:,[member_A]]
                    # test if func is multiplicative separable
                    gensep_bool *= test_separability(func, x, groups, mode='mul', threshold=threshold)
    return gensep_bool


def get_molecule(model, x, sym_th=1e-3, verbose=True):

    n = x.shape[1]
    atoms = [[i] for i in range(n)]
    molecules = []
    moleculess = [copy.deepcopy(atoms)]
    already_full = False
    n_layer = 0
    last_n_molecule = n
    
    while True:
        

        pointer = 0
        current_molecule = []
        remove_atoms = []
        n_atom = 0

        while len(atoms) > 0:
            
            # assemble molecule
            atom = atoms[pointer]
            if verbose:
                print(current_molecule)
                print(atom)

            if len(current_molecule) == 0:
                full = False
                current_molecule += atom
                remove_atoms.append(atom)
                n_atom += 1
            else:
                # try assemble the atom to the molecule
                if len(current_molecule+atom) == x.shape[1] and already_full == False and n_atom > 1 and n_layer > 0:
                    full = True
                    already_full = True
                else:
                    full = False
                    if test_symmetry(model, x, current_molecule+atom, dependence_th=sym_th):
                        current_molecule += atom
                        remove_atoms.append(atom)
                        n_atom += 1

            pointer += 1
            
            if pointer == len(atoms) or full:
                molecules.append(current_molecule)
                if full:
                    molecules.append(atom)
                    remove_atoms.append(atom)
                # remove molecules from atoms
                for atom in remove_atoms:
                    atoms.remove(atom)
                current_molecule = []
                remove_atoms = []
                pointer = 0
                
        # if not making progress, terminate
        if len(molecules) == last_n_molecule:
            def flatten(xss):
                return [x for xs in xss for x in xs]
            moleculess.append([flatten(molecules)])
            break
        else:
            moleculess.append(copy.deepcopy(molecules))
            
        last_n_molecule = len(molecules)

        if len(molecules) == 1:
            break

        atoms = molecules
        molecules = []
        
        n_layer += 1
        
        #print(n_layer, atoms)
        
        
    # sort
    depth = len(moleculess) - 1

    for l in list(range(depth,0,-1)):

        molecules_sorted = []
        molecules_l = moleculess[l]
        molecules_lm1 = moleculess[l-1]


        for molecule_l in molecules_l:
            start = 0
            for i in range(1,len(molecule_l)+1):
                if molecule_l[start:i] in molecules_lm1:

                    molecules_sorted.append(molecule_l[start:i])
                    start = i

        moleculess[l-1] = molecules_sorted

    return moleculess


def get_tree_node(model, x, moleculess, sep_th=1e-2, skip_test=True):

    arities = []
    properties = []
    
    depth = len(moleculess) - 1

    for l in range(depth):
        molecules_l = copy.deepcopy(moleculess[l])
        molecules_lp1 = copy.deepcopy(moleculess[l+1])
        arity_l = []
        property_l = []

        for molecule in molecules_lp1:
            start = 0
            arity = 0
            groups = []
            for i in range(1,len(molecule)+1):
                if molecule[start:i] in molecules_l:
                    groups.append(molecule[start:i])
                    start = i
                    arity += 1
            arity_l.append(arity)

            if arity == 1:
                property = 'Id'
            else:
                property = ''
                # test property
                if skip_test:
                    gensep_bool = False
                else:
                    gensep_bool = test_general_separability(model, x, groups, threshold=sep_th)
                    
                if gensep_bool:
                    property = 'GS'
                if l == depth - 1:
                    if skip_test:
                        add_bool = False
                        mul_bool = False
                    else:
                        add_bool = test_separability(model, x, groups, mode='add', threshold=sep_th)
                        mul_bool = test_separability(model, x, groups, mode='mul', threshold=sep_th)
                    if add_bool:
                        property = 'Add'
                    if mul_bool:
                        property = 'Mul'


            property_l.append(property)


        arities.append(arity_l)
        properties.append(property_l)
        
    return arities, properties


def plot_tree(model, x, in_var=None, style='tree', sym_th=1e-3, sep_th=1e-1, skip_sep_test=False, verbose=False):
    
    moleculess = get_molecule(model, x, sym_th=sym_th, verbose=verbose)
    arities, properties = get_tree_node(model, x, moleculess, sep_th=sep_th, skip_test=skip_sep_test)

    n = x.shape[1]
    var = None

    in_vars = []

    if in_var == None:
        for ii in range(1, n + 1):
            exec(f"x{ii} = sympy.Symbol('x_{ii}')")
            exec(f"in_vars.append(x{ii})")
    elif type(var[0]) == Symbol:
        in_vars = var
    else:
        in_vars = [sympy.symbols(var_) for var_ in var]
        

    def flatten(xss):
        return [x for xs in xss for x in xs]

    def myrectangle(center_x, center_y, width_x, width_y):
        plt.plot([center_x - width_x/2, center_x + width_x/2], [center_y + width_y/2, center_y + width_y/2], color='k') # up
        plt.plot([center_x - width_x/2, center_x + width_x/2], [center_y - width_y/2, center_y - width_y/2], color='k') # down
        plt.plot([center_x - width_x/2, center_x - width_x/2], [center_y - width_y/2, center_y + width_y/2], color='k') # left
        plt.plot([center_x + width_x/2, center_x + width_x/2], [center_y - width_y/2, center_y + width_y/2], color='k') # left

    depth = len(moleculess)

    delta = 1/n
    a = 0.3
    b = 0.15
    y0 = 0.5


    # draw rectangles
    for l in range(depth-1):
        molecules = moleculess[l+1]
        n_molecule = len(molecules)

        centers = []

        acc_arity = 0

        for i in range(n_molecule):
            start_id = len(flatten(molecules[:i]))
            end_id = len(flatten(molecules[:i+1]))

            center_x = (start_id + (end_id - 1 - start_id)/2) * delta + delta/2
            center_y = (l+1/2)*y0
            width_x = (end_id - start_id - 1 + 2*a)*delta
            width_y = 2*b

            # add text (numbers) on rectangles
            if style == 'box':
                myrectangle(center_x, center_y, width_x, width_y)
                plt.text(center_x, center_y, properties[l][i], fontsize=15, horizontalalignment='center',
                verticalalignment='center')
            elif style == 'tree':
                # if 'GS', no rectangle, n=arity tilted lines
                # if 'Id', no rectangle, n=arity vertical lines
                # if 'Add' or 'Mul'. rectangle, "+" or "x"
                # if '', rectangle
                property = properties[l][i]
                if property == 'GS' or property == 'Add' or property == 'Mul':
                    color = 'blue'
                    arity = arities[l][i]
                    for j in range(arity):

                        if l == 0:
                            # x = (start_id + j) * delta + delta/2, center_x
                            # y = center_y - b, center_y + b
                            plt.plot([(start_id + j) * delta + delta/2, center_x], [center_y - b, center_y + b], color=color)
                        else:
                            # x = last_centers[acc_arity:acc_arity+arity], center_x
                            # y = center_y - b, center_y + b
                            plt.plot([last_centers[acc_arity+j], center_x], [center_y - b, center_y + b], color=color)

                    acc_arity += arity

                    if property == 'Add' or property == 'Mul':
                        if property == 'Add':
                            symbol = '+'
                        else:
                            symbol = '*'

                        plt.text(center_x, center_y + b, symbol, horizontalalignment='center',
                verticalalignment='center', color='red', fontsize=40)
                if property == 'Id':
                    plt.plot([center_x, center_x], [center_y-width_y/2, center_y+width_y/2], color='black')
                    
                if property == '':
                    myrectangle(center_x, center_y, width_x, width_y)



            # connections to the next layer
            plt.plot([center_x, center_x], [center_y+width_y/2, center_y+y0-width_y/2], color='k')
            centers.append(center_x)
        last_centers = copy.deepcopy(centers)

    # connections from input variables to the first layer
    for i in range(n):
        x_ = (i + 1/2) * delta
        # connections to the next layer
        plt.plot([x_, x_], [0, y0/2-width_y/2], color='k')
        plt.text(x_, -0.05*(depth-1), f'${latex(in_vars[moleculess[0][i][0]])}$', fontsize=20, horizontalalignment='center')
    plt.xlim(0,1)
    #plt.ylim(0,1);
    plt.axis('off');
    plt.show()


def test_symmetry_var(model, x, input_vars, symmetry_var):
    
    orig_vars = input_vars
    sym_var = symmetry_var
    
    # gradients wrt to input (model)
    input_grad = batch_jacobian(model, x)

    # gradients wrt to input (symmetry var)
    func = lambdify(orig_vars, sym_var,'numpy') # returns a numpy-ready function

    func2 = lambda x: func(*[x[:,[i]] for i in range(len(orig_vars))])
    sym_grad = batch_jacobian(func2, x)
    
    # get id
    idx = []
    sym_symbols = list(sym_var.free_symbols)
    for sym_symbol in sym_symbols:
        for j in range(len(orig_vars)):
            if sym_symbol == orig_vars[j]:
                idx.append(j)
                
    input_grad_part = input_grad[:,idx]
    sym_grad_part = sym_grad[:,idx]
    
    cossim = torch.abs(torch.sum(input_grad_part * sym_grad_part, dim=1)/(torch.norm(input_grad_part, dim=1)*torch.norm(sym_grad_part, dim=1)))
    
    ratio = torch.sum(cossim > 0.9)/len(cossim)
    
    print(f'{100*ratio}% data have more than 0.9 cosine similarity')
    if ratio > 0.9:
        print('suggesting symmetry')
    else:
        print('not suggesting symmetry')
        
        
    
    return cossim