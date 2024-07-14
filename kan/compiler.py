from sympy import *
import sympy
import numpy as np
from kan.MultKAN import MultKAN
import torch

def next_nontrivial_operation(expr, scale=1, bias=0):
    if expr.func == Add or expr.func == Mul:
        n_arg = len(expr.args)
        n_num = 0
        n_var_id = []
        n_num_id = []
        var_args = []
        for i in range(n_arg):
            is_number = expr.args[i].is_number
            n_num += is_number
            if not is_number:
                n_var_id.append(i)
                var_args.append(expr.args[i])
            else:
                n_num_id.append(i)
        if n_num > 0:
            # trivial
            if expr.func == Add:
                for i in range(n_num):
                    if i == 0:
                        bias = expr.args[n_num_id[i]]
                    else:
                        bias += expr.args[n_num_id[i]]
            if expr.func == Mul:
                for i in range(n_num):
                    if i == 0:
                        scale = expr.args[n_num_id[i]]
                    else:
                        scale *= expr.args[n_num_id[i]]
            
            return next_nontrivial_operation(expr.func(*var_args), scale, bias)
        else:
            return expr, scale, bias
    else:
        return expr, scale, bias
    

#def sf2kan(input_variables, expr, grid=3, k=3, noise_scale=0.1, scale_base_mu=0.0, scale_base_sigma=1.0, base_fun=torch.nn.SiLU(), symbolic_enabled=True, affine_trainable=False, grid_eps=1.0, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, device='cpu', seed=0):
def sf2kan(input_variables, expr, grid=5, k=3, auto_save=False):
    
    class Node:
        def __init__(self, expr, mult_bool, depth, scale, bias, parent=None, mult_arity=None):
            self.expr = expr
            self.mult_bool = mult_bool
            if self.mult_bool:
                self.mult_arity = mult_arity
            self.depth = depth

            if len(Nodes) <= depth:
                Nodes.append([])
                index = 0
            else:
                index = len(Nodes[depth])

            Nodes[depth].append(self)

            self.index = index
            if parent == None:
                self.parent_index = None
            else:
                self.parent_index = parent.index
            self.child_index = []

            # update parent's child_index
            if parent != None:
                parent.child_index.append(self.index)


            self.scale = scale
            self.bias = bias


    class SubNode:
        def __init__(self, expr, depth, scale, bias, parent=None):
            self.expr = expr
            self.depth = depth

            if len(SubNodes) <= depth:
                SubNodes.append([])
                index = 0
            else:
                index = len(SubNodes[depth])

            SubNodes[depth].append(self)

            self.index = index
            self.parent_index = None # shape: (2,)
            self.child_index = [] # shape: (n, 2)

            # update parent's child_index
            parent.child_index.append(self.index)

            self.scale = scale
            self.bias = bias


    class Connection:
        def __init__(self, affine, fun, fun_name, parent=None, child=None, power_exponent=None):
            # connection = activation function that connects a subnode to a node in the next layer node
            self.affine = affine #[1,0,1,0] # (a,b,c,d)
            self.fun = fun # y = c*fun(a*x+b)+d
            self.fun_name = fun_name
            self.parent_index = parent.index
            self.depth = parent.depth
            self.child_index = child.index
            self.power_exponent = power_exponent # if fun == Pow
            Connections[(self.depth,self.parent_index,self.child_index)] = self
            
    def create_node(expr, parent=None, n_layer=None):
        #print('before', expr)
        expr, scale, bias = next_nontrivial_operation(expr) 
        #print('after', expr)
        if parent == None:
            depth = 0
        else:
            depth = parent.depth


        if expr.func == Mul:
            mult_arity = len(expr.args)
            node = Node(expr, True, depth, scale, bias, parent=parent, mult_arity=mult_arity)
            # create mult_arity SubNodes, + 1
            for i in range(mult_arity):
                # create SubNode
                expr_i, scale, bias = next_nontrivial_operation(expr.args[i])
                subnode = SubNode(expr_i, node.depth+1, scale, bias, parent=node)
                if expr_i.func == Add:
                    for j in range(len(expr_i.args)):
                        expr_ij, scale, bias = next_nontrivial_operation(expr_i.args[j])
                        # expr_ij is impossible to be Add, should be Mul or 1D
                        if expr_ij.func == Mul:
                            #print(expr_ij)
                            # create a node with expr_ij
                            new_node = create_node(expr_ij, parent=subnode, n_layer=n_layer)
                            # create a connection which is a linear function
                            c = Connection([1,0,float(scale),float(bias)], lambda x: x, 'x', parent=subnode, child=new_node)

                        elif expr_ij.func == Symbol:
                            #print(expr_ij)
                            new_node = create_node(expr_ij, parent=subnode, n_layer=n_layer)
                            c = Connection([1,0,float(scale),float(bias)], lambda x: x, fun_name = 'x', parent=subnode, child=new_node)

                        else:
                            # 1D function case
                            # create a node with expr_ij.args[0]
                            new_node = create_node(expr_ij.args[0], parent=subnode, n_layer=n_layer)
                            # create 1D function expr_ij.func
                            if expr_ij.func == Pow:
                                power_exponent = expr_ij.args[1]
                            else:
                                power_exponent = None
                            Connection([1,0,float(scale),float(bias)], expr_ij.func, fun_name = expr_ij.func, parent=subnode, child=new_node, power_exponent=power_exponent)


                elif expr_i.func == Mul:
                    # create a node with expr_i
                    new_node = create_node(expr_i, parent=subnode, n_layer=n_layer)
                    # create 1D function, linear
                    Connection([1,0,1,0], lambda x: x, fun_name = 'x', parent=subnode, child=new_node)

                elif expr_i.func == Symbol:
                    new_node = create_node(expr_i, parent=subnode, n_layer=n_layer)
                    Connection([1,0,1,0], lambda x: x, fun_name = 'x', parent=subnode, child=new_node)

                else:
                    # 1D functions
                    # create a node with expr_i.args[0]
                    new_node = create_node(expr_i.args[0], parent=subnode, n_layer=n_layer)
                    # create 1D function expr_i.func
                    if expr_i.func == Pow:
                        power_exponent = expr_i.args[1]
                    else:
                        power_exponent = None
                    Connection([1,0,1,0], expr_i.func, fun_name = expr_i.func, parent=subnode, child=new_node, power_exponent=power_exponent)

        elif expr.func == Add:

            node = Node(expr, False, depth, scale, bias, parent=parent)
            subnode = SubNode(expr, node.depth+1, 1, 0, parent=node)

            for i in range(len(expr.args)):
                expr_i, scale, bias = next_nontrivial_operation(expr.args[i])
                if expr_i.func == Mul:
                    # create a node with expr_i
                    new_node = create_node(expr_i, parent=subnode, n_layer=n_layer)
                    # create a connection which is a linear function
                    Connection([1,0,float(scale),float(bias)], lambda x: x, fun_name = 'x', parent=subnode, child=new_node)

                elif expr_i.func == Symbol:
                    new_node = create_node(expr_i, parent=subnode, n_layer=n_layer)
                    Connection([1,0,float(scale),float(bias)], lambda x: x, fun_name = 'x', parent=subnode, child=new_node)

                else:
                    # 1D function case
                    # create a node with expr_ij.args[0]
                    new_node = create_node(expr_i.args[0], parent=subnode, n_layer=n_layer)
                    # create 1D function expr_i.func
                    if expr_i.func == Pow:
                        power_exponent = expr_i.args[1]
                    else:
                        power_exponent = None
                    Connection([1,0,float(scale),float(bias)], expr_i.func, fun_name = expr_i.func, parent=subnode, child=new_node, power_exponent=power_exponent)

        elif expr.func == Symbol:
            # expr.func is a symbol (one of input variables)
            if n_layer == None:
                node = Node(expr, False, depth, scale, bias, parent=parent)
            else:
                node = Node(expr, False, depth, scale, bias, parent=parent)
                return_node = node
                for i in range(n_layer - depth):
                    subnode = SubNode(expr, node.depth+1, 1, 0, parent=node)
                    node = Node(expr, False, subnode.depth, 1, 0, parent=subnode)
                    Connection([1,0,1,0], lambda x: x, fun_name = 'x', parent=subnode, child=node)
                node = return_node

            Start_Nodes.append(node)

        else:
            # expr.func is 1D function
            #print(expr, scale, bias)
            node = Node(expr, False, depth, scale, bias, parent=parent)
            expr_i, scale, bias = next_nontrivial_operation(expr.args[0])
            subnode = SubNode(expr_i, node.depth+1, 1, 0, parent=node)
            # create a node with expr_i.args[0]
            new_node = create_node(expr.args[0], parent=subnode, n_layer=n_layer)
            # create 1D function expr_i.func
            if expr.func == Pow:
                power_exponent = expr.args[1]
            else:
                power_exponent = None
            Connection([1,0,1,0], expr.func, fun_name = expr.func, parent=subnode, child=new_node, power_exponent=power_exponent)

        return node

    Nodes = [[]]
    SubNodes = [[]]
    Connections = {}
    Start_Nodes = []

    create_node(expr, n_layer=None)

    n_layer = len(Nodes) - 1

    Nodes = [[]]
    SubNodes = [[]]
    Connections = {}
    Start_Nodes = []

    create_node(expr, n_layer=n_layer)

    # move affine parameters in leaf nodes to connections
    for node in Start_Nodes:
        c = Connections[(node.depth,node.parent_index,node.index)]
        c.affine[0] = float(node.scale)
        c.affine[1] = float(node.bias)
        node.scale = 1.
        node.bias = 0.
        
    #input_variables = symbol
    node2var = []
    for node in Start_Nodes:
        for i in range(len(input_variables)):
            if node.expr == input_variables[i]:
                node2var.append(i)

    # Nodes 
    n_mult = []
    n_sum = []
    for layer in Nodes:
        n_mult.append(0)
        n_sum.append(0)
        for node in layer:
            if node.mult_bool == True:
                n_mult[-1] += 1
            else:
                n_sum[-1] += 1

    # depth
    n_layer = len(Nodes) - 1

    # converter
    # input tree node id, output kan node id (distinguish sum and mult node)
    # input tree subnode id, output tree subnode id
    # node id
    subnode_index_convert = {}
    node_index_convert = {}
    connection_index_convert = {}
    mult_arities = []
    for layer_id in range(n_layer+1):
        mult_arity = []
        i_sum = 0
        i_mult = 0
        for i in range(len(Nodes[layer_id])):
            node = Nodes[layer_id][i]
            if node.mult_bool == True:
                kan_node_id = n_sum[layer_id] + i_mult
                arity = len(node.child_index)
                for i in range(arity):
                    subnode = SubNodes[node.depth+1][node.child_index[i]]
                    kan_subnode_id = n_sum[layer_id] + np.sum(mult_arity) + i
                    subnode_index_convert[(subnode.depth,subnode.index)] = (int(n_layer-subnode.depth),int(kan_subnode_id))
                i_mult += 1
                mult_arity.append(arity)
            else:
                kan_node_id = i_sum
                if len(node.child_index) > 0:
                    subnode = SubNodes[node.depth+1][node.child_index[0]]
                    kan_subnode_id = i_sum
                    subnode_index_convert[(subnode.depth,subnode.index)] = (int(n_layer-subnode.depth),int(kan_subnode_id))
                i_sum += 1

            if layer_id == n_layer:
                # input layer
                node_index_convert[(node.depth,node.index)] = (int(n_layer-node.depth),int(node2var[kan_node_id]))
            else:
                node_index_convert[(node.depth,node.index)] = (int(n_layer-node.depth),int(kan_node_id))

            # node: depth (node.depth -> n_layer - node.depth)
            #       width (node.index -> kan_node_id)
            # subnode: depth (subnode.depth -> n_layer - subnode.depth)
            #          width (subnote.index -> kan_subnode_id)
        mult_arities.append(mult_arity)

    for index in list(Connections.keys()):
        depth, subnode_id, node_id = index
        # to int(n_layer-depth), 
        _, kan_subnode_id = subnode_index_convert[(depth, subnode_id)]
        _, kan_node_id = node_index_convert[(depth, node_id)]
        connection_index_convert[(depth, subnode_id, node_id)] = (n_layer-depth, kan_subnode_id, kan_node_id)


    n_sum.reverse()
    n_mult.reverse()
    mult_arities.reverse()
    
    width = [[n_sum[i], n_mult[i]] for i in range(len(n_sum))]
    width[0][0] = len(input_variables)

    # allow pass in other parameters (probably as a dictionary) in sf2kan, including grid k etc.
    model = MultKAN(width=width, mult_arity=mult_arities, grid=grid, k=k, auto_save=auto_save)
    
    # clean the graph
    for l in range(model.depth):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l+1]):
                model.fix_symbolic(l,i,j,'0',fit_params_bool=False)
                
    # Nodes
    Nodes_flat = [x for xs in Nodes for x in xs]

    self = model

    for node in Nodes_flat:
        node_depth = node.depth
        node_index = node.index
        kan_node_depth, kan_node_index = node_index_convert[(node_depth,node_index)]
        #print(kan_node_depth, kan_node_index)
        if kan_node_depth > 0:
            self.node_scale[kan_node_depth-1].data[kan_node_index] = float(node.scale)
            self.node_bias[kan_node_depth-1].data[kan_node_index] = float(node.bias)
            
    
    # SubNodes
    SubNodes_flat = [x for xs in SubNodes for x in xs]

    for subnode in SubNodes_flat:
        subnode_depth = subnode.depth
        subnode_index = subnode.index
        kan_subnode_depth, kan_subnode_index = subnode_index_convert[(subnode_depth,subnode_index)]
        #print(kan_subnode_depth, kan_subnode_index)
        self.subnode_scale[kan_subnode_depth].data[kan_subnode_index] = float(subnode.scale)
        self.subnode_bias[kan_subnode_depth].data[kan_subnode_index] = float(subnode.bias)
        
    # Connections
    Connections_flat = list(Connections.values())

    for connection in Connections_flat:
        c_depth = connection.depth
        c_j = connection.parent_index
        c_i = connection.child_index
        kc_depth, kc_j, kc_i = connection_index_convert[(c_depth, c_j, c_i)]

        # get symbolic fun_name
        fun_name = connection.fun_name
        #if fun_name == Pow:
        #    print(connection.power_exponent)

        if fun_name == 'x':
            kfun_name = 'x'
        elif fun_name == exp:
            kfun_name = 'exp'
        elif fun_name == sin:
            kfun_name = 'sin'
        elif fun_name == cos:
            kfun_name = 'cos'
        elif fun_name == tan:
            kfun_name = 'tan'
        elif fun_name == sqrt:
            kfun_name = 'sqrt'
        elif fun_name == log:
            kfun_name = 'log'
        elif fun_name == tanh:
            kfun_name = 'tanh'
        elif fun_name == asin:
            kfun_name = 'arcsin'
        elif fun_name == acos:
            kfun_name = 'arccos'
        elif fun_name == atan:
            kfun_name = 'arctan'
        elif fun_name == atanh:
            kfun_name = 'arctanh'
        elif fun_name == sign:
            kfun_name = 'sgn'
        elif fun_name == Pow:
            alpha = connection.power_exponent
            if alpha == Rational(1,2):
                kfun_name = 'x^0.5'
            elif alpha == - Rational(1,2):
                kfun_name = '1/x^0.5'
            elif alpha == Rational(3,2):
                kfun_name = 'x^1.5'
            else:
                alpha = int(connection.power_exponent)
                if alpha > 0:
                    if alpha == 1:
                        kfun_name = 'x'
                    else:
                        kfun_name = f'x^{alpha}'
                else:
                    if alpha == -1:
                        kfun_name = '1/x'
                    else:
                        kfun_name = f'1/x^{-alpha}'

        model.fix_symbolic(kc_depth, kc_i, kc_j, kfun_name, fit_params_bool=False)
        model.symbolic_fun[kc_depth].affine.data.reshape(self.width_out[kc_depth+1], self.width_in[kc_depth], 4)[kc_j][kc_i] = torch.tensor(connection.affine)
        
    return model
