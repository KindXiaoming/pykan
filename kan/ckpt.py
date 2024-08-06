import yaml
from .MultKAN import MultKAN
import torch
from .utils import SYMBOLIC_LIB

def saveckpt(model, path='model'):
    
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
        device = model.device,
        state_id = model.state_id,
        auto_save = model.auto_save,
        ckpt_path = model.ckpt_path
    )

    for i in range (model.depth):
        dic[f'symbolic.funs_name.{i}'] = model.symbolic_fun[i].funs_name

    with open(f'{path}_config.yml', 'w') as outfile:
        yaml.dump(dic, outfile, default_flow_style=False)

    torch.save(model.state_dict(), f'{path}_state')

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
                 ckpt_path=config['ckpt_path'])

    model_load.load_state_dict(state)
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