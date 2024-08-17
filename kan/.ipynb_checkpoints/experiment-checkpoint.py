import torch
from .MultKAN import *


def runner1(width, dataset, grids=[5,10,20], steps=20, lamb=0.001, prune_round=3, refine_round=3, edge_th=1e-2, node_th=1e-2, metrics=None, seed=1):

    result = {}
    result['test_loss'] = []
    result['c'] = []
    result['G'] = []
    result['id'] = []
    if metrics != None:
        for i in range(len(metrics)):
            result[metrics[i].__name__] = []

    def collect(evaluation):
        result['test_loss'].append(evaluation['test_loss'])
        result['c'].append(evaluation['n_edge'])
        result['G'].append(evaluation['n_grid'])
        result['id'].append(f'{model.round}.{model.state_id}')
        if metrics != None:
            for i in range(len(metrics)):
                result[metrics[i].__name__].append(metrics[i](model, dataset).item())

    for i in range(prune_round):
        # train and prune
        if i == 0:
            model = KAN(width=width, grid=grids[0], seed=seed)
        else:
            model = model.rewind(f'{i-1}.{2*i}')

        model.fit(dataset, steps=steps, lamb=lamb)
        model = model.prune(edge_th=edge_th, node_th=node_th)
        evaluation = model.evaluate(dataset)
        collect(evaluation)

        for j in range(refine_round):
            model = model.refine(grids[j])
            model.fit(dataset, steps=steps)
            evaluation = model.evaluate(dataset)
            collect(evaluation)

    for key in list(result.keys()):
        result[key] = np.array(result[key])

    return result


def pareto_frontier(x,y):

    pf_id = np.where(np.sum((x[:,None] <= x[None,:]) * (y[:,None] <= y[None,:]), axis=0) == 1)[0]
    x_pf = x[pf_id]
    y_pf = y[pf_id]

    return x_pf, y_pf, pf_id