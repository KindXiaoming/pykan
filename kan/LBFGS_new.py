import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from copy import deepcopy
from torch.optim import Optimizer


def is_legal(v):
    """
    Checks that tensor is not NaN or Inf.

    Inputs:
        v (tensor): tensor to be checked

    """
    legal = not torch.isnan(v).any() and not torch.isinf(v)

    return legal


def polyinterp(points, x_min_bound=None, x_max_bound=None, plot=False):
    """
    Gives the minimizer and minimum of the interpolating polynomial over given points
    based on function and derivative information. Defaults to bisection if no critical
    points are valid.

    Based on polyinterp.m Matlab function in minFunc by Mark Schmidt with some slight
    modifications.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 12/6/18.

    Inputs:
        points (nparray): two-dimensional array with each point of form [x f g]
        x_min_bound (float): minimum value that brackets minimum (default: minimum of points)
        x_max_bound (float): maximum value that brackets minimum (default: maximum of points)
        plot (bool): plot interpolating polynomial

    Outputs:
        x_sol (float): minimizer of interpolating polynomial
        F_min (float): minimum of interpolating polynomial

    Note:
      . Set f or g to np.nan if they are unknown

    """
    no_points = points.shape[0]
    order = np.sum(1 - np.isnan(points[:, 1:3]).astype('int')) - 1

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])

    # compute bounds of interpolation area
    if x_min_bound is None:
        x_min_bound = x_min
    if x_max_bound is None:
        x_max_bound = x_max

    # explicit formula for quadratic interpolation
    if no_points == 2 and order == 2 and plot is False:
        # Solution to quadratic interpolation is given by:
        # a = -(f1 - f2 - g1(x1 - x2))/(x1 - x2)^2
        # x_min = x1 - g1/(2a)
        # if x1 = 0, then is given by:
        # x_min = - (g1*x2^2)/(2(f2 - f1 - g1*x2))

        if points[0, 0] == 0:
            x_sol = -points[0, 2] * points[1, 0] ** 2 / (2 * (points[1, 1] - points[0, 1] - points[0, 2] * points[1, 0]))
        else:
            a = -(points[0, 1] - points[1, 1] - points[0, 2] * (points[0, 0] - points[1, 0])) / (points[0, 0] - points[1, 0]) ** 2
            x_sol = points[0, 0] - points[0, 2]/(2*a)

        x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)

    # explicit formula for cubic interpolation
    elif no_points == 2 and order == 3 and plot is False:
        # Solution to cubic interpolation is given by:
        # d1 = g1 + g2 - 3((f1 - f2)/(x1 - x2))
        # d2 = sqrt(d1^2 - g1*g2)
        # x_min = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2))
        d1 = points[0, 2] + points[1, 2] - 3 * ((points[0, 1] - points[1, 1]) / (points[0, 0] - points[1, 0]))
        d2 = np.sqrt(d1 ** 2 - points[0, 2] * points[1, 2])
        if np.isreal(d2):
            x_sol = points[1, 0] - (points[1, 0] - points[0, 0]) * ((points[1, 2] + d2 - d1) / (points[1, 2] - points[0, 2] + 2 * d2))
            x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)
        else:
            x_sol = (x_max_bound + x_min_bound)/2

    # solve linear system
    else:
        # define linear constraints
        A = np.zeros((0, order + 1))
        b = np.zeros((0, 1))

        # add linear constraints on function values
        for i in range(no_points):
            if not np.isnan(points[i, 1]):
                constraint = np.zeros((1, order + 1))
                for j in range(order, -1, -1):
                    constraint[0, order - j] = points[i, 0] ** j
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 1])

        # add linear constraints on gradient values
        for i in range(no_points):
            if not np.isnan(points[i, 2]):
                constraint = np.zeros((1, order + 1))
                for j in range(order):
                    constraint[0, j] = (order - j) * points[i, 0] ** (order - j - 1)
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 2])

        # check if system is solvable
        if A.shape[0] != A.shape[1] or np.linalg.matrix_rank(A) != A.shape[0]:
            x_sol = (x_min_bound + x_max_bound)/2
            f_min = np.Inf
        else:
            # solve linear system for interpolating polynomial
            coeff = np.linalg.solve(A, b)

            # compute critical points
            dcoeff = np.zeros(order)
            for i in range(len(coeff) - 1):
                dcoeff[i] = coeff[i] * (order - i)

            crit_pts = np.array([x_min_bound, x_max_bound])
            crit_pts = np.append(crit_pts, points[:, 0])

            if not np.isinf(dcoeff).any():
                roots = np.roots(dcoeff)
                crit_pts = np.append(crit_pts, roots)

            # test critical points
            f_min = np.Inf
            x_sol = (x_min_bound + x_max_bound) / 2 # defaults to bisection
            for crit_pt in crit_pts:
                if np.isreal(crit_pt) and crit_pt >= x_min_bound and crit_pt <= x_max_bound:
                    F_cp = np.polyval(coeff, crit_pt)
                    if np.isreal(F_cp) and F_cp < f_min:
                        x_sol = np.real(crit_pt)
                        f_min = np.real(F_cp)

            if(plot):
                plt.figure()
                x = np.arange(x_min_bound, x_max_bound, (x_max_bound - x_min_bound)/10000)
                f = np.polyval(coeff, x)
                plt.plot(x, f)
                plt.plot(x_sol, f_min, 'x')

    return x_sol


class LBFGS(Optimizer):
    """
    Implements the L-BFGS algorithm. Compatible with multi-batch and full-overlap
    L-BFGS implementations and (stochastic) Powell damping. Partly based on the 
    original L-BFGS implementation in PyTorch, Mark Schmidt's minFunc MATLAB code, 
    and Michael Overton's weak Wolfe line search MATLAB code.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 10/20/20.

    Warnings:
      . Does not support per-parameter options and parameter groups.
      . All parameters have to be on a single device.

    Inputs:
        lr (float): steplength or learning rate (default: 1)
        history_size (int): update history size (default: 10)
        line_search (str): designates line search to use (default: 'Wolfe')
            Options:
                'None': uses steplength designated in algorithm
                'Armijo': uses Armijo backtracking line search
                'Wolfe': uses Armijo-Wolfe bracketing line search
        dtype: data type (default: torch.float)
        debug (bool): debugging mode

    References:
    [1] Berahas, Albert S., Jorge Nocedal, and Martin Takác. "A Multi-Batch L-BFGS 
        Method for Machine Learning." Advances in Neural Information Processing 
        Systems. 2016.
    [2] Bollapragada, Raghu, et al. "A Progressive Batching L-BFGS Method for Machine 
        Learning." International Conference on Machine Learning. 2018.
    [3] Lewis, Adrian S., and Michael L. Overton. "Nonsmooth Optimization via Quasi-Newton
        Methods." Mathematical Programming 141.1-2 (2013): 135-163.
    [4] Liu, Dong C., and Jorge Nocedal. "On the Limited Memory BFGS Method for 
        Large Scale Optimization." Mathematical Programming 45.1-3 (1989): 503-528.
    [5] Nocedal, Jorge. "Updating Quasi-Newton Matrices With Limited Storage." 
        Mathematics of Computation 35.151 (1980): 773-782.
    [6] Nocedal, Jorge, and Stephen J. Wright. "Numerical Optimization." Springer New York,
        2006.
    [7] Schmidt, Mark. "minFunc: Unconstrained Differentiable Multivariate Optimization 
        in Matlab." Software available at http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html 
        (2005).
    [8] Schraudolph, Nicol N., Jin Yu, and Simon Günter. "A Stochastic Quasi-Newton 
        Method for Online Convex Optimization." Artificial Intelligence and Statistics. 
        2007.
    [9] Wang, Xiao, et al. "Stochastic Quasi-Newton Methods for Nonconvex Stochastic 
        Optimization." SIAM Journal on Optimization 27.2 (2017): 927-956.

    """

    def __init__(self, params, lr=1., history_size=10, line_search='Wolfe',
                 dtype=torch.float, debug=False):

        # ensure inputs are valid
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 <= history_size:
            raise ValueError("Invalid history size: {}".format(history_size))
        if line_search not in ['Armijo', 'Wolfe', 'None']:
            raise ValueError("Invalid line search: {}".format(line_search))

        defaults = dict(lr=lr, history_size=history_size, line_search=line_search, dtype=dtype, debug=debug)
        super(LBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("L-BFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        state = self.state['global_state']
        state.setdefault('n_iter', 0)
        state.setdefault('curv_skips', 0)
        state.setdefault('fail_skips', 0)
        state.setdefault('H_diag',1)
        state.setdefault('fail', True)

        state['old_dirs'] = []
        state['old_stps'] = []

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_update(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def _copy_params(self):
        current_params = []
        for param in self._params:
            current_params.append(deepcopy(param.data))
        return current_params

    def _load_params(self, current_params):
        i = 0
        for param in self._params:
            param.data[:] = current_params[i]
            i += 1

    def line_search(self, line_search):
        """
        Switches line search option.
        
        Inputs:
            line_search (str): designates line search to use
                Options:
                    'None': uses steplength designated in algorithm
                    'Armijo': uses Armijo backtracking line search
                    'Wolfe': uses Armijo-Wolfe bracketing line search
        
        """
        
        group = self.param_groups[0]
        group['line_search'] = line_search
        
        return

    def two_loop_recursion(self, vec):
        """
        Performs two-loop recursion on given vector to obtain Hv.

        Inputs:
            vec (tensor): 1-D tensor to apply two-loop recursion to

        Output:
            r (tensor): matrix-vector product Hv

        """

        group = self.param_groups[0]
        history_size = group['history_size']

        state = self.state['global_state']
        old_dirs = state.get('old_dirs')    # change in gradients
        old_stps = state.get('old_stps')    # change in iterates
        H_diag = state.get('H_diag')

        # compute the product of the inverse Hessian approximation and the gradient
        num_old = len(old_dirs)

        if 'rho' not in state:
            state['rho'] = [None] * history_size
            state['alpha'] = [None] * history_size
        rho = state['rho']
        alpha = state['alpha']

        for i in range(num_old):
            rho[i] = 1. / old_stps[i].dot(old_dirs[i])

        q = vec
        for i in range(num_old - 1, -1, -1):
            alpha[i] = old_dirs[i].dot(q) * rho[i]
            q.add_(-alpha[i], old_stps[i])

        # multiply by initial Hessian 
        # r/d is the final direction
        r = torch.mul(q, H_diag)
        for i in range(num_old):
            beta = old_stps[i].dot(r) * rho[i]
            r.add_(alpha[i] - beta, old_dirs[i])

        return r

    def curvature_update(self, flat_grad, eps=1e-2, damping=False):
        """
        Performs curvature update.

        Inputs:
            flat_grad (tensor): 1-D tensor of flattened gradient for computing 
                gradient difference with previously stored gradient
            eps (float): constant for curvature pair rejection or damping (default: 1e-2)
            damping (bool): flag for using Powell damping (default: False)
        """

        assert len(self.param_groups) == 1

        # load parameters
        if(eps <= 0):
            raise(ValueError('Invalid eps; must be positive.'))

        group = self.param_groups[0]
        history_size = group['history_size']
        debug = group['debug']

        # variables cached in state (for tracing)
        state = self.state['global_state']
        fail = state.get('fail')
        
        # check if line search failed
        if not fail:
            
            d = state.get('d')
            t = state.get('t')
            old_dirs = state.get('old_dirs')
            old_stps = state.get('old_stps')
            H_diag = state.get('H_diag')
            prev_flat_grad = state.get('prev_flat_grad')
            Bs = state.get('Bs')
    
            # compute y's
            y = flat_grad.sub(prev_flat_grad)
            s = d.mul(t)
            sBs = s.dot(Bs)
            ys = y.dot(s)  # y*s

            # update L-BFGS matrix
            if ys > eps * sBs or damping == True:
    
                # perform Powell damping
                if damping == True and ys < eps*sBs:
                    if debug:
                        print('Applying Powell damping...')
                    theta = ((1 - eps) * sBs)/(sBs - ys)
                    y = theta * y + (1 - theta) * Bs
    
                # updating memory
                if len(old_dirs) == history_size:
                    # shift history by one (limited-memory)
                    old_dirs.pop(0)
                    old_stps.pop(0)
    
                # store new direction/step
                old_dirs.append(s)
                old_stps.append(y)
    
                # update scale of initial Hessian approximation
                H_diag = ys / y.dot(y)  # (y*y)
                
                state['old_dirs'] = old_dirs
                state['old_stps'] = old_stps
                state['H_diag'] = H_diag

            else:
                # save skip
                state['curv_skips'] += 1
                if debug:
                    print('Curvature pair skipped due to failed criterion')

        else:
            # save skip
            state['fail_skips'] += 1
            if debug:
                print('Line search failed; curvature pair update skipped')

        return

    def _step(self, p_k, g_Ok, g_Sk=None, options=None):
        """
        Performs a single optimization step.

        Inputs:
            p_k (tensor): 1-D tensor specifying search direction
            g_Ok (tensor): 1-D tensor of flattened gradient over overlap O_k used
                            for gradient differencing in curvature pair update
            g_Sk (tensor): 1-D tensor of flattened gradient over full sample S_k
                            used for curvature pair damping or rejection criterion,
                            if None, will use g_Ok (default: None)
            options (dict): contains options for performing line search (default: None)

        Options for Armijo backtracking line search:
            'closure' (callable): reevaluates model and returns function value
            'current_loss' (tensor): objective value at current iterate (default: F(x_k))
            'gtd' (tensor): inner product g_Ok'd in line search (default: g_Ok'd)
            'eta' (tensor): factor for decreasing steplength > 0 (default: 2)
            'c1' (tensor): sufficient decrease constant in (0, 1) (default: 1e-4)
            'max_ls' (int): maximum number of line search steps permitted (default: 10)
            'interpolate' (bool): flag for using interpolation (default: True)
            'inplace' (bool): flag for inplace operations (default: True)
            'ls_debug' (bool): debugging mode for line search

        Options for Wolfe line search:
            'closure' (callable): reevaluates model and returns function value
            'current_loss' (tensor): objective value at current iterate (default: F(x_k))
            'gtd' (tensor): inner product g_Ok'd in line search (default: g_Ok'd)
            'eta' (float): factor for extrapolation (default: 2)
            'c1' (float): sufficient decrease constant in (0, 1) (default: 1e-4)
            'c2' (float): curvature condition constant in (0, 1) (default: 0.9)
            'max_ls' (int): maximum number of line search steps permitted (default: 10)
            'interpolate' (bool): flag for using interpolation (default: True)
            'inplace' (bool): flag for inplace operations (default: True)
            'ls_debug' (bool): debugging mode for line search

        Outputs (depends on line search):
          . No line search:
                t (float): steplength
          . Armijo backtracking line search:
                F_new (tensor): loss function at new iterate
                t (tensor): final steplength
                ls_step (int): number of backtracks
                closure_eval (int): number of closure evaluations
                desc_dir (bool): descent direction flag
                    True: p_k is descent direction with respect to the line search
                    function
                    False: p_k is not a descent direction with respect to the line
                    search function
                fail (bool): failure flag
                    True: line search reached maximum number of iterations, failed
                    False: line search succeeded
          . Wolfe line search:
                F_new (tensor): loss function at new iterate
                g_new (tensor): gradient at new iterate
                t (float): final steplength
                ls_step (int): number of backtracks
                closure_eval (int): number of closure evaluations
                grad_eval (int): number of gradient evaluations
                desc_dir (bool): descent direction flag
                    True: p_k is descent direction with respect to the line search
                    function
                    False: p_k is not a descent direction with respect to the line
                    search function
                fail (bool): failure flag
                    True: line search reached maximum number of iterations, failed
                    False: line search succeeded

        Notes:
          . If encountering line search failure in the deterministic setting, one
            should try increasing the maximum number of line search steps max_ls.

        """

        if options is None:
            options = {}
        assert len(self.param_groups) == 1

        # load parameter options
        group = self.param_groups[0]
        lr = group['lr']
        line_search = group['line_search']
        dtype = group['dtype']
        debug = group['debug']

        # variables cached in state (for tracing)
        state = self.state['global_state']
        d = state.get('d')
        t = state.get('t')
        prev_flat_grad = state.get('prev_flat_grad')
        Bs = state.get('Bs')

        # keep track of nb of iterations
        state['n_iter'] += 1

        # set search direction
        d = p_k

        # modify previous gradient
        if prev_flat_grad is None:
            prev_flat_grad = g_Ok.clone()
        else:
            prev_flat_grad.copy_(g_Ok)

        # set initial step size
        t = lr

        # closure evaluation counter
        closure_eval = 0

        if g_Sk is None:
            g_Sk = g_Ok.clone()

        # perform Armijo backtracking line search
        if line_search == 'Armijo':

            # load options
            if options:
                if 'closure' not in options.keys():
                    raise(ValueError('closure option not specified.'))
                else:
                    closure = options['closure']

                if 'gtd' not in options.keys():
                    gtd = g_Sk.dot(d)
                else:
                    gtd = options['gtd']

                if 'current_loss' not in options.keys():
                    F_k = closure()
                    closure_eval += 1
                else:
                    F_k = options['current_loss']

                if 'eta' not in options.keys():
                    eta = 2
                elif options['eta'] <= 0:
                    raise(ValueError('Invalid eta; must be positive.'))
                else:
                    eta = options['eta']

                if 'c1' not in options.keys():
                    c1 = 1e-4
                elif options['c1'] >= 1 or options['c1'] <= 0:
                    raise(ValueError('Invalid c1; must be strictly between 0 and 1.'))
                else:
                    c1 = options['c1']

                if 'max_ls' not in options.keys():
                    max_ls = 10
                elif options['max_ls'] <= 0:
                    raise(ValueError('Invalid max_ls; must be positive.'))
                else:
                    max_ls = options['max_ls']

                if 'interpolate' not in options.keys():
                    interpolate = True
                else:
                    interpolate = options['interpolate']

                if 'inplace' not in options.keys():
                    inplace = True
                else:
                    inplace = options['inplace']
                    
                if 'ls_debug' not in options.keys():
                    ls_debug = False
                else:
                    ls_debug = options['ls_debug']

            else:
                raise(ValueError('Options are not specified; need closure evaluating function.'))

            # initialize values
            if interpolate:
                if torch.cuda.is_available():
                    F_prev = torch.tensor(np.nan).cuda()
                else:
                    F_prev = torch.tensor(np.nan)

            ls_step = 0
            t_prev = 0 # old steplength
            fail = False # failure flag

            # begin print for debug mode
            if ls_debug:
                print('==================================== Begin Armijo line search ===================================')
                print('F(x): %.8e  g*d: %.8e' % (F_k, gtd))

            # check if search direction is descent direction
            if gtd >= 0:
                desc_dir = False
                if debug:
                    print('Not a descent direction!')
            else:
                desc_dir = True

            # store values if not in-place
            if not inplace:
                current_params = self._copy_params()

            # update and evaluate at new point
            self._add_update(t, d)
            F_new = closure()
            closure_eval += 1

            # print info if debugging
            if ls_debug:
                print('LS Step: %d  t: %.8e  F(x+td): %.8e  F-c1*t*g*d: %.8e  F(x): %.8e'
                      % (ls_step, t, F_new, F_k + c1 * t * gtd, F_k))

            # check Armijo condition
            while F_new > F_k + c1*t*gtd or not is_legal(F_new):

                # check if maximum number of iterations reached
                if ls_step >= max_ls:
                    if inplace:
                        self._add_update(-t, d)
                    else:
                        self._load_params(current_params)

                    t = 0
                    F_new = closure()
                    closure_eval += 1
                    fail = True
                    break

                else:
                    # store current steplength
                    t_new = t

                    # compute new steplength

                    # if first step or not interpolating, then multiply by factor
                    if ls_step == 0 or not interpolate or not is_legal(F_new):
                        t = t/eta

                    # if second step, use function value at new point along with 
                    # gradient and function at current iterate
                    elif ls_step == 1 or not is_legal(F_prev):
                        t = polyinterp(np.array([[0, F_k.item(), gtd.item()], [t_new, F_new.item(), np.nan]]))

                    # otherwise, use function values at new point, previous point,
                    # and gradient and function at current iterate
                    else:
                        t = polyinterp(np.array([[0, F_k.item(), gtd.item()], [t_new, F_new.item(), np.nan], 
                                                [t_prev, F_prev.item(), np.nan]]))

                    # if values are too extreme, adjust t
                    if interpolate:
                        if t < 1e-3 * t_new:
                            t = 1e-3 * t_new
                        elif t > 0.6 * t_new:
                            t = 0.6 * t_new

                        # store old point
                        F_prev = F_new
                        t_prev = t_new

                    # update iterate and reevaluate
                    if inplace:
                        self._add_update(t - t_new, d)
                    else:
                        self._load_params(current_params)
                        self._add_update(t, d)

                    F_new = closure()
                    closure_eval += 1
                    ls_step += 1 # iterate
                    
                    # print info if debugging
                    if ls_debug:
                        print('LS Step: %d  t: %.8e  F(x+td):   %.8e  F-c1*t*g*d: %.8e  F(x): %.8e'
                              % (ls_step, t, F_new, F_k + c1 * t * gtd, F_k))

            # store Bs
            if Bs is None:
                Bs = (g_Sk.mul(-t)).clone()
            else:
                Bs.copy_(g_Sk.mul(-t))
                
            # print final steplength
            if ls_debug:
                print('Final Steplength:', t)
                print('===================================== End Armijo line search ====================================')

            state['d'] = d
            state['prev_flat_grad'] = prev_flat_grad
            state['t'] = t
            state['Bs'] = Bs
            state['fail'] = fail

            return F_new, t, ls_step, closure_eval, desc_dir, fail

        # perform weak Wolfe line search
        elif line_search == 'Wolfe':

            # load options
            if options:
                if 'closure' not in options.keys():
                    raise(ValueError('closure option not specified.'))
                else:
                    closure = options['closure']

                if 'current_loss' not in options.keys():
                    F_k = closure()
                    closure_eval += 1
                else:
                    F_k = options['current_loss']

                if 'gtd' not in options.keys():
                    gtd = g_Sk.dot(d)
                else:
                    gtd = options['gtd']

                if 'eta' not in options.keys():
                    eta = 2
                elif options['eta'] <= 1:
                    raise(ValueError('Invalid eta; must be greater than 1.'))
                else:
                    eta = options['eta']

                if 'c1' not in options.keys():
                    c1 = 1e-4
                elif options['c1'] >= 1 or options['c1'] <= 0:
                    raise(ValueError('Invalid c1; must be strictly between 0 and 1.'))
                else:
                    c1 = options['c1']

                if 'c2' not in options.keys():
                    c2 = 0.9
                elif options['c2'] >= 1 or options['c2'] <= 0:
                    raise(ValueError('Invalid c2; must be strictly between 0 and 1.'))
                elif options['c2'] <= c1:
                    raise(ValueError('Invalid c2; must be strictly larger than c1.'))
                else:
                    c2 = options['c2']

                if 'max_ls' not in options.keys():
                    max_ls = 10
                elif options['max_ls'] <= 0:
                    raise(ValueError('Invalid max_ls; must be positive.'))
                else:
                    max_ls = options['max_ls']

                if 'interpolate' not in options.keys():
                    interpolate = True
                else:
                    interpolate = options['interpolate']

                if 'inplace' not in options.keys():
                    inplace = True
                else:
                    inplace = options['inplace']
                    
                if 'ls_debug' not in options.keys():
                    ls_debug = False
                else:
                    ls_debug = options['ls_debug']

            else:
                raise(ValueError('Options are not specified; need closure evaluating function.'))

            # initialize counters
            ls_step = 0
            grad_eval = 0 # tracks gradient evaluations
            t_prev = 0 # old steplength

            # initialize bracketing variables and flag
            alpha = 0
            beta = float('Inf')
            fail = False

            # initialize values for line search
            if(interpolate):
                F_a = F_k
                g_a = gtd

                if(torch.cuda.is_available()):
                    F_b = torch.tensor(np.nan).cuda()
                    g_b = torch.tensor(np.nan).cuda()
                else:
                    F_b = torch.tensor(np.nan)
                    g_b = torch.tensor(np.nan)

            # begin print for debug mode
            if ls_debug:
                print('==================================== Begin Wolfe line search ====================================')
                print('F(x): %.8e  g*d: %.8e' % (F_k, gtd))

            # check if search direction is descent direction
            if gtd >= 0:
                desc_dir = False
                if debug:
                    print('Not a descent direction!')
            else:
                desc_dir = True

            # store values if not in-place
            if not inplace:
                current_params = self._copy_params()

            # update and evaluate at new point
            self._add_update(t, d)
            F_new = closure()
            closure_eval += 1

            # main loop
            while True:

                # check if maximum number of line search steps have been reached
                if ls_step >= max_ls:
                    if inplace:
                        self._add_update(-t, d)
                    else:
                        self._load_params(current_params)

                    t = 0
                    F_new = closure()
                    F_new.backward()
                    g_new = self._gather_flat_grad()
                    closure_eval += 1
                    grad_eval += 1
                    fail = True
                    break

                # print info if debugging
                if ls_debug:
                    print('LS Step: %d  t: %.8e  alpha: %.8e  beta: %.8e' 
                          % (ls_step, t, alpha, beta))
                    print('Armijo:  F(x+td): %.8e  F-c1*t*g*d: %.8e  F(x): %.8e'
                          % (F_new, F_k + c1 * t * gtd, F_k))

                # check Armijo condition
                if F_new > F_k + c1 * t * gtd:

                    # set upper bound
                    beta = t
                    t_prev = t

                    # update interpolation quantities
                    if interpolate:
                        F_b = F_new
                        if torch.cuda.is_available():
                            g_b = torch.tensor(np.nan).cuda()
                        else:
                            g_b = torch.tensor(np.nan)

                else:

                    # compute gradient
                    F_new.backward()
                    g_new = self._gather_flat_grad()
                    grad_eval += 1
                    gtd_new = g_new.dot(d)
                    
                    # print info if debugging
                    if ls_debug:
                        print('Wolfe: g(x+td)*d: %.8e  c2*g*d: %.8e  gtd: %.8e'
                              % (gtd_new, c2 * gtd, gtd))

                    # check curvature condition
                    if gtd_new < c2 * gtd:

                        # set lower bound
                        alpha = t
                        t_prev = t

                        # update interpolation quantities
                        if interpolate:
                            F_a = F_new
                            g_a = gtd_new

                    else:
                        break

                # compute new steplength

                # if first step or not interpolating, then bisect or multiply by factor
                if not interpolate or not is_legal(F_b):
                    if beta == float('Inf'):
                        t = eta*t
                    else:
                        t = (alpha + beta)/2.0

                # otherwise interpolate between a and b
                else:
                    t = polyinterp(np.array([[alpha, F_a.item(), g_a.item()], [beta, F_b.item(), g_b.item()]]))

                    # if values are too extreme, adjust t
                    if beta == float('Inf'):
                        if t > 2 * eta * t_prev:
                            t = 2 * eta * t_prev
                        elif t < eta * t_prev:
                            t = eta * t_prev
                    else:
                        if t < alpha + 0.2 * (beta - alpha):
                            t = alpha + 0.2 * (beta - alpha)
                        elif t > (beta - alpha) / 2.0:
                            t = (beta - alpha) / 2.0

                    # if we obtain nonsensical value from interpolation
                    if t <= 0:
                        t = (beta - alpha) / 2.0

                # update parameters
                if inplace:
                    self._add_update(t - t_prev, d)
                else:
                    self._load_params(current_params)
                    self._add_update(t, d)

                # evaluate closure
                F_new = closure()
                closure_eval += 1
                ls_step += 1

            # store Bs
            if Bs is None:
                Bs = (g_Sk.mul(-t)).clone()
            else:
                Bs.copy_(g_Sk.mul(-t))
                
            # print final steplength
            if ls_debug:
                print('Final Steplength:', t)
                print('===================================== End Wolfe line search =====================================')

            state['d'] = d
            state['prev_flat_grad'] = prev_flat_grad
            state['t'] = t
            state['Bs'] = Bs
            state['fail'] = fail

            return F_new, g_new, t, ls_step, closure_eval, grad_eval, desc_dir, fail

        else:

            # perform update
            self._add_update(t, d)

            # store Bs
            if Bs is None:
                Bs = (g_Sk.mul(-t)).clone()
            else:
                Bs.copy_(g_Sk.mul(-t))

            state['d'] = d
            state['prev_flat_grad'] = prev_flat_grad
            state['t'] = t
            state['Bs'] = Bs
            state['fail'] = False

            return t
        
    def step(self, p_k, g_Ok, g_Sk=None, options={}):
        return self._step(p_k, g_Ok, g_Sk, options)


class FullBatchLBFGS(LBFGS):
    """
    Implements full-batch or deterministic L-BFGS algorithm. Compatible with
    Powell damping. Can be used when evaluating a deterministic function and
    gradient. Wraps the LBFGS optimizer. Performs the two-loop recursion,
    updating, and curvature updating in a single step.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 11/15/18.

    Warnings:
      . Does not support per-parameter options and parameter groups.
      . All parameters have to be on a single device.

    Inputs:
        lr (float): steplength or learning rate (default: 1)
        history_size (int): update history size (default: 10)
        line_search (str): designates line search to use (default: 'Wolfe')
            Options:
                'None': uses steplength designated in algorithm
                'Armijo': uses Armijo backtracking line search
                'Wolfe': uses Armijo-Wolfe bracketing line search
        dtype: data type (default: torch.float)
        debug (bool): debugging mode

    """

    def __init__(self, params, lr=1, history_size=10, line_search='Wolfe', 
                 dtype=torch.float, debug=False):
        super(FullBatchLBFGS, self).__init__(params, lr, history_size, line_search, 
             dtype, debug)

    def step(self, options=None):
        """
        Performs a single optimization step.

        Inputs:
            options (dict): contains options for performing line search (default: None)
            
        General Options:
            'eps' (float): constant for curvature pair rejection or damping (default: 1e-2)
            'damping' (bool): flag for using Powell damping (default: False)

        Options for Armijo backtracking line search:
            'closure' (callable): reevaluates model and returns function value
            'current_loss' (tensor): objective value at current iterate (default: F(x_k))
            'gtd' (tensor): inner product g_Ok'd in line search (default: g_Ok'd)
            'eta' (tensor): factor for decreasing steplength > 0 (default: 2)
            'c1' (tensor): sufficient decrease constant in (0, 1) (default: 1e-4)
            'max_ls' (int): maximum number of line search steps permitted (default: 10)
            'interpolate' (bool): flag for using interpolation (default: True)
            'inplace' (bool): flag for inplace operations (default: True)
            'ls_debug' (bool): debugging mode for line search

        Options for Wolfe line search:
            'closure' (callable): reevaluates model and returns function value
            'current_loss' (tensor): objective value at current iterate (default: F(x_k))
            'gtd' (tensor): inner product g_Ok'd in line search (default: g_Ok'd)
            'eta' (float): factor for extrapolation (default: 2)
            'c1' (float): sufficient decrease constant in (0, 1) (default: 1e-4)
            'c2' (float): curvature condition constant in (0, 1) (default: 0.9)
            'max_ls' (int): maximum number of line search steps permitted (default: 10)
            'interpolate' (bool): flag for using interpolation (default: True)
            'inplace' (bool): flag for inplace operations (default: True)
            'ls_debug' (bool): debugging mode for line search

        Outputs (depends on line search):
          . No line search:
                t (float): steplength
          . Armijo backtracking line search:
                F_new (tensor): loss function at new iterate
                t (tensor): final steplength
                ls_step (int): number of backtracks
                closure_eval (int): number of closure evaluations
                desc_dir (bool): descent direction flag
                    True: p_k is descent direction with respect to the line search
                    function
                    False: p_k is not a descent direction with respect to the line
                    search function
                fail (bool): failure flag
                    True: line search reached maximum number of iterations, failed
                    False: line search succeeded
          . Wolfe line search:
                F_new (tensor): loss function at new iterate
                g_new (tensor): gradient at new iterate
                t (float): final steplength
                ls_step (int): number of backtracks
                closure_eval (int): number of closure evaluations
                grad_eval (int): number of gradient evaluations
                desc_dir (bool): descent direction flag
                    True: p_k is descent direction with respect to the line search
                    function
                    False: p_k is not a descent direction with respect to the line
                    search function
                fail (bool): failure flag
                    True: line search reached maximum number of iterations, failed
                    False: line search succeeded

        Notes:
          . If encountering line search failure in the deterministic setting, one
            should try increasing the maximum number of line search steps max_ls.

        """
        
        # load options for damping and eps
        if 'damping' not in options.keys():
            damping = False
        else:
            damping = options['damping']
            
        if 'eps' not in options.keys():
            eps = 1e-2
        else:
            eps = options['eps']
        
        # gather gradient
        grad = self._gather_flat_grad()
        
        # update curvature if after 1st iteration
        state = self.state['global_state']
        if state['n_iter'] > 0:
            self.curvature_update(grad, eps, damping)

        # compute search direction
        p = self.two_loop_recursion(-grad)

        # take step
        return self._step(p, grad, options=options)