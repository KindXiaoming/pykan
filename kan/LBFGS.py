import ast
import inspect
from functools import partial, reduce

import torch
from torch.optim import LBFGS as _LBFGS


class LBFGS(_LBFGS):
    def __init__(
        self,
        params,
        lr=1,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        tolerance_ys=1e-32,
        history_size=100,
        line_search_fn=None,
    ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            tolerance_ys=tolerance_ys,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        super(_LBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

        # ----------------------------------------------------------------------
        # using ast, modify the method step(self, closure)
        # to avoid changing the original source code
        # fix: modify `if ys > 1e-10:` by `if ys > groupe['tolerance_ys']:`

        step_code = inspect.getsource(self.step)
        # remove one indentation
        step_code = "\n".join(line[4:] for line in step_code.split("\n")[1:])
        self.step_tree = ast.parse(step_code)

        for node in ast.walk(self.step_tree):
            if (
                isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Name)
                and node.left.id == "ys"
                and isinstance(node.ops[0], ast.Gt)
                and isinstance(node.comparators[0], ast.Num)
                # and node.comparators[0].n == 1e-10
            ):
                node.comparators[0] = ast.Subscript(
                    value=ast.Name(id="group", ctx=ast.Load()),
                    slice=ast.Index(value=ast.Str(s="tolerance_ys")),
                    ctx=ast.Load(),
                )
                break
        self.step_tree = ast.fix_missing_locations(self.step_tree)
        self.step_method_code = compile(self.step_tree, "<string>", "exec")
        exec(self.step_method_code, globals(), locals())
        self.step = torch.no_grad()(partial(locals()["step"], self))

    def _numel(self):
        if self._numel_cache is None:
            # self._numel_cache = sum(
            #    2 * p.numel() if torch.is_complex(p) else p.numel()
            #    for p in self._params
            # )
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0
            )
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            # if torch.is_complex(view):
            #    view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            # if torch.is_complex(p):
            #    p = torch.view_as_real(p)
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()


if __name__ == "__main__":
    # define a simple model
    x_lbfgs = 10 * torch.ones(2, 1)
    x_lbfgs.requires_grad = True

    opt = LBFGS([x_lbfgs], lr=0.01)

    if 1:
        # retrieve the step code in string format
        import astor

        print(astor.to_source(opt.step_tree))

    # 2d Rosenbrock function
    def f(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def closure():
        opt.zero_grad()
        l = f(x_lbfgs)
        l.backward()
        return l

    # now we can use the optimizer as usual
    for i in range(10):
        opt.step(closure)
