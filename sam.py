import torch

SAM_iter_n = 0

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, "Invalid rho, should be non-negative: %s" % rho

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        '''需要修改hvd，使得每次BP都进行同步
        #/usr/local/lib/python3.5/dist-packages/horovod/torch/__init__.py
        #line 160, rm if
        不需要改这里, hvd同步梯度的hook是加在BP里的, 只需要在每次BP之后立即同步
        '''
        global SAM_iter_n
        SAM_iter_n += 1
        grad_norm = self._grad_norm()
        # print('grad_norm', grad_norm)
        for group in self.param_groups:
            if SAM_iter_n > 50: ## ---------------------------------
                scale = group["rho"] / (grad_norm + 1e-12)
            else:
                scale = 0

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError(
            "SAM doesn't work like the other optimizers, you should first call `first_step` "
            "and the `second_step`; see the documentation for more info.")

    def _grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.grad.float().norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None]),
                    p=2)
        return norm
