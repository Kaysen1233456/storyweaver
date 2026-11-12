import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        Initialize the AdamW optimizer.
        
        Args:
            params (iterable): iterable of parameters to optimize
            lr (float): learning rate (default: 0.001)
            betas (tuple): coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
            eps (float): term added to the denominator for numerical stability (default: 1e-8)
            weight_decay (float): weight decay (L2 penalty) (default: 0.01)
        """

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(AdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        
        Returns:
            loss (float, optional): If closure is provided, returns the loss value.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                state = self.state[p]

                # Initialize state for the first time
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)  # First moment
                    state['v'] = torch.zeros_like(p)  # Second moment

                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']

                # Decoupled weight decay acts on the parameters before the Adam update
                p.mul_(1 - group['lr'] * group['weight_decay'])

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                denom = v.sqrt().div_(bias_correction2 ** 0.5).add_(group['eps'])
                p.addcdiv_(m, denom, value=-group['lr'] / bias_correction1)
        
        return loss
