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

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        
        Returns:
            loss (float, optional): If closure is provided, returns the loss value.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state for the first time
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)  # First moment
                    state['v'] = torch.zeros_like(p.data)  # Second moment
                
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1
                t = state['step']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                alpha_t = group['lr'] * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)

                denom = v.sqrt().add_(group['eps'])
                p.data.addcdiv_(m, denom, value=-alpha_t)
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
        
        return loss
