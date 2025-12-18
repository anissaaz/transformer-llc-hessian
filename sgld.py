from typing import Literal, Union
from torch.optim import Optimizer
import numpy as np
import torch

class SGLD(torch.optim.Optimizer):
    """
    Implements Stochastic Gradient Langevin Dynamics (SGLD) optimizer.
    Reference: Hoogland et al.
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        noise_level=1.0,
        weight_decay=0.0,
        elasticity=0.0,
        temperature: Union[Literal["adaptive"], float] = "adaptive",
        num_samples=1,
    ):
        defaults = dict(
            lr=lr,
            noise_level=noise_level,
            weight_decay=weight_decay,
            elasticity=elasticity,
            temperature=temperature,
            num_samples=num_samples,
        )
        
        super(SGLD, self).__init__(params, defaults)
    
        # save initial parameters if elasticity term is set
        for group in self.param_groups:
            if group["elasticity"] != 0:
                for p in group["params"]:
                    param_state = self.state[p]
                    param_state["initial_param"] = p.data.clone().detach()      # This is w^* (center of basin)
                if group["temperature"] == "adaptive":
                    group["temperature"] = np.log(group["num_samples"])
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure(0)
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue                    # if no gradient, don't update
                param_state = self.state[p]
                
                #import ipdb; ipdb.set_trace();
                
                # Drift term 1: Gradient = N * 1/n sum(delta log p(x|theta))
                # p.grad.data = 1/n sum(delta_L)
                # group["num_samples"] = N -> for scaling
                dw = p.grad.data * group["num_samples"] / group["temperature"]
                
                # weight decay drift: delta log p(theta)
                if group["weight_decay"] != 0:
                    dw.add_(p.data, alpha=group["weight_decay"])
                
                # Drift term 2: elasticity (localization)
                if group["elasticity"] != 0:
                    initial_param = self.state[p]["initial_param"]
                    dw.add_((p.data - initial_param), alpha=group["elasticity"])
                
                # weight update: w = w - (epsilon/2) * drift
                p.data.add_(dw, alpha=-0.5 * group["lr"])
                
                # add Gaussian noise, noise ~ N(0, epsilon)
                noise = torch.normal(
                    mean=0.0, std=group["noise_level"], size=dw.size(), device=dw.device
                )
                p.data.add_(noise, alpha=group["lr"] ** 0.5)