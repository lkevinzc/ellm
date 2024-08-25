import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.optimizer import ParamsT


class LAdam(Adam):
    """
    Adam-style Stochastic Gradient Langevin Dynamics.

    Reference: https://arxiv.org/pdf/2009.09535.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        temperature: Union[float, Tensor] = 1e-2,
        a: Union[float, Tensor] = 1,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        fused: Optional[bool] = None,
    ):
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=False,
            differentiable=False,
            fused=fused,
        )
        self.temperature = temperature
        self.a = a

    @torch.no_grad
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            _ = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            _asgld(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                temperature=self.temperature,
                a=self.a,
            )

        return loss


def _asgld(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    temperature: float,
    a: float,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt()).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt()).add_(eps)

        # Add pure gradient
        param.add_(grad, alpha=-lr)
        # Add the adaptive bias term
        am = a * exp_avg
        param.addcdiv_(am, denom, value=-lr)
        # Add noise
        grad_perturb = torch.normal(
            0, 1, size=param.shape, dtype=param.dtype, device=param.device
        )
        param.add_(math.sqrt(2.0 * temperature * lr) * grad_perturb)
