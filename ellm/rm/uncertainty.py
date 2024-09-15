import einops
import torch


def kl_ensemble(rewards: torch.Tensor) -> torch.Tensor:
    """Epistemic uncertainty measured by model disagreement between ensembles.

    Calculates KL divergence between individual distribution predictions and
    the Bernoulli mixture distribution.

    Args:
        rewards (torch.Tensor): Reward prediction (logits), (E, M, N, 1)

    Returns:
        torch.Tensor: Uncertainty, (M, N, N')
    """
    E = rewards.shape[0]
    reward_gap = rewards - einops.rearrange(rewards, "e m n 1 -> e m 1 n")
    prob = reward_gap.sigmoid()  # (E, M, N, N')

    prob_mean = prob.mean(dim=0, keepdim=True)
    prob_c = 1 - prob
    prob_c_mean = 1 - prob_mean

    component_1 = prob * torch.log(prob / prob_mean)
    component_0 = prob_c * torch.log(prob_c / prob_c_mean)

    repeat_prob_mean = prob_mean.repeat(E, 1, 1, 1)
    kl = torch.where(
        repeat_prob_mean == 0,
        component_0,
        torch.where(repeat_prob_mean == 1, component_1, component_0 + component_1),
    ).sum(dim=0)

    # Avoid numerical errors.
    nan_idx = torch.isnan(kl)
    kl_T = kl.transpose(-1, -2)
    kl[nan_idx] = kl_T[nan_idx]
    return (kl + kl_T) / 2
