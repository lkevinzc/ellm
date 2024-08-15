"""Ensemble deep model to capture epistemic uncertainty."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(
                cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t
            )
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class EnsembleFC(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        weight_decay: float = 0.0,
        bias: bool = True,
    ) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(
            torch.Tensor(ensemble_size, in_features, out_features)
        )
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b


class EnsembleModel(nn.Module):
    def __init__(self, encoding_dim, num_ensemble, hidden_dim=128, activation="relu"):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_dim
        self.nn1 = EnsembleFC(
            encoding_dim, hidden_dim, num_ensemble, weight_decay=0.000025
        )
        self.nn2 = EnsembleFC(
            hidden_dim, hidden_dim, num_ensemble, weight_decay=0.00005
        )

        self.output_dim = 1
        self.nn_out = EnsembleFC(
            hidden_dim, self.output_dim, num_ensemble, weight_decay=0.0001
        )

        self.apply(init_weights)

        if activation == "swish":
            self.activation = Swish()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation {activation}")

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.

        Returns:
            parameters tensor
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def forward(self, encoding):
        x = self.activation(self.nn1(encoding))
        x = self.activation(self.nn2(x))
        score = self.nn_out(x)
        return score

    def init(self):
        device = self.get_params().data.device
        self.init_params = self.get_params().data.clone().to(device)

    def regularization(self):
        return ((self.get_params() - self.init_params) ** 2).sum()
