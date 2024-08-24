import torch

from ellm.rm.model import default_weight_loader
from ellm.rm.networks import EnsembleModel

model = EnsembleModel(2048, 10)


for name, param in model.named_parameters():
    print(name, param.shape)

params_dict = dict(model.named_parameters())

default_weight_loader(params_dict[name], torch.zeros_like(param) + 100)

for name, param in model.named_parameters():
    print(name, param)


import numpy as np

mask = np.ones([10, 8])
for m, i in zip(mask, range(10)):
    m[:3] = 0

print(mask)
