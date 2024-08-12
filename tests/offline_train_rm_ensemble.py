import os

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim
from tqdm import tqdm

from ellm.rm.ensemble import EnsembleModel

data = pd.read_pickle("processed_features_for_rm.pt")
train_data = data[:-1000]
test_data = data[-1000:]

encoding_dim = 2048
num_ensemble = 3
learning_rate = 1e-3
gradient_accumulation = 8

model = EnsembleModel(encoding_dim, num_ensemble).cuda()

wandb.init(
    project="ellm_rm",
    name=os.path.basename(__file__),
    config={
        "lr": learning_rate,
        "num_ensemble": num_ensemble,
        "gradient_accumulation": gradient_accumulation,
    },
    reinit=True,
)


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self,
        chosen_reward: torch.Tensor,
        reject_reward: torch.Tensor,
        margin: torch.Tensor = None,
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


loss_fn = PairWiseLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)


def eval():
    model.eval()
    all_acc = []
    all_loss = []
    for test_b in test_data:
        test_b = test_b.float().cuda()
        batch_inp = test_b[None, :, :].repeat([num_ensemble, 1, 1])
        scores = model(batch_inp)
        mean_scores = scores.detach()
        chosen_scores, rejected_scores = torch.split(mean_scores, 4, dim=1)
        _loss = loss_fn(chosen_scores, rejected_scores)
        all_loss.append(_loss)
        # use expectation for decision
        accuracy = (chosen_scores.mean(0) > rejected_scores.mean(0)).float().mean()
        all_acc.append(accuracy)

    eval_acc = torch.stack(all_acc).mean().cpu()
    eval_loss = torch.stack(all_loss).mean().cpu()
    return {"test_acc": eval_acc, "test_loss": eval_loss}


loss = 0
for i, batch in enumerate(tqdm(train_data)):
    batch = batch.float().cuda()
    batch_inp = batch[None, :, :].repeat([num_ensemble, 1, 1])
    scores = model(batch_inp)
    chosen_scores, rejected_scores = torch.split(scores, 4, dim=1)
    _loss = loss_fn(chosen_scores, rejected_scores)
    loss += _loss
    if i % gradient_accumulation == gradient_accumulation - 1:
        optimizer.zero_grad()
        loss /= gradient_accumulation
        loss.backward()
        optimizer.step()
        wandb.log({"train_loss": loss.detach().cpu()}, step=i)
        loss = 0

    if i % 1000 == 0:
        wandb.log(eval(), step=i)
        model.train()


wandb.finish()
