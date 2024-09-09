import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from fire import Fire
from torch import nn, optim
from tqdm import tqdm

from ellm.rm.networks import EnsembleModel


def main(
    feature_path: str = "processed_features_for_rm.pt",
    encoding_dim=2048,
    hidden_dim=128,
    num_ensemble=3,
    learning_rate=1e-3,
    weight_decay=0,
    weight_reg=0.1,
    gradient_accumulation=8,
    activation="relu",
    track=False,
):
    data = pd.read_pickle(feature_path)
    train_data = data[:-1000]
    test_data = data[-1000:]

    model = EnsembleModel(
        encoding_dim, num_ensemble, hidden_dim=hidden_dim, activation=activation
    ).cuda()
    model.init()

    wandb.init(
        project="ellm_rm",
        name=f"ensemble={num_ensemble}_lr={learning_rate}_hd={hidden_dim}_wr={weight_reg}_wd={weight_decay}_grad-acc={gradient_accumulation}",
        config={
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "num_ensemble": num_ensemble,
            "gradient_accumulation": gradient_accumulation,
            "hidden_dim": hidden_dim,
            "activation": activation,
            "weight_reg": weight_reg,
        },
        reinit=True,
        mode="online" if track else "disabled",
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
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    def eval():
        model.eval()
        all_acc = []
        all_loss = []
        all_margin_means = []
        all_margin_stds = []
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

            all_margin_means.append((chosen_scores - rejected_scores).mean(0))
            all_margin_stds.append((chosen_scores - rejected_scores).std(0))

        eval_acc = torch.stack(all_acc).mean().cpu()
        eval_loss = torch.stack(all_loss).mean().cpu()
        all_margin_means = torch.stack(all_margin_means).mean().cpu()
        all_margin_stds = torch.stack(all_margin_stds).mean().cpu()
        return {
            "test_acc": eval_acc,
            "test_loss": eval_loss,
            "test_reward_margin_mean": all_margin_means,
            "test_reward_margin_std": all_margin_stds,
        }

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
            if weight_reg > 0:
                _reg_loss = model.regularization() * weight_reg
            else:
                _reg_loss = torch.tensor(0)
            loss += _reg_loss
            loss.backward()
            optimizer.step()
            wandb.log(
                {
                    "train_loss": loss.detach().cpu(),
                    "reg_loss": _reg_loss.detach().cpu(),
                },
                step=i,
            )
            loss = 0

        if i % 1000 == 0:
            wandb.log(eval(), step=i)
            model.train()

    wandb.finish()
    torch.save(model.cpu().state_dict(), f"./offline_reward_enn_{num_ensemble}.pt")


if __name__ == "__main__":
    Fire(main)
