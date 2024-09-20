import glob
import os
from argparse import Namespace
from typing import Any, Literal, Tuple, Type

import numpy as np
import torch
import tyro
from matplotlib import pyplot as plt
from ml_collections import ConfigDict
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from torch import LongTensor, Tensor

from ellm.rm import model as reward_model
from ellm.types import RewardData
from ellm.utils.buffer import UniformBuffer


def reward_function(y):
    mean, std_dev = -1, 0.5
    component_1 = (1 / (np.sqrt(2 * np.pi * std_dev**2))) * np.exp(
        -0.5 * ((y - mean) / std_dev) ** 2
    )
    mask = y > mean
    component_1_right = component_1 * mask

    std_dev = 0.6
    component_1 = (1 / (np.sqrt(2 * np.pi * std_dev**2))) * np.exp(
        -0.5 * ((y - mean) / std_dev) ** 2
    ) + 0.135
    component_1_left = component_1 * (1 - mask)

    mean, std_dev = 1, 0.45
    component_2 = (1 / (np.sqrt(2 * np.pi * std_dev**2))) * np.exp(
        -0.5 * ((y - mean) / std_dev) ** 2
    )
    return 7 * (component_1_left + component_1_right + component_2)


class BTRewardModel(reward_model.EnnDoubleTS):
    def __init__(
        self, model_cls: Type[reward_model.EnnDoubleTS], args: Namespace, num_bin: int
    ) -> None:
        super().__init__(args)
        self.model = model_cls(args)
        # The way to take \arg\max.
        self.optimization = args.optimization
        self.support = np.linspace(-3, 3, num_bin, dtype=np.float32)

    def get_duel_actions(self, features: Tensor) -> Tuple[LongTensor]:
        del features  # No context.
        if self.optimization == "oracle":
            features = torch.from_numpy(self.support).view(1, -1, 1)
        else:
            raise ValueError
            features = ...  # sample from policy

        _, first_actions, second_actions = self.model.get_duel_actions(features)
        first_actions, second_actions = (
            first_actions.squeeze().item(),
            second_actions.squeeze().item(),
        )
        clash = first_actions == second_actions

        if self.optimization == "oracle":
            return (
                np.array(
                    (
                        self.support[first_actions],
                        self.support[second_actions],
                    )
                ).reshape(1, 2),
                clash,
            )
        else:
            raise ValueError

    @staticmethod
    def get_model_r(model, x):
        scores = model.model(x[None].repeat(model.model.num_ensemble, 1, 1))
        scores = scores.view(model.model.num_ensemble, -1)
        mean_scores = scores.mean(0)
        std_scores = scores.std(0)
        return mean_scores, std_scores

    def get_model_pred(self, x):
        x = x.view(-1, 1)
        mean_scores, _ = self.get_model_r(self.model, x)
        mean_scores = mean_scores.view(-1, 2)
        return mean_scores[:, 0] > mean_scores[:, 1]

    def learn(self, buffer: UniformBuffer) -> torch.Dict[str, Any]:
        return self.model.learn(buffer)

    def explore(self):
        if self.optimization == "oracle":
            return np.random.choice(
                self.support, (self.model.train_bs, 2), replace=True
            )
        else:
            raise ValueError  # sample from policy


class Evaluator:
    def __init__(self, save_dir, y_plot, num_bin) -> None:
        self.y_test = np.random.uniform(-3, 3, (1000, 2, 1))
        self.y_plot = y_plot
        self.num_bin = num_bin
        self.save_dir = save_dir

    def evaluate_rm_accuracy(self, model: BTRewardModel):
        y_test = self.y_test
        y_input = torch.from_numpy(y_test).float()
        z_pred = model.get_model_pred(y_input)
        z_gt = reward_function(y_test[:, 0]) > reward_function(y_test[:, 1])
        return (z_pred.view(-1).numpy() == z_gt.reshape(-1)).mean()

    def visualize_reward_model(
        self, step, model: BTRewardModel, history_actions, format, final=False
    ):
        r_pred_mean, r_pred_std = model.get_model_r(
            model.model, torch.from_numpy(self.y_plot).view(-1, 1).float()
        )
        r_pred_mean = r_pred_mean.squeeze().detach().numpy()
        r_pred_std = r_pred_std.squeeze().detach().numpy()

        fig = plt.figure(figsize=(6, 5))
        plt.plot(self.y_plot, r_pred_mean, label="pred_r", color="green")
        plt.fill_between(
            self.y_plot,
            r_pred_mean,
            r_pred_mean + 2 * r_pred_std,
            color="green",
            alpha=0.2,
        )
        plt.fill_between(
            self.y_plot,
            r_pred_mean - 2 * r_pred_std,
            r_pred_mean,
            color="green",
            alpha=0.2,
        )
        plt.xlabel("y")
        plt.xlim(-3, 3)
        plt.ylabel("reward model")
        # plt.legend(ncol=1, bbox_to_anchor=(1.25, 1), borderaxespad=0)

        ax = fig.axes[0]
        divider = make_axes_locatable(ax)
        ax_hist = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
        ax_hist.xaxis.set_tick_params(labelbottom=False)
        ax_hist.hist(history_actions, bins=self.num_bin)
        ax_hist.set_ylabel("visitation")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"rm_trajectory_{step:05}.{format}"),
            bbox_inches="tight",
        )
        plt.close()

        if final:
            images = []
            fns = glob.glob(os.path.join(self.save_dir, "rm_trajectory*png"))
            size = None
            for fn in sorted(fns):
                img = Image.open(fn)
                if size is None:
                    size = img.size
                else:
                    img = img.resize(size)
                print(fn, img.size)
                images.append(img)

            images[0].save(
                os.path.join(self.save_dir, f"rm_trajectory.gif"),
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0,
            )
            [os.remove(fn) for fn in fns]


def insert_to_buffer(buffer: UniformBuffer, dueling_actions, feedback):
    X = torch.from_numpy(dueling_actions).float()
    Y = torch.from_numpy(feedback)[:, None].float()
    chosen_features = torch.where(Y == 1, X[:, :1], X[:, 1:])
    rejected_features = torch.where(Y == 0, X[:, :1], X[:, 1:])
    pair_features = torch.cat([chosen_features, rejected_features], dim=1).float()
    batch = RewardData(
        pair_features=pair_features,
        loss_masks=torch.ones(len(pair_features)),
    )
    buffer.extend(batch)


def main(
    total_budget: int = 1000,
    n_sample_plot: int = 200,
    num_ensemble: int = 10,
    num_bin: int = 500,
    strategy: Literal[
        "EnnDoubleTS", "EnnInfoMax", "EnnTSInfoMax", "EnnPE", "EnnDuelingTS"
    ] = "EnnTSInfoMax",
    clash_strategy: Literal["random", "top2"] = "random",
    optimization: Literal["oracle", "policy"] = "oracle",
    save_dir: str = "output/toy",
    tag: str = "",
    format: str = "png",
    enn_lambda: float = 0.1,
    pi_ref_mu: float = -0.5,
    pi_ref_std: float = 0.35,
):
    save_dir = os.path.join(
        save_dir, strategy, clash_strategy if strategy == "EnnDoubleTS" else "", tag
    )
    os.makedirs(save_dir, exist_ok=True)

    # Plot the oracle reward function.
    y_plot = np.linspace(-3, 3, n_sample_plot)
    r_plot = reward_function(y_plot)
    plt.figure()
    plt.plot(y_plot, r_plot, label="reward")
    plt.xlabel("y")
    plt.ylabel("reward")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"reward.{format}"))
    plt.close()

    # Plot the preference landscape.
    preference_landscape = np.zeros((n_sample_plot, n_sample_plot))
    for i in range(n_sample_plot):
        for j in range(n_sample_plot):
            r1 = reward_function(y_plot[i])
            r2 = reward_function(y_plot[j])
            z = torch.tensor(r1 - r2).sigmoid().numpy()
            preference_landscape[i, j] = z

    plt.figure()
    plt.imshow(preference_landscape)
    plt.ylabel("y")
    plt.xlabel("y'")
    plt.savefig(os.path.join(save_dir, f"preference.{format}"))
    plt.close()

    # Construct the reward model (and optionally the policy).
    model_cfg = ConfigDict(
        {
            "enn_max_try": num_ensemble,
            "num_ensemble": num_ensemble,
            "encoding_dim": 1,
            "rm_hidden_dim": 128,
            "rm_act_fn": "relu",
            "rm_lr": 1e-3,
            "rm_wd": 5e-5,
            "enn_lambda": enn_lambda,
            "exp_allow_second_best": False,
            "rm_sgd_steps": 1,
            "optimization": optimization,
        }
    )
    model = BTRewardModel(getattr(reward_model, strategy), model_cfg, num_bin)
    model.model.train_bs = 32
    model.train_bs = 32

    # Construct buffer.
    buffer = UniformBuffer(total_budget)

    # Online dueling bandit loop.
    evaluator = Evaluator(save_dir, y_plot, num_bin)
    num_interaction = 0
    init_clash = []
    evaluator.visualize_reward_model(
        step=0, model=model, history_actions=[], format=format
    )
    while num_interaction < total_budget:
        if num_interaction == 0:
            # Random explore to get the first batch.
            dueling_actions = model.explore()
            feedback = reward_function(dueling_actions[:, 0]) > reward_function(
                dueling_actions[:, 1]
            )
            insert_to_buffer(buffer, dueling_actions, feedback)
            num_interaction += len(dueling_actions)
            continue

        # Query the environment.
        dueling_actions, clash = model.get_duel_actions(None)
        if clash:
            if clash_strategy == "random":
                dueling_actions[0, 1] = np.random.choice(model.support)
            else:
                raise ValueError

        # BT preference model.
        r1 = reward_function(dueling_actions[:, 0])
        r2 = reward_function(dueling_actions[:, 1])
        prob = torch.tensor(r1 - r2).sigmoid()
        feedback = torch.bernoulli(prob).numpy()

        insert_to_buffer(buffer, dueling_actions, feedback)
        init_clash.append(clash)

        # Update the reward model.
        buffer.total_num_queries = num_interaction
        info = model.learn(buffer)

        if num_interaction % 20 == 0 or num_interaction == total_budget - 1:
            info.update(
                {
                    "eval/rm_acc": evaluator.evaluate_rm_accuracy(model),
                    "actor/init_clash_ratio": np.mean(init_clash),
                }
            )
            print(num_interaction, info)
            evaluator.visualize_reward_model(
                num_interaction,
                model,
                buffer.get_all().pair_features.reshape(-1),
                format,
                final=num_interaction == total_budget - 1,
            )
            init_clash.clear()

        num_interaction += 1


if __name__ == "__main__":
    tyro.cli(main)
