import matplotlib.pyplot as plt
import numpy as np


def reward_function(y, offset=0.5):
    mean, std_dev = -1, 0.5
    component_1 = (1 / (np.sqrt(2 * np.pi * std_dev**2))) * np.exp(
        -0.5 * ((y - mean) / std_dev) ** 2
    )
    mean, std_dev = 1, 0.35
    component_2 = (1 / (np.sqrt(2 * np.pi * std_dev**2))) * np.exp(
        -0.5 * ((y - mean) / std_dev) ** 2
    )
    return component_1 + component_2 + offset


def policy(y):
    mean, std_dev = pi_ref_mu, pi_ref_std
    return (1 / (np.sqrt(2 * np.pi * std_dev**2))) * np.exp(
        -0.5 * ((y - mean) / std_dev) ** 2
    )


n_sample_plot = 500
pi_ref_mu = -0.5
pi_ref_std = 0.7
y_plot = np.linspace(-3, 3, n_sample_plot)
r_plot = reward_function(y_plot)
pi_plot = policy(y_plot)

plt.plot(y_plot, r_plot, label="reward")
plt.plot(y_plot, pi_plot, label="pi_ref")
plt.xlabel("y")
plt.ylabel("reward")
plt.legend()

preference_landscape = np.zeros((n_sample_plot, n_sample_plot))
for i in range(n_sample_plot):
    for j in range(n_sample_plot):
        z = reward_function(y_plot[i]) > reward_function(y_plot[j])
        preference_landscape[i, j] = z

plt.imshow(preference_landscape)
plt.ylabel("y")
plt.xlabel("y'")

num_samples = 5000
# y_pairs = np.random.normal(pi_ref_mu, pi_ref_std, (num_samples, 2))
y_pairs = np.random.uniform(-3, 3, (5000, 2))
z_pairs = reward_function(y_pairs[:, 0]) > reward_function(y_pairs[:, 1])
map_coord = lambda x: (x + 3) / 6 * n_sample_plot
plt.imshow(preference_landscape)
plt.ylabel("y")
plt.xlabel("y'")
for i, (y, yp) in enumerate(y_pairs):
    if z_pairs[i]:
        label = "+"
        color = "black"
    else:
        label = "x"
        color = "red"
    plt.plot(map_coord(yp), map_coord(y), label, color=color)


import torch
from ml_collections import ConfigDict

from ellm.rm import model as backbone

model_cls = getattr(backbone, "EnnTSInfoMax")
model_cfg = ConfigDict(
    {
        "enn_max_try": 1,
        "num_ensemble": 1,
        "encoding_dim": 1,
        "rm_hidden_dim": 128,
        "rm_act_fn": "relu",
        "rm_lr": 1e-3,
        "enn_lambda": 0,
        "exp_allow_second_best": False,
        "rm_sgd_steps": 1,
    }
)
model = model_cls(model_cfg)
from ellm.types import RewardData
from ellm.utils.buffer import UniformBuffer

X = torch.from_numpy(y_pairs).float()
Y = torch.from_numpy(z_pairs)[:, None].float()


chosen_features = torch.where(Y == 1, X[:, :1], X[:, 1:])
rejected_features = torch.where(Y == 0, X[:, :1], X[:, 1:])
pair_features = torch.cat([chosen_features, rejected_features], dim=1).float()

batch = RewardData(
    pair_features=pair_features,
    loss_masks=torch.ones(len(pair_features)),
)
buffer = UniformBuffer(10000)
buffer.extend(batch)
buffer.total_num_queries = 16
model.train_bs = 5000

num_epochs = 500
for epoch in range(num_epochs):
    info = model.learn(buffer)
    if (epoch + 1) % 10 == 0:
        print(info)


# Test
def get_model_r(model, x):
    scores = model.model(x[None].repeat(model.model.num_ensemble, 1, 1))
    scores = scores.view(model.model.num_ensemble, -1)
    mean_scores = scores.mean(0)
    std_scores = scores.std(0)
    return mean_scores, std_scores


def get_model_pred(model, x):
    x = x.view(-1, 1)
    mean_scores, _ = get_model_r(model, x)
    mean_scores = mean_scores.view(-1, 2)
    return mean_scores[:, 0] > mean_scores[:, 1]


num_samples = 200
y_test = np.random.uniform(-3, 3, (num_samples, 2, 1))
y_input = torch.from_numpy(y_test).float()
z_pred = get_model_pred(model, y_input)
z_gt = reward_function(y_test[:, 0]) > reward_function(y_test[:, 1])
map_coord = lambda x: (x + 3) / 6 * n_sample_plot
plt.imshow(preference_landscape)
plt.ylabel("y")
plt.xlabel("y'")
for i, (y, yp) in enumerate(y_test):
    if z_pred[i]:
        label = "+"
        color = "black"
    else:
        label = "x"
        color = "red"
    plt.plot(map_coord(yp), map_coord(y), label, color=color)

print("accuracy:", ((z_pred.view(-1).numpy() == z_gt.reshape(-1))).mean())

y_plot = np.linspace(-3, 3, n_sample_plot)

r_pred_mean, r_pred_std = get_model_r(
    model, torch.from_numpy(y_plot).view(-1, 1).float()
)
r_pred_mean = r_pred_mean.squeeze().detach().numpy()

r_plot = reward_function(y_plot)
pi_plot = policy(y_plot)

plt.plot(y_plot, r_plot, label="reward")
plt.plot(y_plot, pi_plot, label="pi_ref")
plt.plot(y_plot, r_pred_mean, label="pred_r")
plt.xlabel("y")
plt.ylabel("reward")
plt.legend()

# import torch
# import torch.nn as nn
# import torch.optim as optim


# class BTRewardModel(nn.Module):
#     def __init__(self):
#         super(BTRewardModel, self).__init__()
#         # Define the layers
#         self.layers = nn.Sequential(
#             nn.Linear(1, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#         )

#     def forward(self, x):
#         # Define the forward pass
#         y, yp = x[:, :1], x[:, 1:]
#         r = self.get_r(y)
#         rp = self.get_r(yp)

#         return torch.sigmoid(r - rp)

#     def get_r(self, x):
#         return self.layers(x)


# # Initialize the model, loss function, and optimizer
# model = BTRewardModel()
# criterion = nn.BCELoss()  # Binary Cross Entropy Loss
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# # Dummy dataset for illustration (input features and labels)
# X = torch.from_numpy(y_pairs).float()
# Y = torch.from_numpy(z_pairs)[:, None].float()


# chosen_features = torch.where(Y == 1, X[:, :1], X[:, 1:])
# rejected_features = torch.where(Y == 0, X[:, :1], X[:, 1:])
# pair_features = torch.cat([chosen_features, rejected_features], dim=1).float()

# # Training loop
# num_epochs = 500
# for epoch in range(num_epochs):
#     # Forward pass
#     outputs = model(pair_features)
#     # loss = - torch.log(torch.where(Y==1, outputs, 1-outputs)).mean()
#     loss = -torch.log(outputs).mean()
#     # loss = criterion(outputs, y)

#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if (epoch + 1) % 10 == 0:  # Print loss every 10 epochs
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# print("Training complete!")

# # Test
# num_samples = 200
# y_test = np.random.uniform(-3, 3, (num_samples, 2))
# y_input = torch.from_numpy(y_test).float()
# z_pred = model(y_input) > 0.5
# z_gt = reward_function(y_test[:, 0]) > reward_function(y_test[:, 1])
# map_coord = lambda x: (x + 3) / 6 * n_sample_plot
# plt.imshow(preference_landscape)
# plt.ylabel("y")
# plt.xlabel("y'")
# for i, (y, yp) in enumerate(y_test):
#     if z_pred[i]:
#         label = "+"
#         color = "black"
#     else:
#         label = "x"
#         color = "red"
#     plt.plot(map_coord(yp), map_coord(y), label, color=color)

# print("accuracy:", ((z_pred.view(-1).numpy() == z_gt)).mean())

# y_plot = np.linspace(-3, 3, n_sample_plot)

# r_pred = (
#     model.get_r(torch.from_numpy(y_plot).view(-1, 1).float()).squeeze().detach().numpy()
# )

# r_plot = reward_function(y_plot)
# pi_plot = policy(y_plot)

# plt.plot(y_plot, r_plot, label="reward")
# plt.plot(y_plot, pi_plot, label="pi_ref")
# plt.plot(y_plot, r_pred, label="pred_r")
# plt.xlabel("y")
# plt.ylabel("reward")
# plt.legend()
