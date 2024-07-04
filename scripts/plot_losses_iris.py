import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument("dir", type=str, help="Directory containing the pickle files")
args = argparser.parse_args()

dir = args.dir

with open(dir + "/all_logs.pkl", "rb") as f:
    all_logs = pickle.load(f)

commitment_losses, reconstruction_losses, perceptual_losses = [], [], []
loss_obs, loss_rewards, loss_ends = [], [], []
loss_actions, loss_values, loss_entropy = [], [], []

all_losses = [
    "tokenizer/train/commitment_loss",
    "tokenizer/train/reconstruction_loss",
    "tokenizer/train/perceptual_loss",
    "world_model/train/loss_obs",
    "world_model/train/loss_rewards",
    "world_model/train/loss_ends",
    "actor_critic/train/loss_actions",
    "actor_critic/train/loss_values",
    "actor_critic/train/loss_entropy",
]

losses_dict = {loss: [] for loss in all_losses}


print(all_logs[-1])
for log_list in all_logs:
    for log in log_list:
        for loss in all_losses:
            if loss in log:
                losses_dict[loss].append(log[loss])
                if loss == 'actor_critic/train/loss_entropy':
                    losses_dict[loss][-1] = -1 * losses_dict[loss][-1]

total_epochs = 170
fig, axes = plt.subplots(2, 5, figsize=(10, 5), sharex=True)
for i, loss_name in enumerate(all_losses):
    if i == len(all_losses)-1:
        i += 1
    axes.flatten()[i].plot(np.arange(total_epochs - len(losses_dict[loss_name]), total_epochs), losses_dict[loss_name])
    axes.flatten()[i].set_title(loss_name, fontsize=8)
    axes.flatten()[i].set_ylabel("Loss", fontsize=8)
    axes.flatten()[i].set_xlabel("Epoch", fontsize=8)
    # x scale fontsize 8
    # y scale fontsize 8
    axes.flatten()[i].tick_params(axis='x', labelsize=8)
    axes.flatten()[i].tick_params(axis='y', labelsize=8)

# add log scale
for ax in axes.flatten():
    ax.set_yscale("log")


plt.show()
