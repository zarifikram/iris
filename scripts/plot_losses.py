import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np
argparser = argparse.ArgumentParser()
argparser.add_argument('dir', type=str, help='Directory containing the pickle files')
args = argparser.parse_args()

dir = args.dir

with open(dir + '/all_logs.pkl', 'rb') as f:
    all_logs = pickle.load(f)

sequence_embedding_prediction_losses = []
sequence_embedding_reward_losses = []
sequence_embedding_ends_losses = []
frame_embedding_losses = []
actor_critic_value_losses = []
actor_critic_policy_losses = []
actor_critic_entropy_losses = []


print(all_logs[-1])
for log_list in all_logs:
    for log in log_list:
        if 'frame_embedder(Assran et al. 2023)/train/prediction_loss' in log:
            frame_embedding_losses.append(log['frame_embedder(Assran et al. 2023)/train/prediction_loss']) 
        if 'sequence_embedder/train/prediction_loss' in log:
            sequence_embedding_prediction_losses.append(log['sequence_embedder/train/prediction_loss'])
        if 'sequence_embedder/train/reward_loss' in log:
            sequence_embedding_reward_losses.append(log['sequence_embedder/train/reward_loss'])
        if 'sequence_embedder/train/ends_loss' in log:
            sequence_embedding_ends_losses.append(log['sequence_embedder/train/ends_loss'])

        if 'actor_critic/train/loss_actions' in log:
            actor_critic_policy_losses.append(log['actor_critic/train/loss_actions'])
        if 'actor_critic/train/loss_values' in log:
            actor_critic_value_losses.append(log['actor_critic/train/loss_values'])
        if 'actor_critic/train/loss_entropy' in log:
            actor_critic_entropy_losses.append(-1*log['actor_critic/train/loss_entropy'])



fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharex=True)
axes[0][0].plot(np.arange(25, 250), frame_embedding_losses)
axes[0][0].set_title('Frame Embedding Losses', fontsize=8)
axes[0][0].set_ylabel('Loss')
axes[0][0].set_xlabel('Epoch')
axes[0][1].plot(np.arange(50, 250), sequence_embedding_prediction_losses)
axes[0][1].set_title('Sequence Embedding Prediction Losses', fontsize=8)
axes[0][1].set_xlabel('Epoch')
axes[0][1].set_ylabel('Loss')
axes[0][2].plot(np.arange(50, 250), sequence_embedding_reward_losses)
axes[0][2].set_title('Sequence Embedding Reward Losses', fontsize=8)
axes[0][2].set_xlabel('Epoch')
axes[0][2].set_ylabel('Loss')
axes[0][3].plot(np.arange(50, 250), sequence_embedding_ends_losses)
axes[0][3].set_title('Sequence Embedding Ends Losses', fontsize=8)
axes[0][3].set_xlabel('Epoch')
axes[0][3].set_ylabel('Loss')
axes[1][1].plot(np.arange(75, 250), actor_critic_value_losses)
axes[1][1].set_title('Actor Critic Value Losses', fontsize=8)
axes[1][1].set_xlabel('Epoch')
axes[1][1].set_ylabel('Loss')
axes[1][2].plot(np.arange(75, 250), actor_critic_policy_losses)
axes[1][2].set_title('Actor Critic Policy Losses', fontsize=8)
axes[1][2].set_xlabel('Epoch')
axes[1][2].set_ylabel('Loss')
axes[1][3].plot(np.arange(75, 250), actor_critic_entropy_losses)
axes[1][3].set_title('Actor Critic Entropy Losses', fontsize=8)
axes[1][3].set_xlabel('Epoch')
axes[1][3].set_ylabel('Loss')
# add log scale
for ax in axes.flatten():
    ax.set_yscale('log')



plt.show()