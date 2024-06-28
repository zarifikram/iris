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

sequence_embedding_losses = [log['sequence_embedder/train/total_loss'] for log in all_logs if 'sequence_embedder/train/total_loss' in log]
frame_embedding_losses = [log['frame_embedder(Assran et al. 2023)/train/total_loss'] for log in all_logs if 'frame_embedder(Assran et al. 2023)/train/total_loss' in log]


for log_list in all_logs:
    for log in log_list:
        if 'sequence_embedder/train/total_loss' in log:
            sequence_embedding_losses.append(log['sequence_embedder/train/total_loss'])
        if 'frame_embedder(Assran et al. 2023)/train/total_loss' in log:
            frame_embedding_losses.append(log['frame_embedder(Assran et al. 2023)/train/total_loss'])

print(f"Sequence Embedding Losses: {len(sequence_embedding_losses)}")
print(f"Frame Embedding Losses: {len(frame_embedding_losses)}")

fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True)
axes[0].plot(np.arange(50, 200), sequence_embedding_losses)
axes[0].set_title('Sequence Embedding Losses')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[1].plot(np.arange(25, 200), frame_embedding_losses)
axes[1].set_title('Frame Embedding Losses')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
# add log scale
axes[0].set_yscale('log')
axes[1].set_yscale('log')
plt.show()
