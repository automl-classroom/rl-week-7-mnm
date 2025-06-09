import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
n_seeds = 4

df_rnd_s0 = pd.read_csv(os.path.join(base_dir, "training_data_seed_0_RNDDQN.csv"))
df_rnd_s1 = pd.read_csv(os.path.join(base_dir, "training_data_seed_1_RNDDQN.csv"))
df_rnd_s2 = pd.read_csv(os.path.join(base_dir, "training_data_seed_2_RNDDQN.csv"))
df_rnd_s3 = pd.read_csv(os.path.join(base_dir, "training_data_seed_3_RNDDQN.csv"))

df_rnd_s0["seed"] = 0
df_rnd_s1["seed"] = 1
df_rnd_s2["seed"] = 2
df_rnd_s3["seed"] = 3

# Align lengths
smallest_rnd = min(df_rnd_s0.shape[0], df_rnd_s1.shape[0], df_rnd_s2.shape[0], df_rnd_s3.shape[0])
df_rnd = pd.concat(
    [df_rnd_s0[:smallest_rnd], df_rnd_s1[:smallest_rnd], df_rnd_s2[:smallest_rnd], df_rnd_s3[:smallest_rnd]],
    ignore_index=True,
)

df_dqn_s0 = pd.read_csv(os.path.join(base_dir, "training_data_seed_0_DQN.csv"))
df_dqn_s1 = pd.read_csv(os.path.join(base_dir, "training_data_seed_1_DQN.csv"))
df_dqn_s2 = pd.read_csv(os.path.join(base_dir, "training_data_seed_2_DQN.csv"))
df_dqn_s3 = pd.read_csv(os.path.join(base_dir, "training_data_seed_3_DQN.csv"))

df_dqn_s0["seed"] = 0
df_dqn_s1["seed"] = 1
df_dqn_s2["seed"] = 2
df_dqn_s3["seed"] = 3

# Align lengths
smallest_dqn = min(df_dqn_s0.shape[0], df_dqn_s1.shape[0], df_dqn_s2.shape[0], df_dqn_s3.shape[0])
df_dqn = pd.concat(
    [df_dqn_s0[:smallest_dqn], df_dqn_s1[:smallest_dqn], df_dqn_s2[:smallest_dqn], df_dqn_s3[:smallest_dqn]],
    ignore_index=True,
)

common_smallest = min(smallest_rnd, smallest_dqn)
df_rnd = df_rnd.iloc[:common_smallest * n_seeds]
df_dqn = df_dqn.iloc[:common_smallest * n_seeds]

steps = df_rnd["steps"].to_numpy().reshape((n_seeds, -1))[0]

rewards_rnd = df_rnd["reward"].to_numpy().reshape((n_seeds, -1))
rewards_dqn = df_dqn["rewards"].to_numpy().reshape((n_seeds, -1))

# Compute min and max scores for normalization
all_rewards = np.concatenate([rewards_rnd.flatten(), rewards_dqn.flatten()])
min_score = all_rewards.min()
max_score = all_rewards.max()

# Normalize rewards
rewards_rnd = (rewards_rnd - min_score) / (max_score - min_score)
rewards_dqn = (rewards_dqn - min_score) / (max_score - min_score)

train_scores = {
    "rnddqn": rewards_rnd,
    "dqn": rewards_dqn,
}

iqm = lambda scores: np.array(
    [metrics.aggregate_iqm(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
)

iqm_scores, iqm_cis = get_interval_estimates(
    train_scores,
    iqm,
    reps=2000,
)

# This is a utility function, but you can also just use a normal line plot with the IQM and CI scores
plot_sample_efficiency_curve(
    steps + 1,
    iqm_scores,
    iqm_cis,
    algorithms=["rnddqn", "dqn"],
    xlabel=r"Number of Evaluations",
    ylabel="IQM Normalized Score",
)
plt.gcf().canvas.manager.set_window_title(
    "IQM Normalized Score - Sample Efficiency Curve"
)
plt.legend()
plt.tight_layout()
plt.show()


# Level 2

import pandas as pd
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Intrinsic Reward Plot
axes[0].plot(df_rnd['steps'], df_rnd['intrinsic_reward'], label='Intrinsic Reward', color='orange')
axes[0].set_title('Intrinsic Reward Over Time')
axes[0].set_xlabel('Steps')
axes[0].set_ylabel('Intrinsic Reward')
axes[0].grid(True)
axes[0].legend()

# Extrinsic Reward Plot
axes[1].plot(df_rnd['steps'], df_rnd['extrinsic_reward'], label='Extrinsic Reward', color='blue')
axes[1].set_title('Extrinsic Reward Over Time')
axes[1].set_xlabel('Steps')
axes[0].set_ylabel('Extrinsic Reward')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()