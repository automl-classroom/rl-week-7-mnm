"""
Deep Q-Learning with RND implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import torch
#from week_4.dqn import DQNAgent, set_seed
from rl_exercises.week_4.dqn import DQNAgent, set_seed
import torch.nn as nn
import torch.optim as optim
import os

class NeuralNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, n_layers: int):
        super(NeuralNetwork, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            # n - 1 Hidden layers
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.network(x)

class RNDDQNAgent(DQNAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        rnd_hidden_size: int = 128,
        rnd_lr: float = 1e-3,
        rnd_update_freq: int = 1000,
        rnd_n_layers: int = 2,
        rnd_reward_weight: float = 0.1,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.seed = seed
        # TODO: initialize the RND networks
        self.rnd_update_freq = rnd_update_freq
        self.rnd_reward_weight = rnd_reward_weight

        self.random_network = NeuralNetwork(
            input_size=self.env.observation_space.shape[0],
            hidden_size=rnd_hidden_size,
            output_size=rnd_hidden_size,
            n_layers=rnd_n_layers,
        )
        for param in self.random_network.parameters():
            param.requires_grad = False
        self.predictor_network = NeuralNetwork(
            input_size=self.env.observation_space.shape[0],
            hidden_size=rnd_hidden_size,
            output_size=rnd_hidden_size,
            n_layers=rnd_n_layers,
        )
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=rnd_lr)

        self.random_network.eval()

    def update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND network on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).
        """
        # TODO: get states and next_states from the batch
        states = torch.tensor(np.array([transition[0] for transition in training_batch]))
        next_states = torch.tensor(np.array([transition[3] for transition in training_batch]))
        # TODO: compute the MSE
        with torch.no_grad():
            random_embeddings = self.random_network(states)
        predictor_embeddings = self.predictor_network(states)
        mse_loss = nn.MSELoss()(predictor_embeddings, random_embeddings)
        # TODO: update the RND network
        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()
        return mse_loss.item()

    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """Compute the RND bonus for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        # TODO: predict embeddings
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            random_embedding = self.random_network(state)
            predictor_embedding = self.predictor_network(state)
        # TODO: get error
        mse_error = nn.MSELoss()(predictor_embedding, random_embedding)
        return mse_error.item() * self.rnd_reward_weight

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        ep_extrinsic_reward = 0.0
        ep_intrinsic_reward = 0.0
        recent_rewards: List[float] = []
        episode_rewards = []
        episode_extrinsic_rewards = []
        episode_intrinsic_rewards = []
        steps = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # TODO: apply RND bonus
            extrinsic_reward = reward 
            intrinsic_reward = self.get_rnd_bonus(next_state)
            total_reward = extrinsic_reward + intrinsic_reward

            # store and step
            self.buffer.add(state, action, total_reward, next_state, done or truncated, {})
            state = next_state
            ep_extrinsic_reward += extrinsic_reward
            ep_intrinsic_reward += intrinsic_reward
            ep_reward += total_reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

                if self.total_steps % self.rnd_update_freq == 0:
                    self.update_rnd(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                episode_extrinsic_rewards.append(ep_extrinsic_reward)
                episode_intrinsic_rewards.append(ep_intrinsic_reward)
                steps.append(frame)
                ep_reward = 0.0
                ep_extrinsic_reward = 0.0
                ep_intrinsic_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards)
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        # Saving to .csv for simplicity
        # Could also be e.g. npz
        print("RNDGN Training complete.")
        training_data = pd.DataFrame({
            "steps": steps,
            "reward": episode_rewards,
            "extrinsic_reward": episode_extrinsic_rewards,
            "intrinsic_reward": episode_intrinsic_rewards
        })
        try:
            print("Saving to:", os.getcwd())
            training_data.to_csv(f"training_data_seed_{self.seed}_RNDDQN.csv", index=False)
            print("CSV saved successfully.")
        except Exception as e:
            print("Failed to save CSV:", e)


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 3) TODO: instantiate & train the agent
    agent = RNDDQNAgent(
        env=env,
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        seed=3,
        rnd_hidden_size=128,
        rnd_lr=1e-3,
        rnd_update_freq=1000,
        rnd_n_layers=2,
        rnd_reward_weight=0.1
    )
    agent.train(
        num_frames=cfg.train.num_frames,
        eval_interval=cfg.train.eval_interval
    )


if __name__ == "__main__":
    main()