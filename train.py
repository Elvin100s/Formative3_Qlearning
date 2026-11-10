"""Train a DQN agent on Atari Boxing.

Example: uv run train.py --lr 1e-4 --gamma 0.99 --out-dir models/Elvin/exp_01
Saves dqn_model.pth, training_log.csv and training_plot.png into --out-dir.
"""
import argparse
import csv
import os
import random

import ale_py
import gymnasium as gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")  # plots go to a file, no display needed
import matplotlib.pyplot as plt

gym.register_envs(ale_py)

ENV_ID = "ALE/Boxing-v5"

# Fixed settings, not tuned in the experiments
BUFFER_SIZE = 50_000
TRAIN_EVERY = 4
TARGET_UPDATE_EVERY = 1_000


def make_env(render_mode=None):
    """Boxing as 4 stacked 84x84 grayscale frames, so the network can see motion."""
    # AtariPreprocessing skips frames itself, so the base env must not
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


class QNetwork(nn.Module):
    """CNN from the DQN paper (Mnih et al., 2015): 4 frames in, one Q-value per action out."""

    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)   # 4x84x84 -> 32x20x20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # -> 64x9x9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # -> 64x7x7
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.flatten(start_dim=1)))
        return self.out(x)


class ReplayBuffer:
    """Fixed-size circular buffer. When full, the oldest experience is overwritten."""

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        """Random sample. Randomness breaks the correlation between consecutive frames."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_states[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


def get_epsilon(step, eps_start, eps_end, eps_decay):
    """Linear decay from eps_start to eps_end over eps_decay steps, then stays at eps_end."""
    fraction = min(step / eps_decay, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def select_action(q_net, state, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)  # explore
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
        return q_net(state_tensor).argmax(dim=1).item()  # exploit


def train_step(q_net, target_net, optimizer, buffer, batch_size, gamma, device):
    states, actions, rewards, next_states, dones = buffer.sample(batch_size, device)

    # Q(s, a) from the online network, for the actions actually taken
    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman target: r + gamma * max_a' Q_target(s', a'), or just r if the episode ended
    with torch.no_grad():
        next_q_max = target_net(next_states).max(dim=1).values
        targets = rewards + gamma * next_q_max * (1 - dones)

    # Huber loss: less sensitive to large errors than MSE
    loss = F.smooth_l1_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10)
    optimizer.step()
    return loss.item()


def save_plot(log_rows, path):
    steps = [row[1] for row in log_rows]
    lengths = [row[2] for row in log_rows]
    rewards = [row[3] for row in log_rows]
    window = 20
    avg_rewards = [np.mean(rewards[max(0, i - window + 1): i + 1]) for i in range(len(rewards))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(steps, rewards, color="#2a78d6", alpha=0.25, label="Episode reward")
    ax1.plot(steps, avg_rewards, color="#2a78d6", linewidth=2, label=f"{window}-episode average")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax2.plot(steps, lengths, color="#2a78d6")
    ax2.set_ylabel("Episode length (steps)")
    ax2.set_xlabel("Environment steps")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.2)
    fig.suptitle("Boxing DQN training")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    env = make_env()
    n_actions = env.action_space.n

    # The target network is a periodically synced copy, so the Bellman targets stay stable
    q_net = QNetwork(n_actions).to(device)
    target_net = QNetwork(n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(BUFFER_SIZE, env.observation_space.shape)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "steps", "episode_length", "reward"])
    log_rows = []

    state, _ = env.reset(seed=args.seed)
    episode, episode_reward, episode_length = 0, 0.0, 0

    for step in range(1, args.total_steps + 1):
        epsilon = get_epsilon(step, args.eps_start, args.eps_end, args.eps_decay)
        action = select_action(q_net, state, epsilon, n_actions, device)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Rewards clipped to -1/0/+1 as in the DQN paper. A time-limit cut-off (truncated) is not a real ending.
        buffer.add(state, action, np.sign(reward), next_state, float(terminated))

        state = next_state
        episode_reward += reward  # the unclipped score is what gets logged
        episode_length += 1

        if step > args.learning_starts and step % TRAIN_EVERY == 0:
            train_step(q_net, target_net, optimizer, buffer, args.batch_size, args.gamma, device)

        if step % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(q_net.state_dict())

        if terminated or truncated:
            episode += 1
            row = [episode, step, episode_length, episode_reward]
            log_writer.writerow(row)
            log_file.flush()
            log_rows.append(row)
            print(f"Episode {episode:4d} | step {step:8d} | length {episode_length:5d} | "
                  f"reward {episode_reward:6.1f} | epsilon {epsilon:.3f}")
            state, _ = env.reset()
            episode_reward, episode_length = 0.0, 0

    log_file.close()
    env.close()

    torch.save(q_net.state_dict(), os.path.join(args.out_dir, "dqn_model.pth"))
    if log_rows:
        save_plot(log_rows, os.path.join(args.out_dir, "training_plot.png"))
    print(f"Done. Model and logs saved to {args.out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari Boxing")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=100_000, help="steps for epsilon to fall from start to end")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=10_000, help="random steps to fill the buffer before learning")
    parser.add_argument("--out-dir", default=".", help="folder for dqn_model.pth, training_log.csv, training_plot.png")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
