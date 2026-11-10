"""Watch the trained DQN agent play Atari Boxing.

Example: uv run play.py --model models/Elvin/exp_03/dqn_model.pth --episodes 2
"""
import argparse

import torch

from train import QNetwork, make_env


def play(model_path, episodes):
    env = make_env(render_mode="human")  # opens a game window

    q_net = QNetwork(env.action_space.n)
    q_net.load_state_dict(torch.load(model_path, map_location="cpu"))
    q_net.eval()

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward, done = 0.0, False

        while not done:
            # Greedy: always pick the action with the highest predicted Q-value
            with torch.no_grad():
                q_values = q_net(torch.as_tensor(state).unsqueeze(0))
                action = q_values.argmax(dim=1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode}: score {total_reward:+.0f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch the trained DQN agent play Boxing")
    parser.add_argument("--model", default="dqn_model.pth")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    play(args.model, args.episodes)
