import argparse
import os
import random
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from src.env import SnakeEnv

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

def train(args):
    env = SnakeEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(obs_dim, n_actions)
    target_net = DQN(obs_dim, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    replay = ReplayBuffer(capacity=args.replay_size)

    epsilon = args.epsilon_start
    epsilon_decay = (args.epsilon_start - args.epsilon_end) / args.epsilon_decay

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = q_vals.argmax().item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            replay.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay) >= args.batch_size:
                batch_state, batch_action, batch_reward, batch_next, batch_done = replay.sample(args.batch_size)

                q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    next_q = target_net(batch_next).max(1)[0]
                    target = batch_reward + args.gamma * next_q * (1 - batch_done)

                loss = criterion(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(args.epsilon_end, epsilon - epsilon_decay)

        if episode % args.log_interval == 0:
            print(f'Episode {episode}/{args.episodes} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}')

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, 'dqn_snake.pt')
    torch.save(policy_net.state_dict(), model_path)
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.1)
    parser.add_argument('--epsilon_decay', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=10000)
    parser.add_argument('--target_update', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()
    train(args)