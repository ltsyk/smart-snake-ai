import argparse
import os
import random
import numpy as np
import sys
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from src.env_improved import ImprovedSnakeEnv

class ImprovedDQN(nn.Module):
    """引入Dueling结构的改进DQN网络"""
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(ImprovedDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.adv_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities.append(max_priority)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(list(self.priorities)[:len(self.buffer)])
        
        # 计算采样概率
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
            indices,
            torch.tensor(weights, dtype=torch.float32)
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

def train_improved(args):
    """改进的训练函数"""
    env = ImprovedSnakeEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # 设置设备 (CUDA/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"🖥️  使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    policy_net = ImprovedDQN(obs_dim, n_actions, args.hidden_size).to(device)
    target_net = ImprovedDQN(obs_dim, n_actions, args.hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    
    if args.use_prioritized_replay:
        replay = PrioritizedReplayBuffer(capacity=args.replay_size)
    else:
        # 简单回放缓冲区实现
        class SimpleReplayBuffer:
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
        replay = SimpleReplayBuffer(capacity=args.replay_size)

    epsilon = args.epsilon_start
    linear_decay = (args.epsilon_start - args.epsilon_end) / args.epsilon_decay
    
    best_score = -float('inf')
    recent_scores = deque(maxlen=100)

    print(f"开始训练改进模型...")
    print(
        f"使用Dueling DQN结构，特征层: {obs_dim} -> {args.hidden_size} -> {args.hidden_size}, "
        f"分支: {args.hidden_size//2} -> {n_actions}"
    )
    
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < args.max_steps:
            # Epsilon-greedy 策略
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_vals = policy_net(state_tensor)
                    action = q_vals.argmax().item()

            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            replay.push(state, action, reward, next_state, done)
            state = next_state

            # 训练
            if len(replay) >= args.batch_size:
                if args.use_prioritized_replay:
                    batch_state, batch_action, batch_reward, batch_next, batch_done, indices, weights = replay.sample(args.batch_size)
                    
                    # 移动到设备
                    batch_state = batch_state.to(device)
                    batch_action = batch_action.to(device)
                    batch_reward = batch_reward.to(device)
                    batch_next = batch_next.to(device)
                    batch_done = batch_done.to(device)
                    weights = weights.to(device)
                    
                    q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze()
                    with torch.no_grad():
                        next_actions = policy_net(batch_next).argmax(1)
                        next_q = target_net(batch_next).gather(1, next_actions.unsqueeze(1)).squeeze()
                        target = batch_reward + args.gamma * next_q * (1 - batch_done)

                    # 计算TD误差
                    td_errors = torch.abs(q_values - target)
                    
                    # 加权损失
                    loss = (weights * (q_values - target) ** 2).mean()
                    
                    # 更新优先级
                    replay.update_priorities(indices, td_errors.detach().cpu().numpy())
                    
                else:
                    batch_state, batch_action, batch_reward, batch_next, batch_done = replay.sample(args.batch_size)
                    
                    # 移动到设备
                    batch_state = batch_state.to(device)
                    batch_action = batch_action.to(device)
                    batch_reward = batch_reward.to(device)
                    batch_next = batch_next.to(device)
                    batch_done = batch_done.to(device)
                    
                    q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze()
                    with torch.no_grad():
                        next_actions = policy_net(batch_next).argmax(1)
                        next_q = target_net(batch_next).gather(1, next_actions.unsqueeze(1)).squeeze()
                        target = batch_reward + args.gamma * next_q * (1 - batch_done)
                    loss = nn.MSELoss()(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), args.grad_clip)
                optimizer.step()

        # 更新目标网络
        if episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 更新epsilon和学习率
        if args.epsilon_strategy == 'linear':
            epsilon = max(args.epsilon_end, epsilon - linear_decay)
        else:
            epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * math.exp(-episode / args.epsilon_decay)
        scheduler.step()

        # 记录分数
        game_score = info.get('score', 0)
        recent_scores.append(game_score)
        
        # 保存最佳模型
        if game_score > best_score:
            best_score = game_score
            os.makedirs(args.save_dir, exist_ok=True)
            best_model_path = os.path.join(args.save_dir, 'best_dqn_snake.pt')
            torch.save(policy_net.state_dict(), best_model_path)

        # 日志输出
        if episode % args.log_interval == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            print(f'Episode {episode:4d}/{args.episodes} | '
                  f'Score: {game_score:2d} | '
                  f'Avg: {avg_score:.2f} | '
                  f'Best: {best_score:2d} | '
                  f'Steps: {steps:3d} | '
                  f'Reward: {total_reward:6.1f} | '
                  f'ε: {epsilon:.3f} | '
                  f'LR: {scheduler.get_last_lr()[0]:.1e}')

    # 保存最终模型
    os.makedirs(args.save_dir, exist_ok=True)
    final_model_path = os.path.join(args.save_dir, 'improved_dqn_snake.pt')
    torch.save(policy_net.state_dict(), final_model_path)
    print(f'训练完成！最终模型保存到: {final_model_path}')
    print(f'最佳模型保存到: {best_model_path}')
    print(f'最佳得分: {best_score}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练改进的DQN模型')
    
    # 基本参数
    parser.add_argument('--episodes', type=int, default=5000, help='训练轮数')
    parser.add_argument('--max_steps', type=int, default=2000, help='每轮最大步数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    
    # 网络参数
    parser.add_argument('--hidden_size', type=int, default=256, help='隐藏层大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪')
    
    # 探索参数
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='初始epsilon')
    parser.add_argument('--epsilon_end', type=float, default=0.05, help='最终epsilon')
    parser.add_argument('--epsilon_decay', type=int, default=2000, help='epsilon衰减速率')
    parser.add_argument('--epsilon_strategy', choices=['linear', 'exp'], default='linear', help='epsilon衰减策略')
    
    # 回放缓冲区
    parser.add_argument('--replay_size', type=int, default=50000, help='回放缓冲区大小')
    parser.add_argument('--use_prioritized_replay', action='store_true', help='使用优先经验回放')
    
    # 其他参数
    parser.add_argument('--target_update', type=int, default=100, help='目标网络更新频率')
    parser.add_argument('--log_interval', type=int, default=50, help='日志输出间隔')
    parser.add_argument('--save_dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_cuda', action='store_true', help='使用CUDA加速训练')
    
    # 学习率衰减
    parser.add_argument('--lr_decay_step', type=int, default=1000, help='学习率衰减步数')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.8, help='学习率衰减系数')
    
    args = parser.parse_args()
    train_improved(args)
