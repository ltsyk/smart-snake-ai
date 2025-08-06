#!/usr/bin/env python3
"""
控制台版本的AI演示，显示游戏状态而不需要图形界面
"""
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
import torch
import torch.nn as nn
from src.env import SnakeEnv

class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def print_game_state(env, step, action, score):
    """打印游戏状态到控制台"""
    action_names = ['上', '下', '左', '右']
    print(f"步数: {step:3d} | 动作: {action_names[action]} | 得分: {score}")

def main():
    parser = argparse.ArgumentParser(description='控制台版本的AI演示')
    parser.add_argument('--model_path', type=str, default='models/dqn_snake.pt',
                        help='训练模型路径')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='最大步数')
    args = parser.parse_args()

    # 加载模型
    model = DQNNet(400, 4)
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ 模型已加载: {args.model_path}")

    # 创建环境
    env = SnakeEnv()
    state = env.reset()
    print("✓ 环境已初始化")
    print("=" * 50)
    
    step = 0
    total_reward = 0
    
    while step < args.max_steps:
        # 模型推理
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # 打印状态
        if reward > 0 or done:  # 只在得分或游戏结束时打印
            print_game_state(env, step, action, env.game.score)
        
        state = next_state
        step += 1
        
        if done:
            print("=" * 50)
            print(f"🎮 游戏结束!")
            print(f"📊 最终得分: {env.game.score}")
            print(f"🏃 总步数: {step}")
            print(f"💰 总奖励: {total_reward}")
            break
    else:
        print("=" * 50)
        print(f"⏱️  达到最大步数限制: {args.max_steps}")
        print(f"📊 当前得分: {env.game.score}")

if __name__ == '__main__':
    main()