#!/usr/bin/env python3
"""
改进版AI模型的演示脚本
"""
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import pygame
from src.env_improved import ImprovedSnakeEnv

class ImprovedDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(ImprovedDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser(description='演示改进版强化学习贪吃蛇模型')
    parser.add_argument('--model_path', type=str, default='models/improved_dqn_snake.pt',
                        help='改进模型文件路径')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='隐藏层大小（需要与训练时保持一致）')
    args = parser.parse_args()

    # 加载模型
    model = ImprovedDQN(408, 4, hidden_size=args.hidden_size)
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ 改进模型已加载: {args.model_path}")

    # 创建环境
    env = ImprovedSnakeEnv()
    state = env.reset()
    
    print("🎮 开始AI演示...")
    print("📝 关闭游戏窗口以退出")
    
    step_count = 0
    
    # 游戏循环
    while not env.game.done:
        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 模型推理
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

        # 执行动作
        next_state, reward, done, info = env.step(action)
        state = next_state
        step_count += 1
        
        # 每100步显示一次信息
        if step_count % 100 == 0:
            print(f"步数: {step_count}, 得分: {info['score']}, 奖励: {reward:.1f}")

        # 渲染游戏
        env.render()
        
        # 限制最大步数，防止无限循环
        if step_count >= 2000:
            print("⏱️  达到最大步数限制")
            break

    print(f"🎯 游戏结束！")
    print(f"📊 最终得分: {env.game.score}")
    print(f"🏃 总步数: {step_count}")
    
    pygame.quit()

if __name__ == '__main__':
    main()