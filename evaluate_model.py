#!/usr/bin/env python3
"""
评估现有模型的性能表现
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from statistics import mean, stdev
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
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

def evaluate_model(model_path, num_episodes=100, max_steps=1000, verbose=False):
    """评估模型性能"""
    # 加载模型
    model = DQNNet(400, 4)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    env = SnakeEnv()
    scores = []
    episode_lengths = []
    
    print(f"开始评估模型 (共 {num_episodes} 轮)...")
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            # 模型推理
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = model(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        scores.append(env.game.score)
        episode_lengths.append(steps)
        
        if verbose and (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: Score = {env.game.score}, Steps = {steps}")
    
    return scores, episode_lengths

def analyze_performance(scores, episode_lengths):
    """分析性能指标"""
    print("\n" + "="*60)
    print("📊 模型性能分析报告")
    print("="*60)
    
    # 得分统计
    print(f"🎯 得分统计:")
    print(f"   平均得分: {mean(scores):.2f}")
    print(f"   最高得分: {max(scores)}")
    print(f"   最低得分: {min(scores)}")
    if len(scores) > 1:
        print(f"   标准差: {stdev(scores):.2f}")
    
    # 步数统计
    print(f"\n🏃 步数统计:")
    print(f"   平均步数: {mean(episode_lengths):.1f}")
    print(f"   最长游戏: {max(episode_lengths)} 步")
    print(f"   最短游戏: {min(episode_lengths)} 步")
    
    # 成功率分析
    successful_games = len([s for s in scores if s > 0])
    success_rate = successful_games / len(scores) * 100
    print(f"\n🎮 游戏表现:")
    print(f"   成功得分游戏: {successful_games}/{len(scores)} ({success_rate:.1f}%)")
    
    # 得分分布
    score_distribution = {}
    for score in scores:
        score_range = f"{score//5*5}-{score//5*5+4}" if score > 0 else "0"
        score_distribution[score_range] = score_distribution.get(score_range, 0) + 1
    
    print(f"\n📈 得分分布:")
    for score_range in sorted(score_distribution.keys(), key=lambda x: int(x.split('-')[0])):
        count = score_distribution[score_range]
        percentage = count / len(scores) * 100
        print(f"   {score_range}: {count} games ({percentage:.1f}%)")
    
    return {
        'mean_score': mean(scores),
        'max_score': max(scores),
        'success_rate': success_rate,
        'mean_steps': mean(episode_lengths)
    }

def main():
    parser = argparse.ArgumentParser(description='评估模型性能')
    parser.add_argument('--model_path', type=str, default='models/dqn_snake.pt',
                        help='模型路径')
    parser.add_argument('--episodes', type=int, default=100,
                        help='评估轮数')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='每轮最大步数')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细输出')
    args = parser.parse_args()
    
    scores, episode_lengths = evaluate_model(
        args.model_path, args.episodes, args.max_steps, args.verbose
    )
    
    metrics = analyze_performance(scores, episode_lengths)
    
    # 保存结果
    results_file = 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"Mean Score: {metrics['mean_score']:.2f}\n")
        f.write(f"Max Score: {metrics['max_score']}\n")
        f.write(f"Success Rate: {metrics['success_rate']:.1f}%\n")
        f.write(f"Mean Steps: {metrics['mean_steps']:.1f}\n")
        f.write(f"All Scores: {scores}\n")
    
    print(f"\n💾 结果已保存到: {results_file}")

if __name__ == '__main__':
    main()