#!/usr/bin/env python3
"""
评估改进版模型的性能表现
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from statistics import mean, stdev
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.env_improved import ImprovedSnakeEnv

class ImprovedDQN(nn.Module):
    """与训练一致的Dueling DQN结构"""
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
        adv = self.adv_stream(features)
        return value + adv - adv.mean(dim=1, keepdim=True)

def evaluate_improved_model(model_path, num_episodes=100, max_steps=1000, verbose=False):
    """评估改进模型性能"""
    # 加载模型
    model = ImprovedDQN(408, 4, hidden_size=128)  # 使用训练时的参数
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    env = ImprovedSnakeEnv()
    scores = []
    episode_lengths = []
    total_rewards = []
    
    print(f"开始评估改进模型 (共 {num_episodes} 轮)...")
    
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
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        scores.append(info.get('score', 0))
        episode_lengths.append(steps)
        total_rewards.append(total_reward)
        
        if verbose and (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: Score = {scores[-1]}, Steps = {steps}, Reward = {total_reward:.1f}")
    
    return scores, episode_lengths, total_rewards

def compare_models(original_results, improved_results):
    """对比原始和改进模型的性能"""
    print("\n" + "="*70)
    print("📊 模型性能对比分析")
    print("="*70)
    
    orig_scores, orig_lengths = original_results
    imp_scores, imp_lengths, imp_rewards = improved_results
    
    print(f"{'指标':<20} | {'原始模型':<15} | {'改进模型':<15} | {'提升':<15}")
    print("-" * 70)
    
    # 平均得分对比
    orig_mean = mean(orig_scores)
    imp_mean = mean(imp_scores)
    improvement = ((imp_mean - orig_mean) / max(orig_mean, 0.01)) * 100
    print(f"{'平均得分':<20} | {orig_mean:<15.2f} | {imp_mean:<15.2f} | {improvement:+.1f}%")
    
    # 最高得分对比
    orig_max = max(orig_scores)
    imp_max = max(imp_scores)
    print(f"{'最高得分':<20} | {orig_max:<15d} | {imp_max:<15d} | {imp_max - orig_max:+d}")
    
    # 成功率对比
    orig_success = len([s for s in orig_scores if s > 0]) / len(orig_scores) * 100
    imp_success = len([s for s in imp_scores if s > 0]) / len(imp_scores) * 100
    print(f"{'成功率':<20} | {orig_success:<15.1f}% | {imp_success:<15.1f}% | {imp_success - orig_success:+.1f}%")
    
    # 平均步数对比
    orig_steps = mean(orig_lengths)
    imp_steps = mean(imp_lengths)
    print(f"{'平均步数':<20} | {orig_steps:<15.1f} | {imp_steps:<15.1f} | {imp_steps - orig_steps:+.1f}")
    
    print(f"{'平均奖励':<20} | {'N/A':<15} | {mean(imp_rewards):<15.1f} | {'New Metric':<15}")

def analyze_improved_performance(scores, episode_lengths, total_rewards):
    """分析改进模型的性能指标"""
    print("\n" + "="*60)
    print("📊 改进模型详细分析报告")
    print("="*60)
    
    # 得分统计
    print(f"🎯 得分统计:")
    print(f"   平均得分: {mean(scores):.2f}")
    print(f"   最高得分: {max(scores)}")
    print(f"   最低得分: {min(scores)}")
    if len(scores) > 1:
        print(f"   标准差: {stdev(scores):.2f}")
    
    # 奖励统计
    print(f"\n💰 奖励统计:")
    print(f"   平均奖励: {mean(total_rewards):.2f}")
    print(f"   最高奖励: {max(total_rewards):.1f}")
    print(f"   最低奖励: {min(total_rewards):.1f}")
    
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
    
    # 高分游戏分析
    high_score_games = len([s for s in scores if s >= 2])
    high_score_rate = high_score_games / len(scores) * 100
    print(f"   高分游戏(≥2分): {high_score_games}/{len(scores)} ({high_score_rate:.1f}%)")
    
    return {
        'mean_score': mean(scores),
        'max_score': max(scores),
        'success_rate': success_rate,
        'mean_steps': mean(episode_lengths),
        'mean_reward': mean(total_rewards)
    }

def main():
    parser = argparse.ArgumentParser(description='评估改进模型性能')
    parser.add_argument('--improved_model', type=str, default='models/improved_dqn_snake.pt',
                        help='改进模型路径')
    parser.add_argument('--original_model', type=str, default='models/dqn_snake.pt',
                        help='原始模型路径')
    parser.add_argument('--episodes', type=int, default=50,
                        help='评估轮数')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='每轮最大步数')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细输出')
    parser.add_argument('--compare', action='store_true',
                        help='与原始模型对比')
    args = parser.parse_args()
    
    # 评估改进模型
    imp_scores, imp_lengths, imp_rewards = evaluate_improved_model(
        args.improved_model, args.episodes, args.max_steps, args.verbose
    )
    
    metrics = analyze_improved_performance(imp_scores, imp_lengths, imp_rewards)
    
    # 如果需要对比
    if args.compare:
        try:
            # 读取原始模型评估结果
            with open('evaluation_results.txt', 'r') as f:
                lines = f.readlines()
                orig_scores = eval(lines[-1].split(': ')[1])
            
            # 简单计算原始模型步数（假设大部分达到最大步数）
            orig_lengths = [1000 if s == 0 else 100 for s in orig_scores]  # 简化估计
            
            compare_models((orig_scores, orig_lengths), (imp_scores, imp_lengths, imp_rewards))
        except:
            print("\n⚠️  无法读取原始模型评估结果，跳过对比")
    
    # 保存结果
    results_file = 'improved_evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Improved Model: {args.improved_model}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"Mean Score: {metrics['mean_score']:.2f}\n")
        f.write(f"Max Score: {metrics['max_score']}\n")
        f.write(f"Success Rate: {metrics['success_rate']:.1f}%\n")
        f.write(f"Mean Steps: {metrics['mean_steps']:.1f}\n")
        f.write(f"Mean Reward: {metrics['mean_reward']:.2f}\n")
        f.write(f"All Scores: {imp_scores}\n")
        f.write(f"All Rewards: {imp_rewards}\n")
    
    print(f"\n💾 结果已保存到: {results_file}")

if __name__ == '__main__':
    main()
