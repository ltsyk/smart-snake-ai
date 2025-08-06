import argparse
import os
import pathlib
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pygame
from src.game import SnakeGame

def main():
    parser = argparse.ArgumentParser(description='演示已训练的强化学习贪吃蛇模型')
    parser.add_argument('--model_path', type=str, required=True, help='已训练模型文件路径，例如 models/dqn_snake.pt')
    args = parser.parse_args()

    # 初始化游戏并加载模型
    game = SnakeGame(model_path=args.model_path)

    # 运行游戏直至结束
    while not game.done:
        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 如果模型已加载，让模型决定动作
        if game.model:
            state = torch.tensor(game.get_state(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = game.model(state)
                action = torch.argmax(q_values, dim=1).item()
            game.step(action)
        else:
            game.step()  # 人工控制

        game.render()

    print(f'演示结束，最终得分: {game.score}')

if __name__ == '__main__':
    main()