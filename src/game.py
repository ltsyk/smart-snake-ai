import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import pygame
import random
import numpy as np
import torch
import torch.nn as nn

# Constants
WINDOW_SIZE = (400, 400)
GRID_SIZE = 20
CELL_SIZE = WINDOW_SIZE[0] // GRID_SIZE
FPS = 10

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIR_VECTORS = [UP, DOWN, LEFT, RIGHT]

def get_random_position(snake):
    """Return a random grid cell not occupied by the snake."""
    while True:
        pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if pos not in snake:
            return pos

class SnakeGame:
    def __init__(self, model_path=None):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption('强化学习贪吃蛇')
        self.clock = pygame.time.Clock()
        self.reset()
        self.model = None
        if model_path:
            # Load the trained DQN model for inference
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

            obs_dim = GRID_SIZE * GRID_SIZE
            self.model = DQNNet(obs_dim, 4)
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()

    def reset(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = random.choice(DIR_VECTORS)
        self.food = get_random_position(self.snake)
        self.score = 0
        self.done = False

    def step(self, action=None):
        """Advance one frame. If action is None, use human keyboard input."""
        if self.done:
            return

        # Human control
        if action is None:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] and self.direction != DOWN:
                self.direction = UP
            elif keys[pygame.K_DOWN] and self.direction != UP:
                self.direction = DOWN
            elif keys[pygame.K_LEFT] and self.direction != RIGHT:
                self.direction = LEFT
            elif keys[pygame.K_RIGHT] and self.direction != LEFT:
                self.direction = RIGHT
        else:
            # Action from model: 0=UP,1=DOWN,2=LEFT,3=RIGHT
            self.direction = DIR_VECTORS[int(action)]

        # Compute new head position
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        new_head = ((head_x + dir_x) % GRID_SIZE, (head_y + dir_y) % GRID_SIZE)

        # Check self‑collision
        if new_head in self.snake:
            self.done = True
            return

        # Insert new head
        self.snake.insert(0, new_head)

        # Check food
        if new_head == self.food:
            self.score += 1
            self.food = get_random_position(self.snake)
        else:
            # Remove tail
            self.snake.pop()

    def render(self):
        self.screen.fill((0, 0, 0))

        # Draw food
        food_rect = pygame.Rect(self.food[0] * CELL_SIZE,
                                self.food[1] * CELL_SIZE,
                                CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, (255, 0, 0), food_rect)

        # Draw snake
        for segment in self.snake:
            seg_rect = pygame.Rect(segment[0] * CELL_SIZE,
                                   segment[1] * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, (0, 255, 0), seg_rect)

        # Score
        font = pygame.font.SysFont(None, 24)
        score_surf = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_surf, (5, 5))

        pygame.display.flip()
        self.clock.tick(FPS)

    def get_state(self):
        """Return a simple state representation for the RL agent."""
        # Flatten grid: 0=empty, 1=snake, 2=food
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        for x, y in self.snake:
            grid[y, x] = 1
        fx, fy = self.food
        grid[fy, fx] = 2
        return grid.flatten()

def main():
    parser = argparse.ArgumentParser(description='强化学习贪吃蛇')
    parser.add_argument('--model_path', type=str, default=None,
                        help='已训练模型文件路径')
    args = parser.parse_args()

    game = SnakeGame(model_path=args.model_path)

    while not game.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # If a model is loaded, let it decide the action
        if game.model:
            state = torch.tensor(game.get_state(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = game.model(state)
                action = torch.argmax(q_values, dim=1).item()
            game.step(action)
        else:
            game.step()  # human control

        game.render()

    print(f'游戏结束，最终得分: {game.score}')

if __name__ == '__main__':
    main()