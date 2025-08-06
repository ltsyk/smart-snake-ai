import gym
import numpy as np
from gym import spaces
from .game import SnakeGame, GRID_SIZE, DIR_VECTORS

class ImprovedSnakeEnv(gym.Env):
    """改进版的贪吃蛇环境，具有更好的奖励机制和状态表示"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # 观察空间：游戏网格 + 额外特征
        # 400 (网格) + 8 (额外特征：头部位置、食物位置、方向、距离)
        self.observation_space = spaces.Box(low=-1, high=2,
                                          shape=(408,),
                                          dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.game = None
        self.last_score = 0
        self.last_distance = 0

    def reset(self):
        self.game = SnakeGame()
        self.game.reset()
        self.last_score = 0
        self.last_distance = self._calculate_food_distance()
        return self._get_enhanced_state()

    def step(self, action):
        old_score = self.game.score
        old_distance = self._calculate_food_distance()
        old_length = len(self.game.snake)
        
        # 执行动作
        self.game.step(action)
        
        # 计算奖励
        reward = self._calculate_reward(old_score, old_distance, old_length, action)
        
        # 更新状态
        self.last_score = self.game.score
        self.last_distance = self._calculate_food_distance()
        
        state = self._get_enhanced_state()
        done = self.game.done
        info = {
            'score': self.game.score,
            'length': len(self.game.snake),
            'distance': self.last_distance
        }
        
        return state, reward, done, info

    def _calculate_food_distance(self):
        """计算蛇头到食物的曼哈顿距离"""
        if not self.game or not self.game.snake:
            return 0
        
        head = self.game.snake[0]
        food = self.game.food
        
        # 考虑网格的环绕特性
        dx = min(abs(head[0] - food[0]), GRID_SIZE - abs(head[0] - food[0]))
        dy = min(abs(head[1] - food[1]), GRID_SIZE - abs(head[1] - food[1]))
        
        return dx + dy

    def _calculate_reward(self, old_score, old_distance, old_length, action):
        """计算更精细的奖励函数"""
        reward = 0
        
        # 1. 吃到食物的大奖励
        if self.game.score > old_score:
            reward += 10.0
        
        # 2. 接近食物的小奖励
        new_distance = self._calculate_food_distance()
        if new_distance < old_distance:
            reward += 1.0
        elif new_distance > old_distance:
            reward -= 0.5
        
        # 3. 生存奖励（鼓励活得更久）
        if not self.game.done:
            reward += 0.1
        
        # 4. 死亡惩罚
        if self.game.done:
            reward -= 10.0
        
        # 5. 避免无意义的移动（防止在原地打转）
        if len(self.game.snake) > 1:
            head = self.game.snake[0]
            neck = self.game.snake[1]
            if head == neck:  # 撞到自己
                reward -= 5.0
        
        return reward

    def _get_enhanced_state(self):
        """获取增强的状态表示"""
        # 基本网格状态
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # 标记蛇身（不同部位不同值）
        for i, (x, y) in enumerate(self.game.snake):
            if i == 0:  # 蛇头
                grid[y, x] = 1.0
            else:  # 蛇身
                grid[y, x] = 0.5
        
        # 标记食物
        fx, fy = self.game.food
        grid[fy, fx] = 2.0
        
        # 扁平化网格
        grid_flat = grid.flatten()
        
        # 额外特征
        head_x, head_y = self.game.snake[0]
        food_x, food_y = self.game.food
        
        # 归一化位置
        head_pos = [head_x / GRID_SIZE, head_y / GRID_SIZE]
        food_pos = [food_x / GRID_SIZE, food_y / GRID_SIZE]
        
        # 当前方向（one-hot编码）
        direction_vector = [0, 0, 0, 0]
        if hasattr(self.game, 'direction'):
            for i, dir_vec in enumerate(DIR_VECTORS):
                if self.game.direction == dir_vec:
                    direction_vector[i] = 1.0
                    break
        
        # 组合所有特征
        enhanced_features = np.array(head_pos + food_pos + direction_vector, dtype=np.float32)
        
        # 合并状态
        full_state = np.concatenate([grid_flat, enhanced_features])
        
        return full_state

    def render(self, mode='human'):
        if self.game:
            self.game.render()

    def close(self):
        pass