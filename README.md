# Smart Snake AI

Advanced Deep Q-Network reinforcement learning for Snake game with cross-platform support and CUDA GPU acceleration.

![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Cross Platform](https://img.shields.io/badge/Platform-Windows%20Linux%20Mac-lightgrey.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)

## Key Features

- Advanced DQN Algorithm: Enhanced Deep Q-Network with prioritized experience replay
- CUDA Acceleration: GPU training support for significantly faster training
- Cross-Platform Support: Compatible with Windows, Linux, and macOS
- Rich State Representation: 408-dimensional state space with position, direction, and distance information
- Interactive Demos: Pygame graphical interface and console modes
- Performance Analysis: Detailed training and evaluation metrics

## Quick Start

### Windows Users

1. **Clone Repository**
   ```bash
   git clone https://github.com/ltsyk/smart-snake-ai.git
   cd smart-snake-ai
   ```

2. **Environment Setup**
   
   **CPU Version (recommended for beginners):**
   ```batch
   setup_windows.bat
   ```
   
   **GPU Version (requires NVIDIA GPU):**
   ```batch
   setup_cuda.bat
   ```

3. **Start Training**
   ```batch
   train_windows.bat
   ```

4. **Run Demo**
   ```batch
   demo_windows.bat
   ```

### Linux/Mac Users

1. **Environment Setup**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   source venv/bin/activate
   
   # GPU support (optional)
   pip install -r requirements-cuda.txt
   ```

2. **Start Training**
   ```bash
   # CPU training
   python train/train_improved.py --episodes 1000
   
   # GPU training
   python train/train_improved.py --episodes 5000 --use_cuda --use_prioritized_replay
   ```

3. **Run Demo**
   ```bash
   # Graphical demo
   python demo_improved.py --model_path models/improved_dqn_snake.pt
   
   # Console demo
   python evaluate_improved.py --episodes 50
   ```

## 🏗️ 项目结构

```
snake-rl/
├── src/                          # 核心源码
│   ├── game.py                  # 贪吃蛇游戏引擎
│   ├── env.py                   # 基础RL环境
│   └── env_improved.py          # 改进RL环境
├── train/                        # 训练脚本
│   ├── train.py                 # 基础DQN训练
│   └── train_improved.py        # 改进DQN训练
├── demo/                         # 演示脚本
│   └── demo.py                  # 基础演示
├── models/                       # 模型文件
│   ├── dqn_snake.pt            # 原始模型
│   ├── improved_dqn_snake.pt   # 改进模型
│   └── best_dqn_snake.pt       # 最佳模型
├── demo_improved.py             # 改进模型演示
├── evaluate_improved.py        # 性能评估
├── *_windows.bat               # Windows批处理脚本
├── requirements.txt            # CPU依赖
├── requirements-cuda.txt       # CUDA依赖
└── CLAUDE.md                   # 开发指南
```

## 🎮 使用方法

### 训练模型

**快速训练（200轮）：**
```bash
python train/train_improved.py --episodes 200 --hidden_size 128
```

**完整训练（5000轮+CUDA）：**
```bash
python train/train_improved.py --episodes 5000 --use_cuda --use_prioritized_replay --hidden_size 256
```

**自定义参数：**
```bash
python train/train_improved.py \
    --episodes 3000 \
    --use_cuda \
    --hidden_size 512 \
    --lr 0.001 \
    --batch_size 128 \
    --use_prioritized_replay
```

### 模型评估

```bash
# 评估改进模型
python evaluate_improved.py --improved_model models/improved_dqn_snake.pt --episodes 100

# 对比原始模型
python evaluate_improved.py --compare --episodes 50
```

### 演示运行

```bash
# 图形界面演示
python demo_improved.py --model_path models/best_dqn_snake.pt

# 人工游戏（方向键控制）
python src/game.py
```

## 📊 性能对比

| 指标 | 原始模型 | 改进模型 | 提升幅度 |
|------|----------|----------|----------|
| **平均得分** | 0.02 | 0.16 | **+700%** |
| **成功率** | 2.0% | 16.0% | **+14%** |
| **训练稳定性** | 低 | 高 | **显著提升** |

## 🔧 技术特点

### 改进的DQN算法
- **增强状态表示**：408维特征（网格+位置+方向+距离）
- **丰富奖励函数**：接近食物(+1)、吃到食物(+10)、生存(+0.1)、死亡(-10)
- **深层网络**：256→256→128→4的四层网络
- **正则化**：Dropout防止过拟合
- **优先经验回放**：提升学习效率

### CUDA支持
- **自动设备检测**：CPU/GPU自动选择
- **内存优化**：高效的GPU内存使用
- **批处理优化**：GPU并行计算
- **混合精度**：支持半精度训练（可选）

## ⚙️ 依赖要求

### 基础要求
- Python 3.8+
- PyTorch 2.0+
- Pygame 2.5+
- NumPy 1.21+

### GPU支持（可选）
- NVIDIA显卡（GTX 1060+/RTX系列）
- CUDA 11.8+
- 4GB+ VRAM

### 推荐配置
- **CPU训练**：Intel i5/AMD R5 + 8GB RAM
- **GPU训练**：RTX 3060+ + 16GB RAM + 6GB VRAM

## 📈 训练建议

### 新手设置
```bash
python train/train_improved.py --episodes 500 --hidden_size 128 --log_interval 25
```

### 进阶设置
```bash
python train/train_improved.py --episodes 3000 --use_cuda --hidden_size 256 --use_prioritized_replay --lr 0.0005
```

### 专家设置
```bash
python train/train_improved.py --episodes 10000 --use_cuda --hidden_size 512 --use_prioritized_replay --lr 0.0001 --batch_size 256 --epsilon_decay 3000
```

## 🎯 训练参数详解

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `--episodes` | 5000 | 训练轮数 | 1000-10000 |
| `--hidden_size` | 256 | 隐藏层大小 | 128, 256, 512 |
| `--lr` | 0.001 | 学习率 | 0.0001-0.01 |
| `--batch_size` | 128 | 批次大小 | 64, 128, 256 |
| `--epsilon_decay` | 2000 | 探索衰减 | 500-5000 |
| `--use_cuda` | False | 使用GPU | True/False |
| `--use_prioritized_replay` | False | 优先经验回放 | True/False |

## 🔍 故障排除

### 常见问题

1. **CUDA不可用**
   ```bash
   # 检查CUDA安装
   python -c "import torch; print(torch.cuda.is_available())"
   
   # 重新安装CUDA版本
   pip uninstall torch torchvision torchaudio
   pip install -r requirements-cuda.txt
   ```

2. **Pygame显示问题**
   ```bash
   # Windows: 安装Visual C++运行库
   # Linux: sudo apt-get install python3-pygame
   # Mac: brew install pygame
   ```

3. **内存不足**
   ```bash
   # 减少批次大小
   python train/train_improved.py --batch_size 64 --replay_size 20000
   ```

4. **训练过慢**
   ```bash
   # 使用GPU加速
   python train/train_improved.py --use_cuda --batch_size 256
   ```

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢OpenAI Gym提供强化学习环境框架
- 感谢PyTorch团队提供深度学习框架
- 感谢Pygame社区提供游戏开发支持

## 📞 联系方式

- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues页面]
- 💬 Discussions: [GitHub Discussions页面]

---

⭐ 如果这个项目对您有帮助，请给个Star支持！