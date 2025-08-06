# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning Snake game implementation using Deep Q-Network (DQN) with PyTorch. The project demonstrates training an RL agent to play Snake and provides both training and demo capabilities.

## Key Architecture Components

### Core Structure
- `src/game.py` - Main Snake game implementation with Pygame rendering and model inference
- `src/env.py` - Gym wrapper (SnakeEnv) that provides standardized RL environment interface
- `train/train.py` - DQN training script with replay buffer and epsilon-greedy exploration
- `demo/demo.py` - Demo script to run trained models
- `models/` - Directory containing trained model weights (dqn_snake.pt)

### Game Logic
- 20x20 grid with wrap-around boundaries (snake doesn't die from wall collision)
- State representation: flattened 400-element array (0=empty, 1=snake, 2=food)
- Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
- Reward: +1 for eating food, 0 otherwise
- Game supports both human control (arrow keys) and AI model control

### RL Architecture
- **DQN Network**: 3-layer fully connected (400 → 128 → 128 → 4)
- **Experience Replay**: Buffer stores transitions for stable learning
- **Target Network**: Separate network updated every 10 episodes for stability
- **Epsilon-Greedy**: Exploration starts at 1.0, decays to 0.1 over 1000 episodes

## Development Commands

### Environment Setup
```bash
# Initial setup (creates venv and installs dependencies)
./setup.sh

# Manual activation
source venv/bin/activate
```

### Training
```bash
# Basic training with default parameters
python train/train.py

# Custom training parameters
python train/train.py --episodes 5000 --lr 0.001 --batch_size 128 --epsilon_decay 2000
```

### Running Games
```bash
# Human-controlled game
python src/game.py

# AI demo with trained model
python demo/demo.py --model_path models/dqn_snake.pt

# AI-controlled game directly
python src/game.py --model_path models/dqn_snake.pt
```

### Dependencies
Core dependencies are in `requirements.txt`:
- `pygame` - Game rendering and input
- `gym` - RL environment interface
- `torch` - Neural network and training
- `numpy` - Numerical operations

## Training Parameters
Default training configuration in `train/train.py`:
- Episodes: 2000
- Learning rate: 0.001
- Batch size: 64
- Gamma (discount): 0.99
- Replay buffer: 10,000 transitions
- Target network update frequency: 10 episodes

## File Organization
- Game engine and model inference are combined in `src/game.py`
- Environment wrapper separates RL interface in `src/env.py`
- Training logic is self-contained in `train/train.py`
- Models are saved to `models/` directory automatically