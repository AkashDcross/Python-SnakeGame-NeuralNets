# 🐍 Snake Game with Neural Networks (Pygame + PyTorch)

A classic Snake game powered by a neural network trained via reinforcement learning. The AI learns to play Snake using deep Q-learning and PyTorch, all rendered in real-time with Pygame.

---

## 📂 Project Structure

snake-pygame/
├── agent.py         # RL agent logic
├── helper.py        # Utility functions
├── model.py         # Neural network model (PyTorch)
├── snake_game.py    # Game environment (Pygame)
├── model/           # Saved models and training checkpoints

---
## 🎮 How It Works

The game works in the following steps:

1. **State Representation**  
   The current game state (position of the snake, food, and obstacles) is converted into a numerical format.

2. **Decision Making**  
   The neural network (agent) predicts the best action to take (up, down, left, right) based on the current state.

3. **Action Execution**  
   The snake moves in the direction chosen by the neural network.

4. **Reward or Penalty**  
   After each action, the agent receives a reward or penalty based on the outcome (e.g., eating food, dying, or moving closer to food).

5. **Learning**  
   The agent updates its neural network using **Deep Q-Learning** based on the feedback it receives, improving its strategy over time.
---
## 🧠 Technologies Used

- **Python 3**: For the overall development of the project.
- **Pygame**: For game rendering and handling the game logic.
- **PyTorch**: For creating and training the neural network.
- **Reinforcement Learning**: To enable the agent to learn through rewards and penalties over time.

---
