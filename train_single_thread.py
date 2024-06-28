from environnement import SnakeGame
from model import build_model
from agent_dqn import DQNAgent
import numpy as np
import time

input_shape = (40, 40, 1)  # dimensions of the grid (width, height, channels)
num_actions = 4  # up, down, left, right
model = build_model(input_shape, num_actions)
target_model = build_model(input_shape, num_actions)
agent = DQNAgent(model, target_model, num_actions)

# synchronizing the target model with the main model
agent.target_model.set_weights(agent.model.get_weights())

num_episodes = 1000
graphical_mode = True  # Set to True if you want to enable graphical mode

tps1 = time.time()

for e in range(num_episodes):
    game = SnakeGame(width=40, height=40, graphical_mode=graphical_mode)
    state = game.reset()
    state = np.reshape(state, [1, 40, 40, 1])
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = game.step(action)
        next_state = np.reshape(next_state, [1, 40, 40, 1])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.train()
    
    game.close()

    if e % 10 == 0:
        print(f"Episode: {e}, Score: {game.score}, Epsilon: {agent.epsilon}")

        tps2 = time.time()
        print((tps2 - tps1))
        


    # Update the target model every 50 episodes
    if e % 50 == 0:
        agent.target_model.set_weights(agent.model.get_weights())

    # Save weight every 100 episodes
    if e % 100 == 0:
        model.save_weights(f'snake_dqn_{e}.weights.h5')

# Enregistrement des poids du modèle après l'entraînement
model.save_weights(f'snake_dqn_{num_episodes}.weights.h5')