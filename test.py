import numpy as np
import pygame
from environnement import SnakeGame
from model import build_model

input_shape = (40, 40, 1)  # dimensions of the grid (width, height, channels)
num_actions = 4  # up, down, left, right
model = build_model(input_shape, num_actions)

# Charger les poids enregistrés
model.load_weights('snake_dqn.weights.h5')

graphical_mode = True  # Activer le mode graphique pour voir le jeu

# Fonction de test
def test_model():
    game = SnakeGame(graphical_mode=graphical_mode)
    state = game.reset()
    state = np.reshape(state, [1, 40, 40, 1])
    done = False
    score = 0
    while not done:
        action = np.argmax(model.predict(state)[0])
        next_state, reward, done, _ = game.step(action)
        next_state = np.reshape(next_state, [1, 40, 40, 1])
        state = next_state
        score += reward
    game.close()
    print(f"Score: {score}")

# Tester le modèle
test_model()
