import numpy as np
import pygame
from environnement import SnakeGame

# Configuration du jeu
width = 40
height = 40
block_size = 20
graphical_mode = True  # Mettre à False pour désactiver le mode graphique

# Initialisation du jeu
game = SnakeGame(width=width, height=height, block_size=block_size, graphical_mode=graphical_mode)

# Réinitialisation du jeu
state = game.reset()

game.close()
