import threading
import numpy as np
import tensorflow as tf
from environnement import SnakeGame
from agent_dqn import DQNAgent
from model import build_model

# Verrou global pour la mise à jour du modèle et de la mémoire
model_lock = threading.Lock()
memory_lock = threading.Lock()

# Fonction d'entraînement d'un seul thread
def train_thread(agent, num_episodes, input_shape, graphical_mode, num_thread):
    for e in range(num_episodes):
        game = SnakeGame(width=input_shape[0], height=input_shape[1], graphical_mode=graphical_mode)
        state = game.reset()
        state = np.reshape(state, [1, input_shape[0], input_shape[1], 1])
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = game.step(action)
            next_state = np.reshape(next_state, [1, input_shape[0], input_shape[1], 1])
            
            with memory_lock:
                agent.remember(state, action, reward, next_state, done)
                
            state = next_state
            with model_lock:
                agent.train()
        
        game.close()

        if e % 10 == 0:
            print(f"Thread {num_thread} | Episode: {e}, Score: {game.score}, Epsilon: {agent.epsilon}")

        # Mettre à jour le modèle cible tous les 50 épisodes
        if e % 50 == 0:
            with model_lock:
                agent.target_model.set_weights(agent.model.get_weights())

        # Save weight every 100 episodes
        if e % 100 == 0 and e > 0:
            model.save_weights(f'weights/snake_{e}e_thread{num_thread}.weights.h5')

# Définition de l'agent et des modèles
input_shape = (40, 40, 1)  # dimensions de la grille (largeur, hauteur, canaux)
num_actions = 4  # haut, bas, gauche, droite
model = build_model(input_shape, num_actions)
target_model = build_model(input_shape, num_actions)
agent = DQNAgent(model, target_model, num_actions)

# Synchronisation du modèle cible avec le modèle principal
agent.target_model.set_weights(agent.model.get_weights())

# Nombre total d'épisodes
total_episodes = 1000
# Nombre de threads à utiliser
num_threads = 16
# Nombre d'épisodes par thread
episodes_per_thread = total_episodes // num_threads

# Créer et lancer les threads
threads = []
for i in range(num_threads):
    thread = threading.Thread(target=train_thread, args=(agent, episodes_per_thread, input_shape, False, i))
    threads.append(thread)
    thread.start()

# Attendre que tous les threads se terminent
for thread in threads:
    thread.join()

print("Entraînement terminé.")

# Enregistrement des poids du modèle après l'entraînement
model.save_weights(f'weights/snake_{total_episodes}_final.weights.h5')