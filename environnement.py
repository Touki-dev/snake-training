import numpy as np
import random
import pygame

class SnakeGame:
    def __init__(self, width=40, height=40, block_size=20, graphical_mode=False):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.graphical_mode = graphical_mode
        self.reset()

        if self.graphical_mode:
            pygame.init()
            self.display = pygame.display.set_mode((self.width * self.block_size, self.height * self.block_size))
            self.clock = pygame.time.Clock()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.food = self._place_food()
        self.done = False
        self.direction = (0, 1)  # initial direction: moving right
        self.score = 0
        return self._get_observation()

    def _place_food(self):
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food

    def _get_observation(self):
        grid = np.zeros((self.width, self.height))
        for segment in self.snake:
            grid[segment] = 1
        grid[self.food] = 2
        return grid

    def step(self, action):
        # define action space: 0=up, 1=down, 2=left, 3=right
        if action == 0 and self.direction != (1, 0):  # Up
            self.direction = (-1, 0)
        elif action == 1 and self.direction != (-1, 0):  # Down
            self.direction = (1, 0)
        elif action == 2 and self.direction != (0, 1):  # Left
            self.direction = (0, -1)
        elif action == 3 and self.direction != (0, -1):  # Right
            self.direction = (0, 1)
        
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        
        if (new_head in self.snake or 
            new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            self.done = True
            return self._get_observation(), -1, self.done, {}
        
        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.food = self._place_food()
            self.score += 1
            reward = 1
        else:
            self.snake.pop()
            reward = 0

        if self.graphical_mode:
            self.render()
        
        return self._get_observation(), reward, self.done, {}

    def render(self):
        self.display.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(segment[1] * self.block_size, segment[0] * self.block_size, self.block_size, self.block_size))
        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food[1] * self.block_size, self.food[0] * self.block_size, self.block_size, self.block_size))
        pygame.display.flip()
        self.clock.tick(10)  # control the frame rate

    def close(self):
        if self.graphical_mode:
            pygame.quit()
