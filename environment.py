import numpy as np
import random

class GridEnvironment:
    def __init__(self, size=8):
        self.size = size
        self.grid = np.zeros((size, size))
        self.obstacles = {}  
        self.rewards = {}    
        self.time_step = 0
        self.max_steps = 64
        self.total_rewards = 0
        self.total_obstacles = 0
        self.trail = set()
        self.reset_positions()

    def reset_positions(self):
        borders = []
        for i in range(self.size):
            borders.extend([(i, 0), (i, self.size-1), (0, i), (self.size-1, i)])
        borders = list(set(borders))
        self.start_pos = random.choice(borders)
        remaining_borders = [pos for pos in borders if pos != self.start_pos]
        self.end_pos = random.choice(remaining_borders)
        self.current_pos = self.start_pos

    def reset(self):
        self.current_pos = self.start_pos
        self.time_step = 0
        self.total_rewards = 0
        self.total_obstacles = 0
        self.trail.clear()
        return self.get_state()

    def get_state(self):
        state = np.zeros((self.size, self.size))
        state[self.current_pos] = 1
        state[self.end_pos] = 2
        for pos, obs_type in self.obstacles.items():
            state[pos] = -obs_type
        for pos, reward_value in self.rewards.items():
            state[pos] = reward_value
        return state

    def add_random_elements(self, obstacle_types, reward_types, count=10):
        all_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        random.shuffle(all_positions)

        added = 0
        for pos in all_positions:
            if pos == self.start_pos or pos == self.end_pos:
                continue
            if pos in self.obstacles or pos in self.rewards:
                continue

            if random.random() < 0.5:
                label, value = random.choice(list(obstacle_types.items()))
                self.obstacles[pos] = value
            else:
                label, value = random.choice(list(reward_types.items()))
                self.rewards[pos] = value

            added += 1
            if added >= count:
                break

    def step(self, action):
        # Movimiento del agente
        x, y = self.current_pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y < self.size - 1:
            y += 1
        elif action == 2 and x < self.size - 1:
            x += 1
        elif action == 3 and y > 0:
            y -= 1

        new_pos = (x, y)
        self.time_step += 1
        reached_goal = new_pos == self.end_pos

        # Guardar rastro
        self.trail.add(self.current_pos)
        self.current_pos = new_pos

        # Calcular recompensa
        reward = 0

        # Recompensa por llegar a la meta (más si llega rápido)
        if reached_goal:
            reward += 10 * (1 - self.time_step / self.max_steps)

        # Penalización por paso base
        reward -= 0.1

        # Obstáculo
        if new_pos in self.obstacles:
            obs_penalty = self.obstacles[new_pos]
            self.total_obstacles += obs_penalty
            reward -= obs_penalty  # penalización extra

        # Recompensa recogida
        if new_pos in self.rewards:
            reward_bonus = abs(self.rewards[new_pos]) * 2
            self.total_rewards += abs(self.rewards[new_pos])
            reward += reward_bonus

        done = reached_goal or self.time_step >= self.max_steps
        return self.get_state(), reward, done 