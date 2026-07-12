import time
import random
import numpy as np

from environments.line_world import LineWorld
from environments.grid_world import GridWorld
from algorithms.dynamic_programming import value_iteration
from algorithms.temporal_difference import q_learning

random.seed(42)
np.random.seed(42)

# ---------- LINE WORLD ----------
env = LineWorld(size=5)

t0 = time.time()
value_iteration(env, gamma=0.99)
print(f"Line World — Value Iteration : {time.time()-t0:.4f}s")

t0 = time.time()
q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.3)
print(f"Line World — Q-Learning      : {time.time()-t0:.4f}s")

# ---------- GRID WORLD ----------
env = GridWorld(rows=5, cols=5)

t0 = time.time()
value_iteration(env, gamma=0.99)
print(f"Grid World — Value Iteration : {time.time()-t0:.4f}s")

t0 = time.time()
q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.3)
print(f"Grid World — Q-Learning      : {time.time()-t0:.4f}s")