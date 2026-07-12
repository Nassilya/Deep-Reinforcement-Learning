import time
import numpy as np
import random
from environments.secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3
from environments.secret_env_adapter import SecretEnvAdapter
from algorithms.temporal_difference import q_learning

random.seed(42)
np.random.seed(42)

print("Mesure du temps de Q-Learning (mêmes paramètres partout)")
for nom, cls in [("SecretEnv0", SecretEnv0), ("SecretEnv1", SecretEnv1),
                 ("SecretEnv2", SecretEnv2), ("SecretEnv3", SecretEnv3)]:
    env = SecretEnvAdapter(cls())
    n = env.env.num_states()
    t0 = time.time()
    q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.3, max_steps=1000)
    print(f"{nom} ({n} états) : {time.time()-t0:.1f}s")