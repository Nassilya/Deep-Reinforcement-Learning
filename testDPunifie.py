import time
import numpy as np

from environments.secret_envs_wrapper import SecretEnv0
from environments.secret_env_adapter import SecretEnvAdapter
from algorithms.dynamic_programming import value_iteration
from algorithms.dynamic_programming import policy_iteration   
from utils.persistence import save

env = SecretEnvAdapter(SecretEnv0())

print("policy_iteration (version unifiée) sur SecretEnv0...")
t0 = time.time()
policy, V = policy_iteration(env, gamma=0.99)
print(f"Durée : {time.time()-t0:.1f}s")

save(policy, "secretenv0_policy_iteration.pkl")

scores = []
for _ in range(20):
    state = env.reset()
    done = False
    steps = 0
    while not done and steps < 1000:
        action = policy.get(state, 0)
        valid = env.available_actions()
        if action not in valid:
            action = valid[0]
        state, reward, done = env.step(action)
        steps += 1
    scores.append(env.env.score())

print(f"Score moyen : {np.mean(scores):.2f} (min {min(scores)}, max {max(scores)})")