import numpy as np
import random
import time

from environments.secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3
from environments.secret_env_adapter import SecretEnvAdapter
from algorithms.temporal_difference import q_learning, sarsa
from utils.persistence import save

random.seed(42)
np.random.seed(42)


def evaluer(env, Q, n_parties=20, max_steps=1000):
    scores = []
    for _ in range(n_parties):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            actions = env.available_actions()
            q_values = [Q.get((state, a), 0.0) for a in actions]
            best = actions[int(np.argmax(q_values))]
            state, reward, done = env.step(best)
            steps += 1
        scores.append(env.env.score())
    return np.mean(scores), min(scores), max(scores)


secret_classes = [
    ("SecretEnv0", SecretEnv0),
    ("SecretEnv1", SecretEnv1),
    ("SecretEnv2", SecretEnv2),
    ("SecretEnv3", SecretEnv3),
]

for nom, cls in secret_classes:
    print(f"\n########## {nom} ##########")

    # Q-Learning
    env = SecretEnvAdapter(cls())
    t0 = time.time()
    policy_ql, Q_ql = q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.3)
    moy, mini, maxi = evaluer(env, Q_ql)
    print(f"Q-Learning : score moyen {moy:.2f} (min {mini}, max {maxi}) | "
          f"{len(Q_ql)} couples | {time.time()-t0:.1f}s")
    save(Q_ql, f"{nom.lower()}_qlearning_Q.pkl")

    # Sarsa
    env = SecretEnvAdapter(cls())
    t0 = time.time()
    policy_s, Q_s = sarsa(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.3)
    moy, mini, maxi = evaluer(env, Q_s)
    print(f"Sarsa      : score moyen {moy:.2f} (min {mini}, max {maxi}) | "
          f"{len(Q_s)} couples | {time.time()-t0:.1f}s")
    save(Q_s, f"{nom.lower()}_sarsa_Q.pkl")

print("\nFinish")