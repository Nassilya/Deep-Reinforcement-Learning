import numpy as np
import random
import time
from environments.secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3
from environments.secret_env_adapter import SecretEnvAdapter
from algorithms.temporal_difference import sarsa, q_learning
from algorithms.dynamic_programming import policy_iteration, value_iteration
from utils.persistence import save

random.seed(42)
np.random.seed(42)

SEUIL_DP = 10000


def evaluer(env, Q_ou_policy, est_Q=True, n_parties=20, max_steps=1000):
    scores = []
    for _ in range(n_parties):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            valid = env.available_actions()
            if est_Q:
                q_values = [Q_ou_policy.get((state, a), 0.0) for a in valid]
                action = valid[int(np.argmax(q_values))]
            else:
                action = Q_ou_policy.get(state, valid[0])
                if action not in valid:
                    action = valid[0]
            state, reward, done = env.step(action)
            steps += 1
        scores.append(env.env.score())
    return np.mean(scores), min(scores), max(scores)


secret_classes = [
    ("SecretEnv0", SecretEnv0),
    ("SecretEnv1", SecretEnv1),
    ("SecretEnv2", SecretEnv2),
    ("SecretEnv3", SecretEnv3),
]

resultats = []

for nom, cls in secret_classes:
    n_etats = cls().num_states()
    print(f"\n{'='*60}")
    print(f"{nom} — {n_etats} états")
    print('='*60)

    #-------------------------------------------------------------------------------------Q-LEARNING 
    env = SecretEnvAdapter(cls())
    t0 = time.time()
    policy_ql, Q_ql, rewards_ql = q_learning(env, episodes=20000, alpha=0.1, gamma=0.99, epsilon=0.3)
    duree = time.time() - t0
    moy, mini, maxi = evaluer(env, Q_ql, est_Q=True)
    print(f"Q-Learning       : score {moy:.2f} (min {mini}, max {maxi}) | {duree:.1f}s")
    save(Q_ql, f"{nom.lower()}_qlearning_Q.pkl")
    resultats.append((nom, "Q-Learning", moy, duree))

    #-------------------------------------------------------------------------SARSA 
    env = SecretEnvAdapter(cls())
    t0 = time.time()
    policy_s, Q_s, rewards_s = sarsa(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.3)
    duree = time.time() - t0
    moy, mini, maxi = evaluer(env, Q_s, est_Q=True)
    print(f"Sarsa            : score {moy:.2f} (min {mini}, max {maxi}) | {duree:.1f}s")
    save(Q_s, f"{nom.lower()}_sarsa_Q.pkl")
    resultats.append((nom, "Sarsa", moy, duree))

    #-------------------------------------------------------------------------DP (ss l'env est assez petit)
    if n_etats > SEUIL_DP:
        print(f"Value Iteration  : NON TESTÉ ({n_etats} états > {SEUIL_DP}, DP impraticable)")
        print(f"Policy Iteration : NON TESTÉ ({n_etats} états > {SEUIL_DP}, DP impraticable)")
        resultats.append((nom, "Value Iteration", None, None))
        resultats.append((nom, "Policy Iteration", None, None))
        continue

    env = SecretEnvAdapter(cls())
    t0 = time.time()
    policy_vi, V_vi, deltas_vi = value_iteration(env, gamma=0.99, theta=1e-4)
    duree = time.time() - t0
    moy, mini, maxi = evaluer(env, policy_vi, est_Q=False)
    print(f"Value Iteration  : score {moy:.2f} (min {mini}, max {maxi}) | {duree:.1f}s")
    save(policy_vi, f"{nom.lower()}_value_iteration.pkl")
    resultats.append((nom, "Value Iteration", moy, duree))

    env = SecretEnvAdapter(cls())
    t0 = time.time()
    policy_pi, V_pi, deltas_pi = policy_iteration(env, gamma=0.99)
    duree = time.time() - t0
    moy, mini, maxi = evaluer(env, policy_pi, est_Q=False)
    print(f"Policy Iteration : score {moy:.2f} (min {mini}, max {maxi}) | {duree:.1f}s")
    save(policy_pi, f"{nom.lower()}_policy_iteration.pkl")
    resultats.append((nom, "Policy Iteration", moy, duree))


#--------------------------------------------------------------------------TABLEAU RÉCAPITULATIF 
print(f"\n\n{'='*70}")
print("RÉCAPITULATIF")
print('='*70)
print(f"{'Environnement':<15}{'Algorithme':<20}{'Score':<12}{'Durée':<10}")
print('-'*70)
for nom, algo, score, duree in resultats:
    s = f"{score:.2f}" if score is not None else "N/A"
    d = f"{duree:.1f}s" if duree is not None else "impraticable"
    print(f"{nom:<15}{algo:<20}{s:<12}{d:<10}")