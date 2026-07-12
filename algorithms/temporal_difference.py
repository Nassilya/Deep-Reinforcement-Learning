import numpy as np
import random

def get_actions(env, state=None):
    if hasattr(env, "available_actions"):
        return env.available_actions()
    return env.actions

def epsilon_greedy(Q, state, env, epsilon):
    actions = get_actions(env, state)
    if random.random() < epsilon:
        return random.choice(actions)
    q_values = [Q.get((state, a), 0.0) for a in actions]
    return actions[int(np.argmax(q_values))]


def sarsa(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1, max_steps=1000):
    Q = {}
    cumul_par_step = []        
    cumul_total = 0.0            

    for _ in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, env, epsilon)
        done = False
        steps = 0

        while not done and steps < max_steps:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(Q, next_state, env, epsilon)

            q_current = Q.get((state, action), 0.0)
            q_next = Q.get((next_state, next_action), 0.0)
            Q[(state, action)] = q_current + alpha * (reward + gamma * q_next - q_current)

            cumul_total += reward              
            cumul_par_step.append(cumul_total) 

            state = next_state
            action = next_action
            steps += 1

    return derive_policy(Q, env), Q, cumul_par_step


def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1, max_steps=1000):
    Q = {}
    cumul_par_step = []         
    cumul_total = 0.0           

    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = epsilon_greedy(Q, state, env, epsilon)
            next_state, reward, done = env.step(action)

            q_current = Q.get((state, action), 0.0)
            next_actions = get_actions(env, next_state)
            q_next_max = max([Q.get((next_state, a), 0.0) for a in next_actions], default=0.0)

            Q[(state, action)] = q_current + alpha * (reward + gamma * q_next_max - q_current)

            cumul_total += reward               
            cumul_par_step.append(cumul_total) 

            state = next_state
            steps += 1

    return derive_policy(Q, env), Q, cumul_par_step

def derive_policy(Q, env):
    policy = {}
    if hasattr(env, "available_actions"):
        etats_vus = set(s for (s, a) in Q.keys())
        for s in etats_vus:
            actions_vues = sorted(set(a for (st, a) in Q.keys() if st == s))
            q_values = [Q[(s, a)] for a in actions_vues]
            policy[s] = actions_vues[int(np.argmax(q_values))]
        return policy

    for s in env.states:
        if env.is_terminal(s):
            policy[s] = 0
            continue
        q_values = [Q.get((s, a), 0.0) for a in env.actions]
        policy[s] = int(np.argmax(q_values))
    return policy