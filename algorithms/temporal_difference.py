import numpy as np
import random


def epsilon_greedy(Q, state, actions, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)                      
    q_values = [Q.get((state, a), 0.0) for a in actions]
    return int(np.argmax(q_values))


def sarsa(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = {}                                                   

    for _ in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, env.actions, epsilon)   
        done = False

        while not done:
            next_state, reward, done = env.step(action)      
            next_action = epsilon_greedy(Q, next_state, env.actions, epsilon)

            q_current = Q.get((state, action), 0.0)
            q_next = Q.get((next_state, next_action), 0.0)    
            Q[(state, action)] = q_current + alpha * (reward + gamma * q_next - q_current)

            state = next_state
            action = next_action                              

    return derive_policy(Q, env), Q


def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = {}

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy(Q, state, env.actions, epsilon)   
            next_state, reward, done = env.step(action)

            q_current = Q.get((state, action), 0.0)
            q_next_max = max([Q.get((next_state, a), 0.0) for a in env.actions])
            Q[(state, action)] = q_current + alpha * (reward + gamma * q_next_max - q_current)

            state = next_state

    return derive_policy(Q, env), Q


def derive_policy(Q, env):
    policy = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = 0
            continue
        q_values = [Q.get((s, a), 0.0) for a in env.actions]
        policy[s] = int(np.argmax(q_values))
    return policy