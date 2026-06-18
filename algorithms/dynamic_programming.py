import numpy as np
def policy_evaluation(env, policy, V, gamma=0.99, theta=1e-6): 
    while True:
        delta = 0                                   
        for s in env.states:
            if env.is_terminal(s):                
                continue
            v_old = V[s]
            a = policy[s]                          
            new_v = 0.0
            for prob, s_next, reward in env.transition_prob(s, a):
                new_v += prob * (reward + gamma * V[s_next])   
            V[s] = new_v
            delta = max(delta, abs(v_old - V[s]))   
        if delta < theta:                          
            break
    return V


def policy_improvement(env, V, gamma=0.99):
    new_policy = {}
    for s in env.states:
        if env.is_terminal(s):
            new_policy[s] = 0                      
            continue
        action_values = []
        for a in env.actions:
            q = 0.0
            for prob, s_next, reward in env.transition_prob(s, a):
                q += prob * (reward + gamma * V[s_next])    
            action_values.append(q)
        new_policy[s] = int(np.argmax(action_values))       
    return new_policy


def policy_iteration(env, gamma=0.99):
    V = {s: 0.0 for s in env.states}                
    policy = {s: 0 for s in env.states}            

    while True:
        V = policy_evaluation(env, policy, V, gamma)
        new_policy = policy_improvement(env, V, gamma)
        if new_policy == policy:                    
            break
        policy = new_policy

    return policy, V

#------------------------------------------------------------------------------------------- Value Iteration
def value_iteration(env, gamma=0.99, theta=1e-6):
    V = {s: 0.0 for s in env.states}               
    while True:
        delta = 0
        for s in env.states:
            if env.is_terminal(s):                 
                continue
            v_old = V[s]
            action_values = []
            for a in env.actions:
                q = 0.0
                for prob, s_next, reward in env.transition_prob(s, a):
                    q += prob * (reward + gamma * V[s_next])   
                action_values.append(q)
            V[s] = max(action_values)               
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break

    policy = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = 0
            continue
        action_values = []
        for a in env.actions:
            q = 0.0
            for prob, s_next, reward in env.transition_prob(s, a):
                q += prob * (reward + gamma * V[s_next])
            action_values.append(q)
        policy[s] = int(np.argmax(action_values))  

    return policy, V