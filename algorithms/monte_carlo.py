import numpy as np
import random

# ============================================================================
# Méthodes Monte Carlo : on apprend à partir d'épisodes COMPLETS (pas de modèle
# des transitions comme en Dynamic Programming, et pas de mise à jour à chaque
# step comme en TD : on attend la fin de l'épisode pour calculer le retour G).
#
# PHASE D'ENTRAÎNEMENT : on génère des épisodes avec une politique exploratrice
# et on met à jour Q(s, a) avec la moyenne des retours observés.
# PHASE D'ÉVALUATION  : on déroule la politique apprise (gloutonne, SANS
# exploration) pour mesurer sa vraie performance -> evaluer_politique()
# ============================================================================


def get_actions(env, state=None):
    if hasattr(env, "available_actions"):
        return env.available_actions()
    return env.actions


def action_gloutonne(Q, state, actions):
    q_values = [Q.get((state, a), 0.0) for a in actions]
    return actions[int(np.argmax(q_values))]


def epsilon_greedy(Q, state, env, epsilon):
    actions = get_actions(env, state)
    if random.random() < epsilon:
        return random.choice(actions)
    return action_gloutonne(Q, state, actions)


def generer_episode(env, choisir_action, max_steps=1000):
    """Joue un épisode complet et retourne la trajectoire
    [(state, action, reward, actions_disponibles), ...]"""
    state = env.reset()
    episode = []
    done = False
    steps = 0
    while not done and steps < max_steps:
        actions = get_actions(env, state)
        action = choisir_action(state, steps, actions)
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward, actions))
        state = next_state
        steps += 1
    return episode


# --------------------------------------------------------------- Monte Carlo ES
def monte_carlo_es(env, episodes=10000, gamma=0.99, max_steps=1000, explore_max=10):
    """
    Monte Carlo Exploring Starts (first visit).
    MC ES exige que chaque paire (état, action) puisse être un point de départ
    d'épisode. Nos environnements démarrent toujours du même état initial, donc
    on simule les exploring starts ainsi :
      - on tire une profondeur aléatoire t_explore dans [0, explore_max]
      - on joue des actions ALÉATOIRES jusqu'à t_explore (pour atteindre un
        état de départ aléatoire), l'action à t_explore est aussi aléatoire
        (c'est le 'start' (s, a)), puis on suit la politique GLOUTONNE
      - la mise à jour ne porte que sur la partie de l'épisode à partir de
        t_explore (le préfixe aléatoire sert seulement à se déplacer)
    Hyperparamètre explore_max : à régler selon la 'profondeur' de
    l'environnement (ex : 4 pour Monty Hall, 30 pour Grid World 5x5).
    """
    Q = {}
    visites = {}                 # nombre de retours observés pour chaque (s, a)
    cumul_par_step = []
    cumul_total = 0.0

    for _ in range(episodes):
        # profondeur du point de départ de l'épisode
        t_explore = random.randint(0, explore_max)

        def choisir(state, t, actions):
            if t <= t_explore:
                return random.choice(actions)          # exploring start
            return action_gloutonne(Q, state, actions)

        episode = generer_episode(env, choisir, max_steps)

        for (_, _, r, _) in episode:
            cumul_total += r
            cumul_par_step.append(cumul_total)

        # mise à jour first visit : on remonte l'épisode depuis la fin,
        # en s'arrêtant au point de départ t_explore (le préfixe purement
        # aléatoire ne suit pas la politique, il ne doit pas être moyenné)
        G = 0.0
        for t in reversed(range(len(episode))):
            s, a, r, _ = episode[t]
            G = gamma * G + r
            if t < t_explore:
                break
            deja_visite = any(e[0] == s and e[1] == a for e in episode[t_explore:t])
            if not deja_visite:
                visites[(s, a)] = visites.get((s, a), 0) + 1
                q_old = Q.get((s, a), 0.0)
                Q[(s, a)] = q_old + (G - q_old) / visites[(s, a)]   # moyenne incrémentale

    return derive_policy(Q, env), Q, cumul_par_step


# ------------------------------------ On-policy first visit Monte Carlo Control
def on_policy_mc_control(env, episodes=10000, gamma=0.99, epsilon=0.1, max_steps=1000):
    """
    On-policy first visit Monte Carlo Control.
    La MÊME politique epsilon-greedy sert à générer les épisodes et est
    améliorée au fur et à mesure (epsilon garantit l'exploration).
    """
    Q = {}
    visites = {}
    cumul_par_step = []
    cumul_total = 0.0

    for _ in range(episodes):
        def choisir(state, t, actions):
            return epsilon_greedy(Q, state, env, epsilon)

        episode = generer_episode(env, choisir, max_steps)

        for (_, _, r, _) in episode:
            cumul_total += r
            cumul_par_step.append(cumul_total)

        G = 0.0
        for t in reversed(range(len(episode))):
            s, a, r, _ = episode[t]
            G = gamma * G + r
            deja_visite = any(e[0] == s and e[1] == a for e in episode[:t])
            if not deja_visite:
                visites[(s, a)] = visites.get((s, a), 0) + 1
                q_old = Q.get((s, a), 0.0)
                Q[(s, a)] = q_old + (G - q_old) / visites[(s, a)]

    return derive_policy(Q, env), Q, cumul_par_step


# ------------------------------------------------ Off-policy Monte Carlo Control
def off_policy_mc_control(env, episodes=10000, gamma=0.99, epsilon=0.1, max_steps=1000):
    """
    Off-policy Monte Carlo Control avec importance sampling pondéré.
    Deux politiques distinctes :
      - politique de COMPORTEMENT (behavior) : epsilon-greedy, génère les épisodes
      - politique CIBLE (target) : gloutonne, c'est elle qu'on apprend
    Le poids W corrige le fait que les épisodes ne sont pas générés par la cible.
    """
    Q = {}
    C = {}                       # somme cumulée des poids pour chaque (s, a)
    cumul_par_step = []
    cumul_total = 0.0

    for _ in range(episodes):
        def choisir(state, t, actions):
            return epsilon_greedy(Q, state, env, epsilon)

        episode = generer_episode(env, choisir, max_steps)

        for (_, _, r, _) in episode:
            cumul_total += r
            cumul_par_step.append(cumul_total)

        G = 0.0
        W = 1.0
        for t in reversed(range(len(episode))):
            s, a, r, actions = episode[t]
            G = gamma * G + r
            C[(s, a)] = C.get((s, a), 0.0) + W
            q_old = Q.get((s, a), 0.0)
            Q[(s, a)] = q_old + (W / C[(s, a)]) * (G - q_old)

            # si l'action ne suit pas la politique cible (gloutonne), le poids
            # des steps précédents devient nul : on arrête la remontée
            if a != action_gloutonne(Q, s, actions):
                break
            # probabilité de a sous la politique de comportement epsilon-greedy
            b = 1.0 - epsilon + epsilon / len(actions)
            W = W / b

    return derive_policy(Q, env), Q, cumul_par_step


# --------------------------------------------------------------------- Outils
def derive_policy(Q, env):
    """Politique gloutonne finale déduite de Q (fin de l'entraînement)."""
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


def evaluer_politique(env, policy, episodes=10000, max_steps=1000):
    """
    PHASE D'ÉVALUATION (séparée de l'entraînement) : on déroule la politique
    apprise SANS exploration ni mise à jour, et on mesure le gain moyen
    par épisode. C'est cette valeur qu'on reporte dans le rapport.
    """
    total = 0.0
    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            actions = get_actions(env, state)
            action = policy.get(state, actions[0])
            if action not in actions:
                action = actions[0]
            state, reward, done = env.step(action)
            total += reward
            steps += 1
    return total / episodes
