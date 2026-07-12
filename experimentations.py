import numpy as np
import random

from environments.line_world import LineWorld
from environments.grid_world import GridWorld
from algorithms.dynamic_programming import value_iteration
from algorithms.temporal_difference import sarsa, q_learning
from utils.plots import comparer_cumuls, tracer_valeurs

random.seed(42)
np.random.seed(42)

def experience_hyperparametre(algo, nom_algo, param, valeurs, fichier, log_x=False):
    """
    Lance un algo avec plusieurs valeurs d'un hyperparamètre et compare les cumuls.
    algo   : la fonction (sarsa ou q_learning)
    param  : "epsilon" ou "alpha"
    """
    print(f"\n=== Effet de {param} sur {nom_algo} (Grid World) ===")
    resultats = {}

    for v in valeurs:
        random.seed(42)                      # même graine : comparaison honnête
        np.random.seed(42)
        env = GridWorld(rows=5, cols=5)
        print(f"  {param} = {v} ...")

        # paramètres par défaut, on remplace celui qu'on étudie
        kwargs = {"episodes": 20000, "alpha": 0.1, "gamma": 0.99, "epsilon": 0.3}
        kwargs[param] = v

        _, _, cumul = algo(env, **kwargs)
        resultats[f"{param} = {v}"] = cumul

    comparer_cumuls(
        resultats,
        titre=f"Grid World — {nom_algo} : effet de {param}",
        fichier=fichier,
        log_x=log_x
    )

def experience_epsilon():
    """Effet du taux d'exploration sur Q-Learning (Grid World)."""
    print("\n=== Effet d'epsilon (Q-Learning, Grid World) ===")
    resultats = {}
    for eps in [0.01, 0.1, 0.3, 0.5]:
        random.seed(42)
        np.random.seed(42)
        env = GridWorld(rows=5, cols=5)
        print(f"  epsilon = {eps} ...")
        _, _, cumul = q_learning(env, episodes=20000, alpha=0.1, gamma=0.99, epsilon=eps)
        resultats[f"epsilon = {eps}"] = cumul

    comparer_cumuls(
        resultats,
        titre="Grid World  Q-Learning : effet du taux d'exploration (epsilon)",
        fichier="graphe_epsilon.png"
    )


def experience_alpha():
    """Effet du taux d'apprentissage sur Sarsa (Grid World)."""
    print("\n=== Effet d'alpha (Sarsa, Grid World) ===")
    resultats = {}
    for a in [0.01, 0.1, 0.5, 0.9]:
        random.seed(42)
        np.random.seed(42)
        env = GridWorld(rows=5, cols=5)
        print(f"  alpha = {a} ...")
        _, _, cumul = sarsa(env, episodes=20000, alpha=a, gamma=0.99, epsilon=0.3)
        resultats[f"alpha = {a}"] = cumul

    comparer_cumuls(
        resultats,
        titre="Grid World  Sarsa : effet du taux d'apprentissage (alpha)",
        fichier="graphe_alpha.png"
    )


def experience_gamma():
    """Effet du facteur d'actualisation sur les valeurs V (Line World)."""
    print("\n=== Effet de gamma (Value Iteration, Line World) ===")
    env = LineWorld(size=5)
    resultats = {}
    for g in [0.5, 0.9, 0.99]:
        _, V, _ = value_iteration(env, gamma=g)
        valeurs = [V[s] for s in env.states]
        print(f"  gamma = {g} : {[round(v, 4) for v in valeurs]}")
        resultats[f"gamma = {g}"] = valeurs

    tracer_valeurs(
        resultats,
        etats=list(env.states),
        titre="Line World Value Iteration : effet de gamma sur les valeurs V(s)",
        fichier="graphe_gamma.png"
    )


if __name__ == "__main__":
    #experience_epsilon()
    #experience_alpha()
    #experience_gamma()
    #les 2 manquants :
    experience_hyperparametre(sarsa, "Sarsa", "epsilon",
                              [0.01, 0.1, 0.3, 0.5], "graphe_epsilon_sarsa.png", log_x=True)

    experience_hyperparametre(q_learning, "Q-Learning", "alpha",
                              [0.01, 0.1, 0.5, 0.9], "graphe_alpha_qlearning.png")
