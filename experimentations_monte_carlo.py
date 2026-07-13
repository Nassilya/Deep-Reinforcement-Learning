import numpy as np
import random

from environments.grid_world import GridWorld
from environments.monty_hall_level1 import MontyHallLevel1
from environments.monty_hall_level2 import MontyHallLevel2
from algorithms.monte_carlo import monte_carlo_es, on_policy_mc_control, off_policy_mc_control
from utils.plots import comparer_cumuls

random.seed(42)
np.random.seed(42)


def experience_epsilon_mc():
    """Effet du taux d'exploration sur l'On-policy MC Control (Monty Hall 2)."""
    print("\n=== Effet d'epsilon (On-policy MC, Monty Hall 2) ===")
    resultats = {}
    for eps in [0.01, 0.1, 0.3, 0.5]:
        random.seed(42)                      # même graine : comparaison honnête
        np.random.seed(42)
        env = MontyHallLevel2()
        print(f"  epsilon = {eps} ...")
        _, _, cumul = on_policy_mc_control(env, episodes=50000, gamma=0.99, epsilon=eps)
        resultats[f"epsilon = {eps}"] = cumul

    comparer_cumuls(
        resultats,
        titre="Monty Hall 2  On-policy MC : effet du taux d'exploration (epsilon)",
        fichier="graphe_epsilon_mc.png"
    )


def experience_explore_max_mc():
    """Effet de la profondeur des exploring starts sur MC ES (Grid World).
    Trop petit : l'agent ne découvre jamais le but en bas à droite,
    la politique gloutonne n'apprend rien."""
    print("\n=== Effet d'explore_max (MC ES, Grid World) ===")
    resultats = {}
    for em in [2, 10, 30, 50]:
        random.seed(42)
        np.random.seed(42)
        env = GridWorld(rows=5, cols=5)
        print(f"  explore_max = {em} ...")
        _, _, cumul = monte_carlo_es(env, episodes=20000, gamma=0.99, explore_max=em)
        resultats[f"explore_max = {em}"] = cumul

    comparer_cumuls(
        resultats,
        titre="Grid World  MC ES : effet de la profondeur des exploring starts",
        fichier="graphe_explore_max_mc.png",
        log_x=True
    )


def experience_gamma_mc():
    """Effet du facteur d'actualisation sur l'On-policy MC Control (Grid World)."""
    print("\n=== Effet de gamma (On-policy MC, Grid World) ===")
    resultats = {}
    for g in [0.5, 0.9, 0.99]:
        random.seed(42)
        np.random.seed(42)
        env = GridWorld(rows=5, cols=5)
        print(f"  gamma = {g} ...")
        _, _, cumul = on_policy_mc_control(env, episodes=20000, gamma=g, epsilon=0.1)
        resultats[f"gamma = {g}"] = cumul

    comparer_cumuls(
        resultats,
        titre="Grid World  On-policy MC : effet du facteur d'actualisation (gamma)",
        fichier="graphe_gamma_mc.png"
    )


def experience_comparaison_mc_mh1():
    """Les 3 algorithmes Monte Carlo comparés sur Monty Hall 1."""
    print("\n=== Comparaison des 3 MC (Monty Hall 1) ===")
    resultats = {}
    for nom, lancer in {
        "MC ES": lambda e: monte_carlo_es(e, episodes=20000, explore_max=2),
        "On-policy MC": lambda e: on_policy_mc_control(e, episodes=20000, epsilon=0.1),
        "Off-policy MC": lambda e: off_policy_mc_control(e, episodes=20000, epsilon=0.1),
    }.items():
        random.seed(42)
        np.random.seed(42)
        env = MontyHallLevel1()
        print(f"  {nom} ...")
        _, _, cumul = lancer(env)
        resultats[nom] = cumul

    comparer_cumuls(
        resultats,
        titre="Monty Hall 1 : comparaison des 3 algorithmes Monte Carlo",
        fichier="graphe_comparaison_mc_mh1.png"
    )


if __name__ == "__main__":
    experience_epsilon_mc()
    experience_explore_max_mc()
    experience_gamma_mc()
    experience_comparaison_mc_mh1()
