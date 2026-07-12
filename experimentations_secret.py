import numpy as np
import random
import time
import matplotlib.pyplot as plt

from environments.secret_envs_wrapper import SecretEnv0, SecretEnv2
from environments.secret_env_adapter import SecretEnvAdapter
from algorithms.temporal_difference import sarsa, q_learning
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
            valid = env.available_actions()
            q_values = [Q.get((state, a), 0.0) for a in valid]
            action = valid[int(np.argmax(q_values))]
            state, reward, done = env.step(action)
            steps += 1
        scores.append(env.env.score())
    return np.mean(scores), min(scores), max(scores)


# =====================================================================
# GRAPHIQUE 1 : temps de calcul en fonction du nombre d'états (log-log)
# =====================================================================
def graphe_scalabilite():
    print("\n=== Graphique 1 : passage à l'échelle (DP vs TD) ===")

    etats     = [5,       25,      8192,   65536]
    temps_dp  = [0.0001,  0.0007,  340.3,  21600]
    temps_td  = [0.03,    0.28,    12.6,   63.9]  

    plt.figure(figsize=(11, 6))
    plt.plot(etats, temps_dp, marker="o", linewidth=2,
             label="Dynamic Programming (Value Iteration)")
    plt.plot(etats, temps_td, marker="s", linewidth=2,
             label="Temporal Difference (Q-Learning)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Nombre d'états (échelle log)")
    plt.ylabel("Temps de calcul en secondes (échelle log)")
    plt.title("Passage à l'échelle : coût de calcul selon la taille de l'espace d'états")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    noms = ["Line World", "Grid World", "SecretEnv0", "SecretEnv1"]
    for x, y, nom in zip(etats, temps_dp, noms):
        plt.annotate(nom, (x, y), textcoords="offset points", xytext=(5, 8), fontsize=9)

    plt.savefig("graphe_scalabilite.png", dpi=150, bbox_inches="tight")
    plt.show()


# =====================================================================
# GRAPHIQUE 2 : compromis qualité / coût sur SecretEnv0
# =====================================================================
def graphe_qualite_cout():
    print("\n=== Graphique 2 : compromis qualité/coût (SecretEnv0) ===")

    algos  = ["Policy\nIteration", "Value\nIteration", "Sarsa", "Q-Learning"]
    scores = [11.0,  11.0,  9.0, 9.0]
    temps  = [343.6, 340.3, 2.1, 12.6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    couleurs = ["#2E86AB", "#2E86AB", "#F18F01", "#F18F01"]

    ax1.bar(algos, scores, color=couleurs)
    ax1.set_ylabel("Score obtenu")
    ax1.set_title("Qualité de la politique (SecretEnv0)")
    ax1.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(scores):
        ax1.text(i, v + 0.2, str(v), ha="center", fontweight="bold")

    ax2.bar(algos, temps, color=couleurs)
    ax2.set_ylabel("Temps de calcul (secondes)")
    ax2.set_title("Coût de calcul (SecretEnv0)")
    ax2.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(temps):
        ax2.text(i, v + 8, f"{v}s", ha="center", fontweight="bold")

    plt.suptitle("Compromis qualité / coût : DP (bleu) vs TD (orange)", fontsize=13)
    plt.tight_layout()
    plt.savefig("graphe_qualite_cout.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    graphe_scalabilite()     
    graphe_qualite_cout()