import matplotlib.pyplot as plt
import numpy as np

#Courbe de récompense cumulée par épisode (Sarsa / Q Learning)
def tracer_apprentissage(rewards, titre="Courbe d'apprentissage", fenetre=100):
    if len(rewards) < fenetre:
        fenetre = max(1, len(rewards) // 10)
    lissee = np.convolve(rewards, np.ones(fenetre) / fenetre, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(lissee)
    plt.xlabel("Épisode")
    plt.ylabel(f"Récompense cumulée (moyenne sur {fenetre} épisodes)")
    plt.title(titre)
    plt.grid(True)
    plt.show()


#Courbe du delta par itération (Policy / Value Iteration)
def tracer_convergence(deltas, titre="Convergence"):
    plt.figure(figsize=(10, 5))
    plt.plot(deltas)
    plt.xlabel("Itération")
    plt.ylabel("Delta (changement maximal)")
    plt.title(titre)
    plt.grid(True)
    plt.show()

#Superpose plusieurs courbes
def comparer_apprentissage(dict_rewards, titre="Comparaison", fenetre=100):
    plt.figure(figsize=(10, 5))
    for nom, rewards in dict_rewards.items():
        f = min(fenetre, max(1, len(rewards) // 10))
        lissee = np.convolve(rewards, np.ones(f) / f, mode='valid')
        plt.plot(lissee, label=nom)
    plt.xlabel("Épisode")
    plt.ylabel("Récompense cumulée (lissée)")
    plt.title(titre)
    plt.legend()
    plt.grid(True)
    plt.show()


def comparer_cumuls(dict_cumuls, titre="Comparaison des algorithmes", fichier=None, log_x=False):
    plt.figure(figsize=(11, 6))
    for nom, cumuls in dict_cumuls.items():
        plt.plot(cumuls, label=nom)
    plt.xlabel("Nombre de steps")
    plt.ylabel("Récompense cumulée depuis le début")
    plt.title(titre)
    if log_x:
        plt.xscale("log")
    plt.legend()
    plt.grid(True)
    if fichier:
        plt.savefig(fichier, dpi=150, bbox_inches="tight")
    plt.show()

def tracer_valeurs(dict_valeurs, etats, titre="Valeurs V(s)", fichier=None):
    plt.figure(figsize=(10, 6))
    for label, valeurs in dict_valeurs.items():
        plt.plot(etats, valeurs, marker="o", label=label)
    plt.xlabel("État")
    plt.ylabel("Valeur V(s)")
    plt.title(titre)
    plt.xticks(etats)
    plt.legend()
    plt.grid(True)
    if fichier:
        plt.savefig(fichier, dpi=150, bbox_inches="tight")
    plt.show()