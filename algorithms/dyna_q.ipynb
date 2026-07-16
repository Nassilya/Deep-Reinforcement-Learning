import random
from environments.rps_deux_round import JeuPFC        # <-- on recupere l'environnement de l'autre fichier

# --- hyperparametres ---
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
N_PLANIF = 10
NB_PARTIES = 2000

Q = [[0.0, 0.0, 0.0] for _ in range(5)]
modele = {}


def meilleure_action(etat):
    maxi = max(Q[etat])
    candidates = []
    for a in range(3):
        if Q[etat][a] == maxi:
            candidates.append(a)
    return random.choice(candidates)


def choisir_action(etat):
    if random.random() < EPSILON:
        return random.randint(0, 2)
    return meilleure_action(etat)


def mise_a_jour_q(etat, action, recompense, etat_suivant):
    if etat_suivant == 4:
        futur = 0.0
    else:
        futur = max(Q[etat_suivant])
    cible = recompense + GAMMA * futur
    Q[etat][action] = Q[etat][action] + ALPHA * (cible - Q[etat][action])


# --- entrainement ---
jeu = JeuPFC()

for partie in range(NB_PARTIES):
    etat = jeu.reset()
    fini = False
    while not fini:
        action = choisir_action(etat)                    # 1) agir
        etat_suivant, recompense, fini = jeu.step(action)
        mise_a_jour_q(etat, action, recompense, etat_suivant)   # 2) apprendre
        modele[(etat, action)] = (recompense, etat_suivant)     # 3) memoriser
        experiences_connues = list(modele.keys())
        for _ in range(N_PLANIF):                        # 4) planifier
            (e, a) = random.choice(experiences_connues)
            (r, e_suiv) = modele[(e, a)]
            mise_a_jour_q(e, a, r, e_suiv)
        etat = etat_suivant

print("Entrainement termine (", NB_PARTIES, "parties )")
print()

noms_etats = [
    "etat 0 (round 0)          ",
    "etat 1 (j'avais Pierre)   ",
    "etat 2 (j'avais Feuille)  ",
    "etat 3 (j'avais Ciseaux)  ",
]
print("Q-table apprise :")
print("                             Pierre   Feuille  Ciseaux")
for etat in range(4):
    ligne = noms_etats[etat]
    for a in range(3):
        ligne = ligne + str(round(Q[etat][a], 2)).rjust(9)
    print(ligne)
print()

total = 0
for _ in range(1000):
    etat = jeu.reset()
    fini = False
    while not fini:
        action = meilleure_action(etat)
        etat, recompense, fini = jeu.step(action)
        total = total + recompense
print("Score moyen par partie (test) :", total / 1000)
