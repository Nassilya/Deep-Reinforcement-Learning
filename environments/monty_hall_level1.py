import random


class MontyHallLevel1:
    """
    Monty Hall niveau 1 : 3 portes (A, B, C)
    Une porte gagnante est tirée au hasard (cachée pour l'agent)
    Étape 1 : l'agent choisit une porte parmi les 3 (actions 0, 1, 2)
    Le présentateur retire ensuite une porte perdante parmi les 2 non choisies
    Étape 2 : l'agent garde sa porte (action 0) ou change pour la porte restante (action 1)
    Reward final : 1.0 si la porte ouverte est gagnante, 0.0 sinon

    Les portes sont symétriques (aucune n'est meilleure qu'une autre au départ),
    donc 3 états suffisent :
      0 = début (l'agent doit choisir une porte)
      1 = une porte a été retirée (l'agent doit garder ou changer)
      2 = terminal (la porte est ouverte)
    """

    def __init__(self):
        self.nb_portes = 3
        self.noms_portes = ["A", "B", "C"]
        self.states = [0, 1, 2]
        self.terminal_states = [2]
        self.porte_gagnante = None
        self.porte_choisie = None
        self.porte_retiree = None
        self.state = None

    def reset(self):
        self.porte_gagnante = random.randrange(self.nb_portes)
        self.porte_choisie = None
        self.porte_retiree = None
        self.state = 0
        return self.state

    def is_terminal(self, state):
        return state in self.terminal_states

    def available_actions(self):
        if self.state == 0:
            return [0, 1, 2]        # choisir la porte A, B ou C
        return [0, 1]               # 0 = garder, 1 = changer

    def step(self, action):
        if action not in self.available_actions():
            raise Exception("Action invalide dans cet état !")

        if self.state == 0:
            # Étape 1 : l'agent choisit une porte
            self.porte_choisie = action
            # Le présentateur retire une porte perdante non choisie
            candidates = [p for p in range(self.nb_portes)
                          if p != self.porte_choisie and p != self.porte_gagnante]
            self.porte_retiree = random.choice(candidates)
            self.state = 1
            return self.state, 0.0, False

        # Étape 2 : garder (0) ou changer (1)
        if action == 1:
            self.porte_choisie = [p for p in range(self.nb_portes)
                                  if p != self.porte_choisie and p != self.porte_retiree][0]
        self.state = 2
        reward = 1.0 if self.porte_choisie == self.porte_gagnante else 0.0
        return self.state, reward, True

    def render(self):
        ligne = ""
        for p in range(self.nb_portes):
            if p == self.porte_retiree:
                ligne += f"[X {self.noms_portes[p]}]"          # porte retirée
            elif p == self.porte_choisie:
                ligne += f"[A {self.noms_portes[p]}]"          # porte choisie par l'agent
            else:
                ligne += f"[? {self.noms_portes[p]}]"          # porte fermée
        print(ligne)
        if self.state == 2:
            resultat = "GAGNÉ" if self.porte_choisie == self.porte_gagnante else "PERDU"
            print(f"Porte gagnante : {self.noms_portes[self.porte_gagnante]} -> {resultat}")
