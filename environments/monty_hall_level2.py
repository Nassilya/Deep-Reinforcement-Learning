import random


class MontyHallLevel2:
    """
    Monty Hall niveau 2 : 5 portes, l'agent effectue 4 actions successives
    Une porte gagnante est tirée au hasard (cachée pour l'agent)

    Déroulé d'une partie :
      Action 1 : l'agent choisit une porte parmi les 5 (actions 0 à 4)
                 -> le présentateur retire une porte perdante non choisie (reste 4)
      Action 2 : garder (0) ou changer (1) -> une porte perdante est retirée (reste 3)
      Action 3 : garder (0) ou changer (1) -> une porte perdante est retirée (reste 2)
      Action 4 : garder (0) ou changer (1) -> la porte choisie est ouverte
    Quand l'agent 'change' et qu'il reste plusieurs autres portes,
    il bascule sur une des autres portes restantes au hasard.
    Reward final : 1.0 si la porte ouverte est gagnante, 0.0 sinon

    Les portes sont symétriques, mais la probabilité de gagner dépend de
    l'HISTORIQUE des décisions garder/changer. Les états codent donc cet historique :
      0             = début (choisir une porte parmi 5)
      1             = après le choix initial (4 portes restantes)
      2 + d2        = après la 2e action (3 portes restantes), d2 = garder(0)/changer(1)
      4 + 2*d2 + d3 = après la 3e action (2 portes restantes)
      8             = terminal (la porte est ouverte)
    """

    def __init__(self):
        self.nb_portes = 5
        self.noms_portes = ["A", "B", "C", "D", "E"]
        self.states = list(range(9))
        self.terminal_states = [8]
        self.porte_gagnante = None
        self.porte_choisie = None
        self.portes_restantes = None
        self.decisions = None          # historique des décisions garder/changer
        self.state = None

    def reset(self):
        self.porte_gagnante = random.randrange(self.nb_portes)
        self.porte_choisie = None
        self.portes_restantes = list(range(self.nb_portes))
        self.decisions = []
        self.state = 0
        return self.state

    def is_terminal(self, state):
        return state in self.terminal_states

    def available_actions(self):
        if self.state == 0:
            return [0, 1, 2, 3, 4]     # choisir une des 5 portes
        return [0, 1]                  # 0 = garder, 1 = changer

    def _retirer_porte_perdante(self):
        # Le présentateur retire une porte perdante et non choisie
        candidates = [p for p in self.portes_restantes
                      if p != self.porte_choisie and p != self.porte_gagnante]
        retiree = random.choice(candidates)
        self.portes_restantes.remove(retiree)

    def step(self, action):
        if action not in self.available_actions():
            raise Exception("Action invalide dans cet état !")

        if self.state == 0:
            # Action 1 : choix initial de la porte
            self.porte_choisie = action
            self._retirer_porte_perdante()
            self.state = 1
            return self.state, 0.0, False

        # Actions 2 à 4 : garder (0) ou changer (1)
        if action == 1:
            autres = [p for p in self.portes_restantes if p != self.porte_choisie]
            self.porte_choisie = random.choice(autres)
        self.decisions.append(action)

        if len(self.decisions) < 3:
            # il reste encore des décisions à prendre : une porte perdante est retirée
            self._retirer_porte_perdante()
            d = self.decisions
            if len(d) == 1:
                self.state = 2 + d[0]
            else:
                self.state = 4 + 2 * d[0] + d[1]
            return self.state, 0.0, False

        # dernière décision : on ouvre la porte choisie
        self.state = 8
        reward = 1.0 if self.porte_choisie == self.porte_gagnante else 0.0
        return self.state, reward, True

    def render(self):
        ligne = ""
        for p in range(self.nb_portes):
            if p not in self.portes_restantes:
                ligne += f"[X {self.noms_portes[p]}]"          # porte retirée
            elif p == self.porte_choisie:
                ligne += f"[A {self.noms_portes[p]}]"          # porte choisie par l'agent
            else:
                ligne += f"[? {self.noms_portes[p]}]"          # porte fermée
        print(ligne)
        if self.state == 8:
            resultat = "GAGNÉ" if self.porte_choisie == self.porte_gagnante else "PERDU"
            print(f"Porte gagnante : {self.noms_portes[self.porte_gagnante]} -> {resultat}")
