from environments.line_world import LineWorld
from environments.grid_world import GridWorld
from environments.monty_hall_level1 import MontyHallLevel1
from environments.monty_hall_level2 import MontyHallLevel2
from algorithms.dynamic_programming import policy_iteration, value_iteration
from algorithms.temporal_difference import sarsa, q_learning
from algorithms.monte_carlo import (monte_carlo_es, on_policy_mc_control,
                                    off_policy_mc_control, evaluer_politique)
from utils.human_agent import play_human, play_human_general
from utils.persistence import save, load
from utils.plots import tracer_apprentissage, tracer_convergence, comparer_apprentissage
from utils.plots import comparer_cumuls

def afficher_politique(env, policy, titre):
    print(f"\n=== {titre} ===")
    for s in env.states:
        if env.is_terminal(s):
            print(f"  état {s} : terminal")
        else:
            print(f"  état {s} : action {policy[s]}")


def tester_politique(env, policy, max_steps=50):
    state = env.reset()
    print("État initial :")
    env.render()
    total = 0.0
    etape = 0
    for _ in range(max_steps):
        etape += 1
        action = policy[state]
        state, reward, done = env.step(action)
        total += reward
        print(f"--- Étape {etape} : action {action}, reward {reward} ---")
        env.render()
        if done:
            break
    print(f"Récompense totale : {total}")


def derouler_politique(env, policy, max_steps=20):
    """Déroulé pas à pas d'une politique apprise (pour la soutenance), pour
    les environnements où les actions disponibles dépendent de l'état."""
    state = env.reset()
    print("État initial :")
    env.render()
    total = 0.0
    for etape in range(1, max_steps + 1):
        actions = env.available_actions()
        action = policy.get(state, actions[0])
        if action not in actions:
            action = actions[0]
        state, reward, done = env.step(action)
        total += reward
        print(f"--- Étape {etape} : action {action}, reward {reward} ---")
        env.render()
        if done:
            break
    print(f"Récompense totale : {total}")


def entrainer_et_evaluer_mc(env, nom_env, episodes=20000, gamma=0.99, epsilon=0.1, explore_max=10):
    """Lance les 3 algorithmes Monte Carlo sur un environnement :
    PHASE 1 : entraînement (politique exploratrice)
    PHASE 2 : évaluation de la politique apprise (gloutonne, sans exploration)"""
    resultats = {}

    algos = {
        "MC_ES": lambda: monte_carlo_es(env, episodes=episodes, gamma=gamma, explore_max=explore_max),
        "MC_on_policy": lambda: on_policy_mc_control(env, episodes=episodes, gamma=gamma, epsilon=epsilon),
        "MC_off_policy": lambda: off_policy_mc_control(env, episodes=episodes, gamma=gamma, epsilon=epsilon),
    }

    for nom_algo, lancer in algos.items():
        # ---------- PHASE D'ENTRAÎNEMENT ----------
        policy, Q, cumul = lancer()
        afficher_politique(env, policy, f"{nom_env} - {nom_algo}")
        save(Q, f"{nom_env}_{nom_algo}_Q.pkl".lower())
        save(policy, f"{nom_env}_{nom_algo}_policy.pkl".lower())

        # ---------- PHASE D'ÉVALUATION ----------
        gain_moyen = evaluer_politique(env, policy, episodes=10000)
        print(f"  -> Évaluation (10000 épisodes, sans exploration) : gain moyen = {gain_moyen:.3f}")

        resultats[nom_algo] = {"policy": policy, "Q": Q, "cumul": cumul, "gain": gain_moyen}

    return resultats


if __name__ == "__main__":

    # ----------------------------------------------------------- LINE WORLD  
    print("---------- LINE WORLD ---------- ")
    env_line = LineWorld(size=5)
    policy, V, deltas_pi = policy_iteration(env_line, gamma=0.99)

    print("Valeurs apprises :")
    for s in env_line.states:
        print(f"  état {s} : V = {V[s]:.3f}")

    print("\nPolitique optimale (0=gauche, 1=droite) :")
    for s in env_line.states:
        if env_line.is_terminal(s):
            print(f"  état {s} : terminal")
        else:
            action_str = "droite" if policy[s] == 1 else "gauche"
            print(f"  état {s} : {action_str}")

    save(policy, "lineworld_policy_iteration.pkl")
    save(V, "lineworld_policy_iteration_V.pkl")
    policy_chargee = load("lineworld_policy_iteration.pkl")
    print("Politique rechargée :", policy_chargee)

    policy_vi, V_vi, deltas_vi = value_iteration(env_line, gamma=0.99)
    print("Value Iteration - politique :", policy_vi)
    print("Value Iteration - valeurs :", V_vi)
    save(policy_vi, "lineworld_value_iteration.pkl")
    save(V_vi, "lineworld_value_iteration_V.pkl")


    policy_sarsa_line, Q_sarsa_line, rewards_sarsa_line = sarsa(env_line, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1)
    afficher_politique(env_line, policy_sarsa_line, "Line World - Sarsa")
    save(Q_sarsa_line, "lineworld_sarsa_Q.pkl")

    policy_ql_line, Q_ql_line, rewards_ql_line = q_learning(env_line, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1)
    afficher_politique(env_line, policy_ql_line, "Line World - Q-Learning")
    save(Q_ql_line, "lineworld_qlearning_Q.pkl")

    # -------------------------------------------------------------------- GRID WORLD  
    print("\n\n---------- GRID WORLD ---------- ")
    env = GridWorld(rows=5, cols=5)          

    policy_pi, V_pi, deltas_pi_grid = policy_iteration(env, gamma=0.99)
    afficher_politique(env, policy_pi, "Grid World - Policy Iteration")
    save(policy_pi, "gridworld_policy_iteration.pkl")
    save(V_pi, "gridworld_policy_iteration_V.pkl")

    policy_vi_grid, V_vi_grid, deltas_vi_grid = value_iteration(env, gamma=0.99)
    afficher_politique(env, policy_vi_grid, "Grid World - Value Iteration")
    save(policy_vi_grid, "gridworld_value_iteration.pkl")
    save(V_vi_grid, "gridworld_value_iteration_V.pkl")

#Sarsa converge dès 5000 épisodes avec epsilon=0.1 alors que Q Learning nécessite 20000 épisodes et epsilon=0.3 pour éviter de rester bloqué
    policy_sarsa, Q_sarsa, cumul_sarsa = sarsa(env, episodes=20000, alpha=0.1, gamma=0.99, epsilon=0.1)
    afficher_politique(env, policy_sarsa, "Grid World - Sarsa")
    save(Q_sarsa, "gridworld_sarsa_Q.pkl")

    policy_ql, Q_ql, cumul_ql = q_learning(env, episodes=20000, alpha=0.1, gamma=0.99, epsilon=0.3)
    afficher_politique(env, policy_ql, "Grid World - Q-Learning")
    save(Q_ql, "gridworld_qlearning_Q.pkl")

    # --------------------------------------------------------------------- Déroulé des politiques 
    print("\n\n======== TEST DES POLITIQUES SUR LINE WORLD ========")

    print("\n--- Déroulé Policy Iteration (Line World) ---")
    tester_politique(env_line, policy)

    print("\n--- Déroulé Value Iteration (Line World) ---")
    tester_politique(env_line, policy_vi)

    print("\n--- Déroulé Sarsa (Line World) ---")
    tester_politique(env_line, policy_sarsa_line)

    print("\n--- Déroulé Q-Learning (Line World) ---")
    tester_politique(env_line, policy_ql_line)

    print("\n\n======== TEST DES POLITIQUES SUR GRID WORLD ========")

    print("\n--- Déroulé Policy Iteration (Grid World) ---")
    tester_politique(env, policy_pi)

    print("\n--- Déroulé Value Iteration (Grid World) ---")
    tester_politique(env, policy_vi_grid)

    print("\n--- Déroulé Sarsa (Grid World) ---")
    tester_politique(env, policy_sarsa)

    print("\n--- Déroulé Q-Learning (Grid World) ---")
    tester_politique(env, policy_ql)

#Sarsa vs Q-Learning sur Grid World
comparer_cumuls(
    {"Sarsa": cumul_sarsa, "Q-Learning": cumul_ql},
    titre="Grid World : cumul des récompenses (Sarsa vs Q-Learning)"
)


tracer_convergence(deltas_vi_grid, titre="Grid World : convergence de Value Iteration")


# ==================================================================== MONTE CARLO
# Partie Monte Carlo : MC ES, On-policy first visit, Off-policy
# sur les 4 environnements (dont Monty Hall level 1 et 2)

print("\n\n---------- MONTY HALL LEVEL 1 (Monte Carlo) ---------- ")
# Résultat attendu : à l'état 1, la politique doit CHANGER de porte (action 1)
# -> gain moyen ~ 0.667 (contre 0.333 si on garde)
env_mh1 = MontyHallLevel1()
res_mh1 = entrainer_et_evaluer_mc(env_mh1, "MontyHall1", episodes=20000, explore_max=2)

print("\n\n---------- MONTY HALL LEVEL 2 (Monte Carlo) ---------- ")
# Résultat attendu : garder, garder, puis CHANGER à la dernière décision
# -> gain moyen ~ 0.8 (la porte initiale ne gagne qu'avec proba 1/5)
env_mh2 = MontyHallLevel2()
res_mh2 = entrainer_et_evaluer_mc(env_mh2, "MontyHall2", episodes=50000, explore_max=4)

print("\n\n---------- LINE WORLD (Monte Carlo) ---------- ")
res_line_mc = entrainer_et_evaluer_mc(env_line, "LineWorld", episodes=5000, explore_max=4)

print("\n\n---------- GRID WORLD (Monte Carlo) ---------- ")
res_grid_mc = entrainer_et_evaluer_mc(env, "GridWorld", episodes=50000, explore_max=30)

# --------------------------------------------------- Déroulé pas à pas (soutenance)
print("\n\n======== DÉROULÉ DES POLITIQUES MONTE CARLO ========")

print("\n--- Déroulé MC ES (Monty Hall 1) ---")
derouler_politique(env_mh1, res_mh1["MC_ES"]["policy"])

print("\n--- Déroulé MC on-policy (Monty Hall 2) ---")
derouler_politique(env_mh2, res_mh2["MC_on_policy"]["policy"])

# --------------------------------------------------- Courbes d'apprentissage
comparer_cumuls(
    {nom: res["cumul"] for nom, res in res_mh1.items()},
    titre="Monty Hall 1 : cumul des récompenses (Monte Carlo)",
    fichier="graphe_mc_montyhall1.png"
)
comparer_cumuls(
    {nom: res["cumul"] for nom, res in res_mh2.items()},
    titre="Monty Hall 2 : cumul des récompenses (Monte Carlo)",
    fichier="graphe_mc_montyhall2.png"
)
comparer_cumuls(
    {nom: res["cumul"] for nom, res in res_grid_mc.items()},
    titre="Grid World : cumul des récompenses (Monte Carlo)",
    fichier="graphe_mc_gridworld.png"
)

# --------------------------------------------------- Agent humain (décommenter)
# print("\n\n======== JOUER À LA MAIN ========")
# play_human_general(MontyHallLevel1())
# play_human_general(MontyHallLevel2())