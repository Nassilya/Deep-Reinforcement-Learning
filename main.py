from environments.line_world import LineWorld
from environments.grid_world import GridWorld
from algorithms.dynamic_programming import policy_iteration, value_iteration
from algorithms.temporal_difference import sarsa, q_learning
from utils.human_agent import play_human
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