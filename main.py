
from environments.line_world import LineWorld
from algorithms.dynamic_programming import policy_iteration
from utils.human_agent import play_human
from utils.persistence import save, load
from algorithms.dynamic_programming import value_iteration


if __name__ == "__main__":
    env = LineWorld(size=5)

    
    policy, V = policy_iteration(env, gamma=0.99)

    print("Valeurs apprises :")
    for s in env.states:
        print(f"  état {s} : V = {V[s]:.3f}")

    print("\nPolitique optimale (0=gauche, 1=droite) :")
    for s in env.states:
        if env.is_terminal(s):
            print(f"  état {s} : terminal")
        else:
            action_str = "droite" if policy[s] == 1 else "gauche"
            print(f"  état {s} : {action_str}")

#--------------------------------------------------------------Entraînement puis sauvegarde
policy, V = policy_iteration(env, gamma=0.99)
save(policy, "lineworld_policy_iteration.pkl")
save(V, "lineworld_policy_iteration_V.pkl")

#-------------------------------------------------------------Recharger sans réentraîner
policy_chargee = load("lineworld_policy_iteration.pkl")
V_chargee = load("lineworld_policy_iteration_V.pkl")

print("Politique rechargée :", policy_chargee)

policy_vi, V_vi = value_iteration(env, gamma=0.99)
print("Value Iteration - politique :", policy_vi)
print("Value Iteration - valeurs :", V_vi)