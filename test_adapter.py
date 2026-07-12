from environments.secret_envs_wrapper import SecretEnv0
from environments.secret_env_adapter import SecretEnvAdapter
import random
env = SecretEnvAdapter(SecretEnv0())

state = env.reset()
print("État initial :", state)
print("Actions valides :", env.available_actions())

#TEST quelques pas au hasard parmi les actions valides
for i in range(5):
    if env.env.is_game_over():
        print("Partie terminée.")
        break
    action = random.choice(env.available_actions())
    next_state, reward, done = env.step(action)
    print(f"Pas {i+1} : action {action} -> état {next_state}, reward {reward}, done {done}")