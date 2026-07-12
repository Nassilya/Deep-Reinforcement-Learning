#from environments.secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3
from environments.secret_envs_wrapper import SecretEnv0

#for cls in [SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3]:
#    env = cls()
#    print(f"{cls.__name__} : "
#         f"{env.num_states()} états, "
#         f"{env.num_actions()} actions, "
#         f"{env.num_rewards()} récompenses")


env = SecretEnv0()
env.reset()
print("État initial (state_id) :", env.state_id())
print("Actions disponibles :", env.available_actions())
print("Game over ?", env.is_game_over())
print("Score :", env.score())
env.display()