def play_human(env):
    state = env.reset()
    done = False
    total_reward = 0.0

    print("0 = gauche, 1 = droite, q = quitter")
    env.render()

    while not done:
        choice = input("Action (0/1) ? ")
        if choice == "q":
            break
        if choice not in ("0", "1"):
            print("Entrée invalide")
            continue

        action = int(choice)
        state, reward, done = env.step(action)
        total_reward += reward
        env.render()
        print(f"-> état={state}, reward={reward}, terminé={done}")

    print(f"Partie terminée ! Récompense totale : {total_reward}")


def play_human_general(env):
    """Agent humain pour les environnements où les actions disponibles
    changent selon l'état (Monty Hall...) : utilise available_actions()."""
    state = env.reset()
    done = False
    total_reward = 0.0

    print("q = quitter")
    env.render()

    while not done:
        actions = env.available_actions()
        choice = input(f"Action possible {actions} ? ")
        if choice == "q":
            break
        if not choice.isdigit() or int(choice) not in actions:
            print("Entrée invalide")
            continue

        state, reward, done = env.step(int(choice))
        total_reward += reward
        env.render()
        print(f"-> état={state}, reward={reward}, terminé={done}")

    print(f"Partie terminée ! Récompense totale : {total_reward}")