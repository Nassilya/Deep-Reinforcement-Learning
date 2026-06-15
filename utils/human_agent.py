def play_human(env):
    """Permet à un humain de jouer pour vérifier les règles de l'environnement"""
    state = env.reset()
    done = False
    total_reward = 0.0

    print("=== Mode humain : 0 = gauche, 1 = droite, q = quitter ===")
    env.render()

    while not done:
        choice = input("Action (0/1) ? ")
        if choice == "q":
            break
        if choice not in ("0", "1"):
            print("Entrée invalide.")
            continue

        action = int(choice)
        state, reward, done = env.step(action)
        total_reward += reward
        env.render()
        print(f"-> état={state}, reward={reward}, terminé={done}")

    print(f"Partie terminée. Récompense totale : {total_reward}")