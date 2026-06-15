import pickle
import os


def save(data, filename, folder="saved_models"):
    """
    Sauvegarde n'importe quel objet Python (politique, V, Q...) dans un fichier

    data     : l'objet à sauvegarder (dict de politique, dict de valeurs...)
    filename : nom du fichier
    folder   : dossier de destination (créé s'il n'existe pas)
    """
    os.makedirs(folder, exist_ok=True)            
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:                   #write binary
        pickle.dump(data, f)
    print(f"Sauvegardé dans : {path}")
    return path


def load(filename, folder="saved_models"):
    """
    Charge un objet précédemment sauvegardé

    filename : nom du fichier à charger
    folder   : dossier où chercher
    Renvoie l'objet Python tel qu'il était au moment de la sauvegarde
    """
    path = os.path.join(folder, filename)
    with open(path, "rb") as f:                   #read binary
        data = pickle.load(f)
    print(f"Chargé depuis : {path}")
    return data