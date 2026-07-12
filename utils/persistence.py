import pickle
import os

def save(data, filename, folder="saved_models"):
    os.makedirs(folder, exist_ok=True)            
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:                  
        pickle.dump(data, f)
    print(f"Sauvegardé dans : {path}")
    return path


def load(filename, folder="saved_models"):
    path = os.path.join(folder, filename)
    with open(path, "rb") as f:                   
        data = pickle.load(f)
    print(f"Chargé depuis : {path}")
    return data