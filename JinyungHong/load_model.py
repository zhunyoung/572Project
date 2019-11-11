import numpy as np
from echo_state_network import Reservoir
import pickle


def load_model():
    path_reservoir = "reservoir.pkl"
    path_pca = "pca.pkl"
    path_ridge = "ridge.pkl"
    path_ridgeclf = "ridgeclf.pkl"

    with open(path_reservoir, "rb") as file_reservoir:
        model_reservoir = pickle.load(file_reservoir)
    with open(path_pca, "rb") as file_pca:
        model_pca = pickle.load(file_pca)
    with open(path_ridge, "rb") as file_ridge:
        model_ridge = pickle.load(file_ridge)
    with open(path_ridgeclf, "rb") as file_ridgeclf:
        model_ridgeclf = pickle.load(file_ridgeclf)
    return model_reservoir, model_pca, model_ridge, model_ridgeclf


def predict(X_test):
    X_test = np.expand_dims(X_test, axis=2)
    model_reservoir, model_pca, model_ridge, model_ridgeclf = load_model()
    res = Reservoir(dict_params=model_reservoir, fromPickle=True)

    input_rep_te = res.getReservoirEmbedding(X_test, model_pca, model_ridge, n_drop=5, bidir=True, test=True)
    preds = model_ridgeclf.predict(input_rep_te)

    return preds


