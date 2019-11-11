import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, average_precision_score
from echo_state_network import Reservoir
import pickle
import matplotlib.pyplot as plt
from inspect import signature


def save_model(model_reservoir, model_pca, model_ridge, model_ridgeclf):
    path_reservoir = "reservoir.pkl"
    path_pca = "pca.pkl"
    path_ridge = "ridge.pkl"
    path_ridgeclf = "ridgeclf.pkl"

    with open(path_reservoir, "wb") as file_reservoir:
        pickle.dump(model_reservoir, file_reservoir, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_pca, "wb") as file_pca:
        pickle.dump(model_pca, file_pca, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_ridge, "wb") as file_ridge:
        pickle.dump(model_ridge, file_ridge, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_ridgeclf, "wb") as file_ridgeclf:
        pickle.dump(model_ridgeclf, file_ridgeclf, protocol=pickle.HIGHEST_PROTOCOL)


## Load input data_train
path_dir = "./MealNoMealData/"
path_meal = ["mealData1.csv", "mealData2.csv", "mealData3.csv", "mealData4.csv", "mealData5.csv"]
path_nomeal = ["Nomeal1.csv", "Nomeal2.csv", "Nomeal3.csv", "Nomeal4.csv", "Nomeal5.csv"]
pd_data_meal = pd.DataFrame()
for path in path_meal:
    pd_data = pd.read_csv(path_dir+path, names=range(0, 31), engine="python", skip_blank_lines=True)
    pd_data_meal = pd.concat([pd_data_meal, pd_data], axis=0)

pd_data_nomeal = pd.DataFrame()
for path in path_nomeal:
    pd_data = pd.read_csv(path_dir+path, names=range(0, 31), engine="python", skip_blank_lines=True)
    pd_data_nomeal = pd.concat([pd_data_nomeal, pd_data], axis=0)

data_meal = pd_data_meal.interpolate(axis=1, limit=60, limit_direction='both').fillna(method="ffill")
data_nomeal = pd_data_nomeal.interpolate(axis=1, limit=60, limit_direction='both').fillna(method="ffill")

X = np.concatenate([data_meal, data_nomeal], axis=0)
Y = np.concatenate([np.ones((data_meal.shape[0], 1)), np.zeros((data_nomeal.shape[0], 1))], axis=0)

# 10-Fold Cross Validation
kf = KFold(10, True, 1)
list_scores = []
list_models = []
n = 0
for train_index, test_index in kf.split(X):
    print("{} Fold".format(n+1))
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # Preproecessing
    pca = PCA(n_components=200)
    # Original function of embedding and readout => Ridge
    ridge_embedding = Ridge(alpha=10, fit_intercept=True)
    readout = RidgeClassifier(alpha=5)
    res = Reservoir(n_internal_units=450, spectral_radius=0.6, leak=0.5,
                    connectivity=0.25, input_scaling=0.001, noise_level=0.01, circle=False)
    input_repr = res.getReservoirEmbedding(X_train, pca, ridge_embedding, n_drop=5, bidir=True, test=False)
    input_repr_te = res.getReservoirEmbedding(X_test, pca, ridge_embedding, n_drop=5, bidir=True, test=True)
    readout.fit(input_repr, Y_train)
    preds = readout.predict(input_repr_te)

    ## Performance
    # 1. Precision and Recall Curve
    Y_score = readout.decision_function(input_repr_te)
    average_precision = average_precision_score(Y_test, Y_score)
    precision, recall, _ = precision_recall_curve(Y_test, Y_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("2-class Precision-Recall curve of {0} fold : AP={1:0.2f}".format(n+1, average_precision))

    plt.show()

    # 2. Test Accuracy and F1 Scroe
    accuracy = accuracy_score(Y_test, preds)
    f1 = f1_score(Y_test, preds)
    print("Test Accuracy : {}".format(accuracy))
    print("F1 Score : {}".format(f1))

    # Models score : average of sum of test accuracy and f1 score will be chosen
    list_scores.append((accuracy + f1) / 2)
    ## Save Model
    dict_models = {}
    dict_models["reservoir"] = res.save()
    dict_models["pca"] = pca
    dict_models["ridge"] = ridge_embedding
    dict_models["ridgeclf"] = readout
    list_models.append(dict_models)

    n += 1

# Choose the model which has the best score
best_model_index = np.argmax(list_scores)
dict_best_model = list_models[best_model_index]
print("Final model's score : {}".format(list_scores[best_model_index]))

# Please, comment this out if you want to save the model
# save_model(model_reservoir=dict_best_model["reservoir"],
#            model_pca=dict_best_model["pca"],
#            model_ridge=dict_best_model["ridge"],
#            model_ridgeclf=dict_best_model["ridgeclf"])

