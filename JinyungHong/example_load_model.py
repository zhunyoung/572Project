import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from load_model import predict

## Load test data
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

data_meal = pd_data_meal.interpolate(axis=1, limit=60, limit_direction='both').fillna(method="ffill").to_numpy()
data_nomeal = pd_data_nomeal.interpolate(axis=1, limit=60, limit_direction='both').fillna(method="ffill").to_numpy()

X = np.concatenate([data_meal, data_nomeal], axis=0)
Y = np.concatenate([np.ones((data_meal.shape[0], 1)), np.zeros((data_nomeal.shape[0], 1))], axis=0)

## Use predict and it will return prediction
preds = predict(X)
print(accuracy_score(Y, preds))
print(f1_score(Y, preds))