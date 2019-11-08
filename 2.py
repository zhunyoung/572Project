import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
plt.rcParams["figure.figsize"] = (20,10)
import sys


# read csv file into Pandas DataFrame
def file2df(path):
    df = pd.read_csv(path, delimiter=',', names=list(range(31)))
    # Fill NaN data points
    df = df.interpolate(axis=1, limit=60, limit_direction='both')
    # reverse the order of the columns since the timestamp for the data is descending
    df = df.iloc[:, ::-1]
    # remove NaN
    return df.dropna()

# read a list of csv files into Pandas DataFrame
def files2df(pathList):
    df = file2df(pathList[0])
    for path in pathList[1:]:
        df = pd.concat([df, file2df(path)], axis=0)
    return df.reset_index(drop=True)

# Feature extraction for DataFrame
def featureExtraction(df):
    ###################
    #### Feature 1 ####
    ###################
    feature1 = pd.DataFrame(columns=["4", "3", "2", "1", "0"])

    ###################
    #### Feature 2 ####
    ###################
    feature2 = pd.DataFrame(columns=["4", "3", "2", "1", "0"])

    ###################
    #### Feature 3 #### 
    ###################
    # Create feature matrix for polynomial regression with degree 4
    feature3 = pd.DataFrame(columns=["4", "3", "2", "1", "0"])
    x = list(range(31))
    # y is each row of the df
    for rowIdx, row in df.iterrows():
        y = row.values
        rowFeature = np.polyfit(x, y, 4)
        feature3 = feature3.append(pd.Series(rowFeature, index=feature3.columns), ignore_index=True)

    ###################
    #### Feature 4 ####
    ###################
    feature4 = pd.DataFrame(columns=["4", "3", "2", "1", "0"])


    return pd.concat([feature1, feature2, feature3, feature4], axis=1).fillna(0)

# Save DataFrame into path
def df2path(df, path):
    df.to_csv(path, encoding='utf-8', index=False)

# PATH
subjects = ['1', '2', '3', '4', '5']
categories = ['Meal', 'Nomeal']
pathList = {}
pathList['Meal'] = ['MealNoMealData/mealData'+i+'.csv' for i in subjects]
pathList['NoMeal'] = ['MealNoMealData/Nomeal'+i+'.csv' for i in subjects]

# Extract features from the data for each catogery
CGMSeries = {}
for c in categories:
    CGMSeries[c] = files2df(pathList[c])
    CGMSeries[c] = featureExtraction(CGMSeries[c])
    # store the features in a csv file
    df2path(CGMSeries[c], c+'.csv')
    print(CGMSeries[c])
