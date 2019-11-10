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
    feature2 = pd.DataFrame(columns=["8", "7", "6", "5", "4", "3", "2", "1"])
    i2 = 0
    while i2 < len(df):
        tempt2 = df.iloc[i2].values
        tempt2_fft = np.abs(fft(tempt2))/len(tempt2)
        T2 = tempt2_fft[range(1,9)]/sum(tempt2_fft[range(1,9)])
        feature2 = feature2.append(pd.Series((T),index=feature2.columns),ignore_index=True)

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

# PCA
def zeroMean(dataMat):
    # remove the mean value for each column.
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    return newData, meanVal

def pca(dataMat,n):
    # pca calculation, n is usually 5 in our project.
    newData,mealVal = zeroMean(dataMat)
    covMat = np.cov(newData,rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat)) # get eigen values & eigen vectors
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n+1):-1]
    #print(n_eigValIndice)
    n_eigVect = eigVects[:,n_eigValIndice]
    return n_eigVect # return top 5 feature vectors.

# KNN
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row,dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# KNN
def knn(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

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
