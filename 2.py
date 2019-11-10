""" 
1. To test on the given dataset:

python 2.py

whose expected output is: (note that the best params may change for different runs)

    Best params for random forest: {'max_depth': 90, 'min_samples_leaf': 10, 'n_estimators': 30}
    We first use 80% of data to train the model and then predict on the remaining 20% of data...
    Accuracy on validation data (20% of all data) after hypertuning for Random Forest:0.626263

2. To make prediction on a given file, e.g., MealNoMealData/mealData1.csv

python 2.py -file MealNoMealData/mealData1.csv

whose expected output is:

    Best params for random forest: {'max_depth': 60, 'min_samples_leaf': 10, 'n_estimators': 20}
    We first use 80% of data to train the model and then predict on the remaining 20% of data...
    Accuracy on validation data (20% of all data) after hypertuning for Random Forest:0.636364

    Then, let's predict the labels for the given file...
    [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 0. 0. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1.]

"""

import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

# apply PCA to features
def applyPCA(df, pca):
    return pca.transform(df)


def main():
    parser = argparse.ArgumentParser(description='CSE572 Project 2')
    parser.add_argument('-file', required=False, type=str, default='')
    args = parser.parse_args()

    # PATH
    subjects = ['1', '2', '3', '4', '5']
    categories = ['Meal', 'Nomeal']
    pathList = {}
    pathList['Meal'] = ['MealNoMealData/mealData'+i+'.csv' for i in subjects]
    pathList['Nomeal'] = ['MealNoMealData/Nomeal'+i+'.csv' for i in subjects]

    # Extract features from the data for each catogery
    CGMSeries = {}
    for c in categories:
        CGMSeries[c] = files2df(pathList[c])
        CGMSeries[c] = featureExtraction(CGMSeries[c])

    # Apply PCA
    n_components = 5
    pca = PCA(n_components=n_components, whiten=True)
    # fit PCA with only Meal data
    pca.fit(CGMSeries['Meal'])

    ##################
    # get all data for training and testing
    ##################

    # modifying features by applying pca and adding label
    CGMSeriesPCA = {}
    for c in categories:
        if c == 'Meal':
            CGMSeriesPCA[c] = np.ones((len(CGMSeries[c].index), n_components+1))
        elif c == 'Nomeal':
            CGMSeriesPCA[c] = np.zeros((len(CGMSeries[c].index), n_components+1))
        else:
            print('Error!')
        CGMSeriesPCA[c][:,:-1] = pca.transform(CGMSeries[c])

    # combine the 2 data with 2 labels, random the order
    allData = np.concatenate((CGMSeriesPCA['Meal'], CGMSeriesPCA['Nomeal']), axis=0)
    np.random.shuffle(allData)

    ##################
    # train a model: random forest
    ##################

    # Note that I don't split training and validation data since I will use 
    # sklearn 'grid search with cross validation' function to find the optimal hyper-parameters
    X_train = allData[:,:-1]
    y_train = allData[:,-1]

    rand_parameters={'n_estimators': [10,20,30], 'min_samples_leaf': range(5,30,5), 'max_depth': range(50,100,10)}
    rfc = RandomForestClassifier()
    rfc_grid = GridSearchCV(rfc, rand_parameters, cv=5)
    rfc_grid.fit(X_train, y_train)

    best_params = rfc_grid.best_params_
    print('Best params for random forest: {}'.format(best_params))
    rfc = RandomForestClassifier(**best_params)

    # we use 80% of the data to train again and use 20% of the data for testing
    print('We first use 80% of data to train the model and then predict on the remaining 20% of data...')
    idx = int(0.8*len(allData))
    rfc.fit(X_train[:idx,:], y_train[:idx])
    # y_predict = rfc.predict(X_train[idx:,:])
    print('Accuracy on validation data (20% of all data) after hypertuning for Random Forest:{0:6f}'.format(rfc.score(X_train[idx:,:],y_train[idx:])))

    # if input file is given, we use all the data for training and predict on the csv file
    if args.file != '':
        print('\nThen, let\'s predict the labels for the given file...')
        rfc.fit(X_train, y_train)
        try:
            df = file2df(args.file)
        except:
            print('Error! The input file {} cannot be read as a csv file'.format(args.file))
            sys.exit()
        df = featureExtraction(df)
        X_test = pca.transform(df)
        y_predict = rfc.predict(X_test)
        print(y_predict)
  
if __name__== "__main__":
    main()
