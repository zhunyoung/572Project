# import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.mlab import PCA
from sklearn.decomposition import PCA
from scipy.fftpack import fft

plt.rcParams["figure.figsize"] = (20, 10)
import sys

# The parent path
path = "./DataFolder/"
# The number of time slots and events
# Each subject has a unique number of events and time slots
list_shapes = []
# The file path and subject
list_file_subjects = ["1.csv", "2.csv", "3.csv", "4.csv", "5.csv"]
file_CGM_Datenum = "CGMDatenumLunchPat"
file_CGM_Series = "CGMSeriesLunchPat"
file_Insulin_Basal = "InsulinBasalLunchPat"
file_Insulin_Bolus = "InsulinBolusLunchPat"
file_Insulin_Datenum = "InsulinDatenumLunchPat"

# Load CGMSeries data for each subject
list_df_CGM_Series = []
for file_subject in list_file_subjects:
    df_CGM_Series = pd.read_csv(path + file_CGM_Series + file_subject)
    list_df_CGM_Series.append(df_CGM_Series)
    list_shapes.append(df_CGM_Series.shape)

# Need to apply the data to interploation
for i in range(len(list_file_subjects)):
    # Linear Interpolation
    list_df_CGM_Series[i] = list_df_CGM_Series[i].interpolate(axis=1, limit=60, limit_direction='both')
    # reverse the order of the columns since the timestamp for the data is descending
    list_df_CGM_Series[i] = list_df_CGM_Series[i][list_df_CGM_Series[i].columns[::-1]]

# print(list_df_CGM_Series[0])
# sys.exit()

# FEATURE 1 FROM JIAN
list_df_CGM_Series_feature1 = []
for file_subject in list_file_subjects:
    df_CGM_Series = pd.read_csv(path + file_CGM_Series + file_subject)
    list_df_CGM_Series_feature1.append(df_CGM_Series)

# Need to apply the data to interploation
for i in range(len(list_file_subjects)):
    # Linear Interpolation
    list_df_CGM_Series_feature1[i] = list_df_CGM_Series_feature1[i].interpolate(axis=1, limit=60,
                                                                                limit_direction='both')

list_df_Time_Series = []
for time_subjects in list_file_subjects:
    df_Time_Series = pd.read_csv(path + file_CGM_Datenum + time_subjects)
    list_df_Time_Series.append(df_Time_Series)

# Need to apply the data to interploation
for i in range(len(list_file_subjects)):
    # Linear Interpolation
    list_df_Time_Series[i] = list_df_Time_Series[i].interpolate(axis=1, limit=60,
                                                                                limit_direction='both')
list_df_Zerocrossing_feature1 = []
for i in range(len(list_file_subjects)):
    list_df_Zerocrossing_feature1.append(pd.DataFrame(columns=["Zero crossing time"]))

for i in range(0, 5):
    numrows = list_shapes[i][0]
    for j in range(0, numrows):
        if (i == 1 and j == 16):
            list_df_Zerocrossing_feature1[i] = list_df_Zerocrossing_feature1[i].append(
                pd.Series([0], index=list_df_Zerocrossing_feature1[i].columns), ignore_index=True)
            # print(i,j)
            continue
        elif (i == 2 and j == 0):
            list_df_Zerocrossing_feature1[i] = list_df_Zerocrossing_feature1[i].append(
                pd.Series([0], index=list_df_Zerocrossing_feature1[i].columns), ignore_index=True)
            # print(i,j)
            continue
        elif (i == 2 and j == 9):
            list_df_Zerocrossing_feature1[i] = list_df_Zerocrossing_feature1[i].append(
                pd.Series([0], index=list_df_Zerocrossing_feature1[i].columns), ignore_index=True)
            # print(i,j)
            continue
        elif (i == 2 and j == 24):
            list_df_Zerocrossing_feature1[i] = list_df_Zerocrossing_feature1[i].append(
                pd.Series([0], index=list_df_Zerocrossing_feature1[i].columns), ignore_index=True)
            # print(i,j)
            continue
        elif (i == 2 and j == 72):
            list_df_Zerocrossing_feature1[i] = list_df_Zerocrossing_feature1[i].append(
                pd.Series([0], index=list_df_Zerocrossing_feature1[i].columns), ignore_index=True)
            # print(i,j)
            continue
        for k in range(0, 31):
            if ((list_df_CGM_Series_feature1[i].iat[j, k] - list_df_CGM_Series_feature1[i].iat[j, k + 1]) * (
                    list_df_CGM_Series_feature1[i].iat[j, k + 1] - list_df_CGM_Series_feature1[i].iat[j, k + 2]) < 0):
                list_df_Zerocrossing_feature1[i] = list_df_Zerocrossing_feature1[i].append(
                    pd.Series([list_df_Time_Series[i].iat[j, k] - list_df_Time_Series[i].iat[j, 30]],
                              index=list_df_Zerocrossing_feature1[i].columns), ignore_index=True)
                # print(i,j,k)
                break

            elif ((list_df_CGM_Series_feature1[i].iat[j, k] - list_df_CGM_Series_feature1[i].iat[j, k + 1]) * (
                    list_df_CGM_Series_feature1[i].iat[j, k + 2] - list_df_CGM_Series_feature1[i].iat[j, k + 3]) <= 0):
                list_df_Zerocrossing_feature1[i] = list_df_Zerocrossing_feature1[i].append(
                    pd.Series([list_df_Time_Series[i].iat[j, k] - list_df_Time_Series[i].iat[j, 30]],
                              index=list_df_Zerocrossing_feature1[i].columns), ignore_index=True)
                # print(i,j,k)
                break
for i in range(len(list_file_subjects)):
    print('Feature 1:\n===============\n')
    print(list_df_Zerocrossing_feature1[i], '\n\n\n')

# FEATURE 2 FROM LILI
# Feature 2 -- Top8 FFT values.
# set sampling points -- 20
n = 8  # Top 8 FFT values
list_df_FFT_feature2 = []
for i in range(len(list_file_subjects)):
    list_df_FFT_feature2.append(pd.DataFrame(columns=["FFT 1", "2", "3", "4", "5", "6", "7", "8"]))

for i in range(len(list_file_subjects)):
    tempt = list_df_CGM_Series[i]

    for j in range(list_shapes[i][0]):
        tempt_event = tempt.loc[j].values
        tempt_fft = abs(fft(tempt_event)) / len(tempt_event)
        if pd.isnull(tempt_fft).any():
            T = [0] * 8
        else:
            T = tempt_fft[range(1, n + 1)] / sum(tempt_fft[range(1, n + 1)])
        list_df_FFT_feature2[i] = list_df_FFT_feature2[i].append(pd.Series((T), index=list_df_FFT_feature2[i].columns),
                                                                 ignore_index=True)

print(list_df_FFT_feature2[0])
for i in range(len(list_file_subjects)):
    print('Feature 2:\n===============\n')
    print(list_df_FFT_feature2[i], '\n\n\n')

# FEATURE 3 FROM ZHUN
# Create feature matrix for polynomial regression with degree 4
list_df_feature3 = []
for i in range(len(list_file_subjects)):
    list_df_feature3.append(pd.DataFrame(columns=["degree 4", "3", "2", "1", "0"]))
    x = list(range(list_shapes[i][1]))
    for j in range(list_shapes[i][0]):
        y = list_df_CGM_Series[i].values[j]
        if pd.isnull(y).any():
            feature = [0] * 5
        else:
            feature = np.polyfit(x, y, 4)
        list_df_feature3[i] = list_df_feature3[i].append(pd.Series(feature, index=list_df_feature3[i].columns),
                                                         ignore_index=True)


# for i in range(len(list_file_subjects)):
#     print('Feature 3:\n===============\n')
#     print(list_df_feature3[i], '\n\n\n')
#     result = PCA(np.array(list_df_feature3[i].values, dtype=np.float64))
#     print(result.Wt, '\n')


# FEATURE 4 FROM JINYUNG
# define the function for TIR that will be used in apply function
def func_TIR(value):
    if value < 54:
        return 1
    elif value >= 54 and value < 70:
        return 2
    elif value >= 70 and value < 180:
        return 3
    elif value >= 180 and value < 250:
        return 4
    else:
        return 5


# Create intermediate feature matrix for feature 4
list_df_inter_feature4 = []
for i in range(len(list_file_subjects)):
    list_df_inter_feature4.append(pd.DataFrame())

# Apply func_TIR to dataset
for i in range(len(list_file_subjects)):
    num_rows = list_shapes[i][0]
    for j in range(num_rows):
        list_df_inter_feature4[i] = list_df_inter_feature4[i].append(list_df_CGM_Series[i].iloc[j].apply(func_TIR))

# Create feature matrix
list_df_feature4 = []
for i in range(len(list_file_subjects)):
    list_df_feature4.append(pd.DataFrame(columns=["x<54", "54<=x<70", "70<=x<180", "180<=x<250", "250<=x"]))

for i in range(len(list_file_subjects)):
    num_rows = list_shapes[i][0]
    for j in range(num_rows):
        num_cols = list_shapes[i][1]
        ratio_cat1 = (list_df_inter_feature4[i].iloc[j] == 1).sum() / num_cols
        ratio_cat2 = (list_df_inter_feature4[i].iloc[j] == 2).sum() / num_cols
        ratio_cat3 = (list_df_inter_feature4[i].iloc[j] == 3).sum() / num_cols
        ratio_cat4 = (list_df_inter_feature4[i].iloc[j] == 4).sum() / num_cols
        ratio_cat5 = (list_df_inter_feature4[i].iloc[j] == 5).sum() / num_cols
        list_df_feature4[i] = list_df_feature4[i].append(
            pd.Series([ratio_cat1, ratio_cat2, ratio_cat3, ratio_cat4, ratio_cat5], index=list_df_feature4[i].columns),
            ignore_index=True)

# for i in range(len(list_file_subjects)):
#     print('Feature 4:\n===============\n')
#     print(list_df_feature4[i], '\n\n\n')
# result = PCA(np.array(list_df_feature4[i].values, dtype=np.float64))
# print(result.Wt, '\n')

## Merge all features into one feature matrix.
list_df_feature_matrices = []
for i in range(len(list_file_subjects)):
    df_feature_matrix = pd.concat(
        [list_df_Zerocrossing_feature1[i], list_df_FFT_feature2[i], list_df_feature3[i], list_df_feature4[i]],
        axis=1).fillna(0)
    list_df_feature_matrices.append(df_feature_matrix)
for i in range(len(list_file_subjects)):
    print('Feature ALL:\n===============\n')
    print(list_df_feature_matrices[i], '\n\n\n')
    pca = PCA(n_components=5, whiten=True)
    # pca.fit(np.array(list_df_feature_matrices[i].values))
    newdata = pca.fit_transform(np.array(list_df_feature_matrices[i].values))
    # print(pca)
    print(newdata)
    print(newdata.shape)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    # result = PCA(np.array(list_df_feature_matrices[i].values, dtype=np.float64, standardize=False))
    # print(result.Wt, '\n')
