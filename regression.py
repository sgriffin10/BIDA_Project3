from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

df = pd.read_csv("US_Accidents_May19_cleaned_noNAs_1.csv")

# pd.options.display.float_format = "{:.2f}".format 
# print(df.describe())
# print(df.shape)
# print(df.dtypes)

#"Start_Time","Side","City","County","State","Zipcode","Timezone"
# variables_list = ["Start_Lat","Distance(mi)","Temperature(F)","Wind_Chill(F)"]

# for column in variables_list:
#     X = pd.DataFrame(df[column])
#     y = pd.DataFrame(df["Severity"])
#     model = LinearRegression()
#     scores = []
#     kfold = KFold(n_splits=3, shuffle=True, random_state=42)
#     for i, (train, test) in enumerate(kfold.split(X, y)):
#         model.fit(X.iloc[train,:], y.iloc[train,:])
#         score = model.score(X.iloc[test,:], y.iloc[test,:])
#         scores.append('%.8f'%score)
#     print(column + ":", scores)

# # Bunch has target and data attributes.
# print(df.keys())
# # in df, it appears that features are listed in feature_names
# print(df.feature_names)

# Split the train and test data sets
# X = pd.DataFrame(df["Temperature(F)"])
# y = pd.DataFrame(df["Severity"])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
# train, test = train_test_split(df, test_size = 0.25)


# # Instantiate the LinearRegression estimator
# #"Start_Time","Side","City","County","State","Zipcode","Timezone"
variables_list = ["Start_Lat","Distance(mi)","Temperature(F)","Wind_Chill(F)"]
model_list = [LinearRegression(),SGDRegressor()]

for column in variables_list:
    X = pd.DataFrame(df[column])
    y = pd.DataFrame(df["Severity"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
    for models in model_list:
        model = models;
        model.fit(X = X_train, y = y_train);
        # model.fit(X = pd.DataFrame(df[column]), y = pd.DataFrame(df["Severity"]));
        # print(model.intercept_)
        # for i, column in df.columns[:]:
        print("\u0332".join(column))
        # print()
        print("Coefficient:", model.coef_)
        # print(f'{column:>10}: {model.coef_[i]}')
        predicted = model.predict((X_test))  
        expected = y_test
        # # Accuracy
        print("R-Squared Score: ", metrics.r2_score(expected, predicted))
        print("Mean Absolute Error: ", metrics.mean_absolute_error(expected, predicted))
        print("Mean Squared Error: ", metrics.mean_squared_error(expected, predicted))
        print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(expected, predicted)))
        print("Mean Squared Log Error: ", np.sqrt(metrics.mean_squared_log_error(expected, predicted)))
        print()

# # ---visualization

# df = pd.DataFrame()
# df['Expected'] = pd.Series(expected)
# df['Predicted'] = pd.Series(predicted)
# plt.scatter(df['Expected'], df['Predicted'], alpha=0.5)
# plt.title('Scatter plot ')
# plt.xlabel('Expected')
# plt.ylabel('Predicted')
# plt.show()
