import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

DataFrame = pd.read_csv("C:\\Users\\cxu7\\Desktop\\PycharmProjects\\BIDA\\BI Final Project Raw Data.csv", low_memory=False)

#data is a list of features, file_name is a name of the your file as the path eg ("C:\\Users\\cxu7\\Desktop\\PycharmProjects\\BIDA\\BI Final Project Raw Data.csv")
def createModels(data, file_name):
    DataFrame = pd.read_csv(file_name,
                            low_memory=False)


# Split the train and test data sets
    X_train, X_test, y_train, y_test = train_test_split(DataFrame[data], DataFrame.Facebook, random_state=1)

# Instantiate the LinearRegression and SGDRegressor estimators
    model = LinearRegression()
    model2 = SGDRegressor()
    model.fit(X=X_train, y = y_train)
    model2.fit(X=X_train, y=y_train)


# Print the intercepts and coefficients LinearRegression() and SGDRegressor() came up with
    print(model.intercept_)
    print(model2.intercept_)
# for i, name in enumerate(DataFrame.feature_names):
#     print(f'{name:>10}: {model.coef_[i]}')

    predicted = model.predict((X_test))
    predicted2 = model2.predict((X_test))
    expected = y_test

# Accuracy
    print("r2_score for linear regression")
    print(metrics.r2_score(expected, predicted))
    print("r2_score for SGD Regressor")
    print(metrics.r2_score(expected, predicted2))
    print("mean absolute error for linear regression")
    print(metrics.mean_absolute_error(expected, predicted))
    print("mean absolute error for SGD regressor")
    print(metrics.mean_absolute_error(expected, predicted2))
    print("mean squared error for linear regression")
    print(metrics.mean_squared_error(expected, predicted))
    print("mean squared error for SGD regressor")
    print(np.sqrt(metrics.mean_squared_error(expected, predicted)))

    df = pd.DataFrame()
    df['Expected'] = pd.Series(expected)
    df['Predicted'] = pd.Series(predicted)
    df['Predicted2'] = pd.Series(predicted2)
    font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 12, }
    plt.figure()
    plt.scatter(df['Expected'], df['Predicted'], alpha=0.5)
    plt.title('Linear Regression')
    plt.xlabel('Expected')
    plt.ylabel('Predicted')
    plt.text(5000, 100, "r-squared value: " + str(metrics.r2_score(expected, predicted)), fontdict=font)
    plt.text(5000, 200, "mean-squared error: " + str(metrics.mean_squared_error(expected, predicted)), fontdict=font)
    plt.figure()
    plt.scatter(df['Expected'], df['Predicted2'], alpha=0.5)
    plt.title('SGD Regressor')
    plt.xlabel('Expected')
    plt.ylabel('Predicted2')
    plt.text(5000, 100, "r-squared value: " + str(metrics.r2_score(expected, predicted2)), fontdict=font)
    plt.text(5000, 200, "mean-squared error: " + str(metrics.mean_squared_error(expected, predicted2)), fontdict=font)
    plt.show()

#Running All Features/Calling Function
createModels(["SentimentTitle","SentimentHeadline","All Fields", "Month & year"], "C:\\Users\\cxu7\\Desktop\\PycharmProjects\\BIDA\\BI Final Project Raw Data.csv")
#Running Feature Set #2:
# createModels(["SentimentTitle","SentimentHeadline","All Fields"], "C:\\Users\\cxu7\\Desktop\\PycharmProjects\\BIDA\\BI Final Project Raw Data.csv")
#Running Feature Set #3:
# createModels(["SentimentTitle","SentimentHeadline"], "C:\\Users\\cxu7\\Desktop\\PycharmProjects\\BIDA\\BI Final Project Raw Data.csv")