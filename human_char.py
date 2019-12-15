from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
# from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

df = pd.read_csv("car_insurance_claim_noNAs_2.csv")

human_list = ["AGE", "YOJ", "INCOME", "PARENT1", "HOME_VAL", "MSTATUS", "GENDER", "EDUCATION", "OCCUPATION"]
y = df["CLM_AMT"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model_list = [LinearRegression(),SGDRegressor()]

for models in model_list:
    model = models;
    model.fit(X=x_train, y=y_train);
    print(model.intercept_)
    print("Coefficient:", model.coef_) #prints coefficient
    predicted = model.predict((X_test))  
    expected = y_test

    print("R-Squared Score: ", metrics.r2_score(expected, predicted))
    print("Mean Absolute Error: ", metrics.mean_absolute_error(expected, predicted))
    print("Mean Squared Error: ", metrics.mean_squared_error(expected, predicted))
    print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(expected, predicted)))
    print()
