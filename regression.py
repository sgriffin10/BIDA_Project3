from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

df = pd.read_csv("car_insurance_claim_noNAs_2.csv")

# print(df.describe())
# print(df.shape)
# print(df.dtypes)

model_list = [LinearRegression()] #,SGDRegressor()
empty_list = []
for column in df.columns[1:18]:
    empty_list.append(column)
for column in df.columns[19:22]:
        empty_list.append(column)
print(empty_list)
    


human_list = ["AGE", "YOJ", "INCOME", "PARENT1", "HOME_VAL", "MSTATUS", "GENDER", "EDUCATION", "OCCUPATION"]

#Spencer Comment: Not sure what regression parameters are????
# def regression_model(dataframe):
# for column in variables_list: #runs through integer variables list above
X = pd.DataFrame(df[empty_list]) #Whatever is in the list above
y = pd.DataFrame(df["CLM_AMT"]) #Severity is dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1) #25/75 test-training split
for models in model_list: #loops through the 2 models in list above
    model = models;
    model.fit(X = X_train, y = y_train);
    # print(model.intercept_) #prints intercept
    # print("\u0332".join(human_list)) #prints underline
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    print("Coefficient:", model.coef_) #prints coefficient
    predicted = model.predict((X_test))  
    expected = y_test
    ### Accuracy Scores ###
    # Spencer Comment: Will need to assign the below to variables and then print
    print("R-Squared Score: ", metrics.r2_score(expected, predicted))
    print("Mean Absolute Error: ", metrics.mean_absolute_error(expected, predicted))
    print("Mean Squared Error: ", metrics.mean_squared_error(expected, predicted))
    print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(expected, predicted)))
    # print("Mean Squared Log Error: ", np.sqrt(metrics.mean_squared_log_error(expected, predicted)))
    print() #prints space

# def main():
#     regression_model(df)

# if __name__ == "__main__":
#     main()
