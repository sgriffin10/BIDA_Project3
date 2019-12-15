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

### Instantiate the LinearRegression estimator ###
variables_list = ["Start_Lat","Distance(mi)","Temperature(F)","Wind_Chill(F)","Precipitation(in)"] # a list of integer variables
# Non-Integer Variables: "Start_Time","Side","City","County","State","Zipcode","Timezone" 
model_list = [LinearRegression(),SGDRegressor()] #a list of the regression models

#Spencer Comment: Not sure what regression parameters are????

for column in variables_list: #runs through integer variables list above
    X = pd.DataFrame(df[column]) #Whatever is in the list above
    y = pd.DataFrame(df["Severity"]) #Severity is dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0) #25/75 test-training split
    for models in model_list: #loops through the 2 models in list above
        model = models;
        model.fit(X = X_train, y = y_train);
        # print(model.intercept_) #prints intercept
        print("\u0332".join(column)) #prints underline
        print("Coefficient:", model.coef_) #prints coefficient
        predicted = model.predict((X_test))  
        expected = y_test
        ### Accuracy Scores ###
        # Spencer Comment: Will need to assign the below to variables and then print
        print("R-Squared Score: ", metrics.r2_score(expected, predicted))
        print("Mean Absolute Error: ", metrics.mean_absolute_error(expected, predicted))
        print("Mean Squared Error: ", metrics.mean_squared_error(expected, predicted))
        print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(expected, predicted)))
        print("Mean Squared Log Error: ", np.sqrt(metrics.mean_squared_log_error(expected, predicted)))
        print() #prints space

# #  ---visualization

# Spencer Comment: We will need to change this section to be specific

# df = pd.DataFrame()
# df['Expected'] = pd.Series(expected)
# df['Predicted'] = pd.Series(predicted)
# plt.scatter(df['Expected'], df['Predicted'], alpha=0.5)
# plt.title('Scatter plot ')
# plt.xlabel('Expected')
# plt.ylabel('Predicted')
# plt.show()
