from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def regression_model(dataframe, set_list, list_of_models):
    
    #Creates Testing/Training Sets
    X = pd.DataFrame(dataframe[set_list]) 
    y = pd.DataFrame(dataframe["CLM_AMT"]) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1) 

    #Model Loop
    for models in list_of_models: 
        model = models;
        model.fit(X = X_train, y = y_train.values.ravel());
        # print(model.intercept_) #prints intercept
        np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
        print("Coefficient:", model.coef_) 
        predicted = model.predict((X_test))  
        expected = y_test
        
        ### Accuracy Scores ###
        print("R-Squared Score: ", metrics.r2_score(expected, predicted))
        print("Mean Absolute Error: ", metrics.mean_absolute_error(expected, predicted))
        print("Mean Squared Error: ", metrics.mean_squared_error(expected, predicted))
        print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(expected, predicted)))
        # print("Mean Squared Log Error: ", np.sqrt(metrics.mean_squared_log_error(expected, predicted)))
        print() 

def main():

    #Reads CSV into dataframe
    df = pd.read_csv("car_insurance_claim_noNAs_2.csv")

    #Creating List of Columns
    entire_list = []
    for column in df.columns[1:19]:
        entire_list.append(column)
    for column in df.columns[20:23]:
            entire_list.append(column)

    #Creating Subset Lists
    subset_list_human_char = ["KIDSDRIV", "AGE", "YOJ", "INCOME", "PARENT1", "HOME_VAL", "MSTATUS", "GENDER", "EDUCATION", "OCCUPATION"]
    subset_list_car_char = ["TRAVTIME", "CAR_USE", "BLUEBOOK", "CAR_TYPE", "OLDCLAIM", "CLM_FREQ", "REVOKED", "MVR_PTS", "CAR_AGE", "CLAIM_FLAG", "URBANICITY"]

    #List of Models
    model_list = [LinearRegression(),SGDRegressor()]

    #Calls Regresson Model 
    regression_model(df, entire_list, model_list)
    regression_model(df, subset_list_human_char, model_list)
    regression_model(df, subset_list_car_char, model_list)

    

if __name__ == "__main__":
    main()
