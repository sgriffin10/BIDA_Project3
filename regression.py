from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def linear_model(dataframe, set_list):
    
    #Creates Testing/Training Sets
    X = pd.DataFrame(dataframe[set_list]) 
    y = pd.DataFrame(dataframe["CLM_AMT"]) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1) 
    
    model = LinearRegression();
    model.fit(X = X_train, y = y_train);
    # print(model.intercept_) #prints intercept
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    print("Coefficient:", model.coef_) 
    predicted1 = model.predict((X_test))  
    expected1 = y_test
    # print(predicted1, expected1)
    return predicted1, expected1
        
def SGD_model(dataframe, set_list):
    
    #Creates Testing/Training Sets
    X = pd.DataFrame(dataframe[set_list]) 
    y = pd.DataFrame(dataframe["CLM_AMT"]) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1) 
    
    model = SGDRegressor();
    model.fit(X = X_train, y = y_train.values.ravel());
    # print(model.intercept_) #prints intercept
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    print("Coefficient:", model.coef_) 
    predicted2 = model.predict((X_test))  
    expected2 = y_test
    return predicted2, expected2

def linear_accuracy_score(dataframe, setlist):
    ### Creating Accuracy Variables ###
    predicted, expected = linear_model(dataframe, setlist)
    R_Squared_Score = metrics.r2_score(expected, predicted)
    Mean_Absolute_Error = metrics.mean_absolute_error(expected, predicted)
    Mean_Squared_Error = metrics.mean_squared_error(expected, predicted)
    Root_Mean_Squared_Error = np.sqrt(metrics.mean_squared_error(expected, predicted))
    # Mean_Squared_Log_Error = np.sqrt(metrics.mean_squared_log_error(expected, predicted))

    ### Printing Accuracy Scores ###
    print("R-Squared Score: ", R_Squared_Score)
    print("Mean Absolute Error: ", Mean_Absolute_Error)
    print("Mean Squared Error: ", Mean_Squared_Error)
    print("Root Mean Squared Error: ", Root_Mean_Squared_Error)
    # print("Mean Squared Log Error: ", Mean_Squared_Log_Error)#logarithm brings the points closer ; look at it in a graph
    print() 

def sgd_accuracy_score(dataframe, setlist):
    ### Creating Accuracy Variables ###
    predicted, expected = SGD_model(dataframe, setlist)
    R_Squared_Score = metrics.r2_score(expected, predicted)
    Mean_Absolute_Error = metrics.mean_absolute_error(expected, predicted)
    Mean_Squared_Error = metrics.mean_squared_error(expected, predicted)
    Root_Mean_Squared_Error = np.sqrt(metrics.mean_squared_error(expected, predicted))
    # Mean_Squared_Log_Error = np.sqrt(metrics.mean_squared_log_error(expected, predicted))

    ### Printing Accuracy Scores ###
    print("R-Squared Score: ", R_Squared_Score)
    print("Mean Absolute Error: ", Mean_Absolute_Error)
    print("Mean Squared Error: ", Mean_Squared_Error)
    print("Root Mean Squared Error: ", Root_Mean_Squared_Error)
    # print("Mean Squared Log Error: ", Mean_Squared_Log_Error)#logarithm brings the points closer ; look at it in a graph
    print() 

def linear_reg_visusalization(dataframe, setlist):

    # predicted, expected = linear_model(dataframe, setlist)
    df_ = pd.DataFrame()
    df_ = df_.set_index('date')
    print(df)
    # df.index = pd.to_datetime(df.index)
    # df_['Expected'] = linear_model(dataframe, setlist)
    # df_['Predicted'] = pd.Series(predicted)
    # # font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 12, }
    # plt.figure()
    # plt.scatter(dataframe['Expected'], dataframe['Predicted'], alpha=0.5)
    # plt.title('Linear Regression Scatter Plot')
    # plt.xlabel('Expected')
    # plt.ylabel('Predicted')
    # plt.text(5000, 100, "r-squared value: " + str(R_Squared_Score), fontdict=font)
    # plt.text(5000, 200, "mean-squared error: " + str(Mean_Squared_Error), fontdict=font)
    # plt.figure()
    # plt.show()
    # plt.scatter(df['Expected'], df['Predicted2'], alpha=0.5)
    # plt.title('SGD Regressor')
    # plt.xlabel('Expected')
    # plt.ylabel('Predicted2')
    # plt.text(5000, 100, "r-squared value: " + str(metrics.r2_score(expected, predicted2)), fontdict=font)
    # plt.text(5000, 200, "mean-squared error: " + str(metrics.mean_squared_error(expected, predicted2)), fontdict=font)



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

    # linear_accuracy_score(df, entire_list)
    # sgd_accuracy_score(df, entire_list)
    # linear_accuracy_score(df, subset_list_human_char)
    # sgd_accuracy_score(df, subset_list_human_char)
    # linear_accuracy_score(df, subset_list_car_char)
    # sgd_accuracy_score(df, subset_list_car_char)

    linear_reg_visusalization(df, entire_list)

    #List of Models
    # model_list = [LinearRegression(),SGDRegressor(eta0=0.000000004)]
    # play w. parameters
    # explain that we tried the best we can; but that we could explain it at some later time
    # share the original & clean code 


    #Calls Regresson Model 
    # regression_model(df, entire_list, model_list)
    # regression_model(df, subset_list_human_char, model_list)
    # regression_model(df, subset_list_car_char, model_list)

    

if __name__ == "__main__":
    main()





