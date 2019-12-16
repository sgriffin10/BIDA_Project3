from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def regression(df, list_of_set, list_of_model):
    for a_list in list_of_set:
        X_train, X_test, y_train, y_test = train_test_split(df[a_list], df["CLM_AMT"], test_size = 0.25, random_state=1) 
        #Model Loop
        for models in list_of_model: 
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
            # print("Mean Squared Log Error: ", np.sqrt(metrics.mean_squared_log_error(expected + 1286, predicted + 1286)))#logarithm brings the points closer ; look at it in a graph
            print() 

            df_graph = pd.DataFrame()
            df_graph['Expected'] = pd.Series(expected)
            df_graph['Predicted'] = pd.Series(predicted)
            # font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 12, }
            # plt.figure()
            plt.scatter(df_graph['Expected'], df_graph['Predicted'], alpha=0.5)
            plt.title('Scatter Plot')
            plt.xlabel('Expected')
            plt.ylabel('Predicted')
            plt.text(5000, 1000, "r-squared value: " + str(metrics.r2_score(expected, predicted)))
            plt.text(5000, 200, "mean-squared error: " + str(metrics.mean_squared_error(expected, predicted)))
            plt.show()
            # df = pd.read_csv("car_insurance_claim_noNAs_2.csv")


def main():
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

    set_list = [entire_list, subset_list_human_char, subset_list_car_char]
    # print(entire_thing)

    #List of Models
    model_list = [LinearRegression(),SGDRegressor(eta0=0.000000004)]
    # play w. parameters
    # explain that we tried the best we can; but that we could explain it at some later time
    # share the original & clean code 

    regression(df, set_list, model_list)

if __name__ == "__main__":
    main()