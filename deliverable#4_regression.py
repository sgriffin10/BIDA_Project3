from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np


def run_regression_model(data_set, data_target, model):
    X_train, X_test, y_train, y_test = train_test_split(data_set, data_target, test_size=0.25, random_state=1)
    model.fit(X=X_train, y=y_train);

    ### Printing Intercept & Coeffficients
    ### We comment line 15-18 because we don't want to print out intercept and coefficients everytime when we call function get_accuracy_score and get_visualization
    # print(model.intercept_) #prints intercept
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    # print("Coefficient:", model.coef_) 

    predicted = model.predict(X_test)
    expected = y_test
    # print(predicted, expected)
    return predicted, expected


def get_accuracy_score(data_set, data_target, model):
    ### Creating Accuracy Variables ###
    predicted, expected = run_regression_model(data_set, data_target, model)
    R_Squared_Score = metrics.r2_score(expected, predicted)
    Mean_Absolute_Error = metrics.mean_absolute_error(expected, predicted)
    Mean_Squared_Error = metrics.mean_squared_error(expected, predicted)
    Root_Mean_Squared_Error = np.sqrt(metrics.mean_squared_error(expected, predicted))
    # Mean_Squared_Log_Error = np.sqrt(metrics.mean_squared_log_error(expected, predicted)) # Natural Logarithm cannot take negative value, so we comment it out

    ### Printing Accuracy Scores ###
    print("R-Squared Score: ", R_Squared_Score)
    print("Mean Absolute Error: ", Mean_Absolute_Error)
    print("Mean Squared Error: ", Mean_Squared_Error)
    print("Root Mean Squared Error: ", Root_Mean_Squared_Error)
    # print("Mean Squared Log Error: ", Mean_Squared_Log_Error)
    print() 
    return R_Squared_Score, Mean_Squared_Error


def get_visualization(data_set, data_target, model):
    predicted, expected = run_regression_model(data_set, data_target, model)
    R_Squared_Score, Mean_Squared_Error = get_accuracy_score(data_set, data_target, model)
    df_graph = pd.DataFrame()
    df_graph['Expected'] = pd.Series(expected)
    df_graph['Predicted'] = pd.Series(predicted)
    plt.scatter(df_graph['Expected'], df_graph['Predicted'], alpha=0.5)
    plt.title('Scatter Plot')
    plt.xlabel('Expected')
    plt.ylabel('Predicted')
    plt.text(5000, 100, "r-squared value: " + str(R_Squared_Score))
    plt.text(5000, 500, "mean-squared error: " + str(Mean_Squared_Error))
    plt.show()


def main():
    #Reads CSV into dataframe
    df = pd.read_csv("car_insurance_claim_noNAs_2.csv")

    #Original Data Set (With All Attributes)
    df_data_set = df.drop(['CLM_AMT'], axis=1)
    df_target = df['CLM_AMT']
    # print(df.shape) # 23 columns
    # print(df_data.shape) # 22 columns
    # print(df_target.shape) # single column

    #Creating Two Subsets
    human_char_list = ["KIDSDRIV", "AGE", "YOJ", "INCOME", "PARENT1", "HOME_VAL", "MSTATUS", "GENDER", "EDUCATION", "OCCUPATION"]
    human_char_subset = df[human_char_list]
    car_char_list = ["TRAVTIME", "CAR_USE", "BLUEBOOK", "CAR_TYPE", "OLDCLAIM", "CLM_FREQ", "REVOKED", "MVR_PTS", "CAR_AGE", "CLAIM_FLAG", "URBANICITY"]
    car_char_subset = df[car_char_list]

    #Accuracy Score
    get_accuracy_score(df_data_set, df_target, LinearRegression());
    get_accuracy_score(df_data_set, df_target, SGDRegressor());
    get_accuracy_score(human_char_subset, df_target, LinearRegression());
    get_accuracy_score(human_char_subset, df_target, SGDRegressor());
    get_accuracy_score(car_char_subset, df_target, LinearRegression());
    get_accuracy_score(car_char_subset, df_target, SGDRegressor());

    #Visualization
    get_visualization(df_data_set, df_target, LinearRegression());
    get_visualization(df_data_set, df_target, SGDRegressor())
    get_visualization(human_char_subset, df_target, LinearRegression());
    get_visualization(human_char_subset, df_target, SGDRegressor());
    get_visualization(car_char_subset, df_target, LinearRegression());
    get_visualization(car_char_subset, df_target, SGDRegressor());


if __name__ == "__main__":
    main()
