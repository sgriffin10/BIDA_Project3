from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

df = pd.read_csv("US_Accidents_May19_cleaned_noNAs_1.csv")

# pd.options.display.float_format = "{:.2f}".format 
# print(df.describe())
# print(df.shape)
# print(df.dtypes)
target_name = [0, 1, 2, 3]

variables_list = df.columns[19:28]
# print(variables_list)
def int_float_list(name_of_list):
    new_list = []
    for column in variables_list:
        if df[column].dtypes == "int64" or df[column].dtypes == "float64":
            new_list.append(column)
    return new_list

new_list = int_float_list(variables_list) #double check
# Non-Integer Variables: "Start_Time","Side","City","County","State","Zipcode","Timezone" 
model_list = [KNeighborsClassifier(n_neighbors=100),KNeighborsClassifier(n_neighbors=50),KNeighborsClassifier(n_neighbors=10),KNeighborsClassifier(n_neighbors=5)] #a list of the classification models

#Spencer Comment: Not sure what regression parameters are????

for column in new_list: #runs through integer variables list above
    X = pd.DataFrame(df[column]) #Whatever is in the list above
    y = pd.DataFrame(df["Severity"]) #Severity is dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1) #25/75 test-training split
    for models in model_list: #loops through the models in list above
        model = models;
        model.fit(X = X_train, y = y_train.values.ravel()); #ravel() fixes dataConversion warning
        # print(model.intercept_) #prints intercept
        print("\u0332".join(column)) #prints underline
        predicted = model.predict(X=X_test)
        expected = y_test
        print(predicted[:20])
        print(expected[:20])
        ### Accuracy Scores ###
        print() #prints space
        wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]# Isolate the wrong predictions
        print(wrong)
        print(model.score(X_test, y_test))#Accuracy
        #confusion matrix - each cell represents the frequency of a class
        # rows --> expected classes
        # columns --> predicted classes
        confusion = confusion_matrix(y_true=expected, y_pred = predicted)
        print(confusion)
        names = [str(digit) for digit in target_name] #not sure if correct
        print(classification_report(expected, predicted, target_names = names))



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
