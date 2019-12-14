from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

df = pd.read_csv("US_Accidents_May19_cleaned_2.csv")
print(df.DESCR)
print(df.data.shape)
print(type(df))

# Bunch has target and data attributes.
print(df.keys())
# in df, it appears that features are listed in feature_names
print(df.feature_names)

# # Split the train and test data sets
# X_train, X_test, y_train, y_test = train_test_split(df.data, df.target, random_state=1)

# # Instantiate the LinearRegression estimator
# model = LinearRegression();
# model.fit(X=X_train, y = y_train);

# # Print the intercepts and coefficients LinearRegression() came up with
# print(model.intercept_)
# for i, name in enumerate(df.feature_names):
#     print(f'{name:>10}: {model.coef_[i]}')

# predicted = model.predict((X_test)) * 1000
# expected = y_test*1000


# # Accuracy

# print(metrics.r2_score(expected, predicted))
# print(metrics.mean_absolute_error(expected, predicted))
# print(metrics.mean_squared_error(expected, predicted))
# print(np.sqrt(metrics.mean_squared_error(expected, predicted)))
# print(np.sqrt(metrics.mean_squared_log_error(expected, predicted)))

# # ---visualization

# df = pd.DataFrame()
# df['Expected'] = pd.Series(expected)
# df['Predicted'] = pd.Series(predicted)
# plt.scatter(df['Expected'], df['Predicted'], alpha=0.5)
# plt.title('Scatter plot ')
# plt.xlabel('Expected')
# plt.ylabel('Predicted')
# plt.show()
