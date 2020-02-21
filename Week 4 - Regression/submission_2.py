# @author: Landon Volkmann

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('winequality-red.csv')

# handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

# Working with Numeric Features
numeric_features = train.select_dtypes(include=[np.number])

corr = numeric_features.corr()

print(corr['quality'].sort_values(ascending=False)[:5], '\n')
print(corr['quality'].sort_values(ascending=False)[-5:])

print("NOTICE three top features are: ", end="")
print("alcohol, volatile acidity, and sulphates")

# Build a linear model
y = np.log(train.quality)
X = np.log(data[['alcohol', 'volatile acidity', 'sulphates']])
print(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=.33)

from sklearn import linear_model

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
# Evaluate the performance and visualize results
print("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error

print('RMSE is: \n', mean_squared_error(y_test, predictions))

# visualize

fig, axes = plt.subplots(1, len(X_train.columns.values), sharey=True, constrained_layout=True, figsize=(30, 15))

# Credit: https://stackoverflow.com/questions/52404857/how-do-i-plot-for-multiple-linear-regression-model-using-matplotlib
for i, e in enumerate(X_test.columns):
    lr.fit(X_test[e].values[:, np.newaxis], y_test.values)
    axes[i].set_title("Best fit line")
    axes[i].set_xlabel(str(e))
    axes[i].set_ylabel('Quality')
    axes[i].scatter(X_test[e].values[:, np.newaxis], y_test, color='g')
    axes[i].plot(X_test[e].values[:, np.newaxis],
                 lr.predict(X_test[e].values[:, np.newaxis]), color='k')

plt.show()
