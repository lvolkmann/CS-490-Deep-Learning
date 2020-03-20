import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# QUESTION 1
# Adding Dense layers to the existing source code

dataset = pd.read_csv("diabetes.csv", header=None).values

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8],
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
my_first_nn.summary()
loss, accuracy = my_first_nn.evaluate(X_test, Y_test)

print("LOSS: {}".format(loss))
print("ACCURACY: {}".format(accuracy))

# Now the same model again with more Dense layers

np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
my_first_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
my_first_nn.summary()
loss2, accuracy2 = my_first_nn.evaluate(X_test, Y_test)

print("NEW LOSS: {} CHANGE: {}".format(loss2, loss2 - loss))
print("NEW ACCURACY: {} CHANGE: {}".format(accuracy2, accuracy2 - accuracy))

# Now the same model again with more epochs

np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
my_first_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=150,
                                     initial_epoch=0)
my_first_nn.summary()
loss3, accuracy3 = my_first_nn.evaluate(X_test, Y_test)

print("NEW LOSS: {} CHANGE: {}".format(loss3, loss3 - loss2))
print("NEW ACCURACY: {} CHANGE: {}".format(accuracy3, accuracy3 - accuracy2))


# QUESTION 2
# changing the data set

dataset_df = pd.read_csv("Breas Cancer.csv")

dataset_df["diagnosis"] = dataset_df["diagnosis"].replace('M', 1)
dataset_df["diagnosis"] = dataset_df["diagnosis"].replace('B', 0)

dataset = dataset_df.values


X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:31], dataset[:,1],
                                                   test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model

# we now have a larger input dimension
my_first_nn.add(Dense(20, input_dim=29, activation='relu')) # hidden layer
my_first_nn.add(Dense(20, input_dim=29, activation='relu')) # hidden layer
my_first_nn.add(Dense(20, input_dim=29, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
my_first_nn.summary()
loss, accuracy = my_first_nn.evaluate(X_test, Y_test)

print("LOSS: {}".format(loss))
print("ACCURACY: {}".format(accuracy))


# QUESTION 3
# TODO: normalize

# Normalizing imports
from sklearn.preprocessing import StandardScaler


dataset_df = pd.read_csv("Breas Cancer.csv")

dataset_df["diagnosis"] = dataset_df["diagnosis"].replace('M', 1)
dataset_df["diagnosis"] = dataset_df["diagnosis"].replace('B', 0)

dataset = dataset_df.values


print("BEFORE NORMALIZATION")
print(dataset[:,2:])

sc = StandardScaler().fit(dataset[:,2:])
dataset[:,2:] = sc.transform(dataset[:,2:])

print("AFTER NORMALIZATION")
print(dataset[:,2:])

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:31], dataset[:,1],
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=29, activation='relu')) # hidden layer
my_first_nn.add(Dense(20, input_dim=29, activation='relu')) # hidden layer
my_first_nn.add(Dense(20, input_dim=29, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
my_first_nn.summary()
loss2, accuracy2 = my_first_nn.evaluate(X_test, Y_test)

print("NEW LOSS: {} CHANGE: {}".format(loss2, loss2 - loss))
print("NEW ACCURACY: {} CHANGE: {}".format(accuracy2, accuracy2 - accuracy))