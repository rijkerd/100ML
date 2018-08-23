import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simple linear Regression
# Is a method to predict dependent variable(Y)
# based on the values of independent variables (X)


# Task predict the percentage of marks that a student
# is expected to score based upon the number of hours the
# studied

# (Y) is the Dependent Variable Score
# (X) is the independent Variable Hours

# Data Preprocessing
dataset = pd.read_csv('studentscores.csv')  # Read the data from the csv file

X = dataset.iloc[:, :1].values  # Extract the Hours dataset
Y = dataset.iloc[:, 1].values  # Extract the score from the dataset

# Use for splitting Model Training and testing data
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/4, random_state=0)


# Fiiting Simple Linear Regression Model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)


# Predicting the Result
Y_pred = regressor.predict(X_test)

# Visualization

# Visualising the Training Results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')


# Visualize the Test Results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
