import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

# Reading the csv files
dataset = pd.read_csv('Data.csv')


# Importing the dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
# print(X)

# Handling the missing data
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


# Creating a dummy variable
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder._fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# Splitting the dataset into training sets and Test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

print(X_train)
