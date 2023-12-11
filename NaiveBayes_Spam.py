#################################################################################################
Business Problem: This dataset contains information of users in a social network. 
This social network has several business clients which can post ads on it. 
One of the clients has a car company which has just launched a luxury SUV for a ridiculous price. 
Build a Bernoulli Naïve Bayes model using this dataset and classify which of the users of the social network are going to purchase this luxury SUV. 
1 implies that there was a purchase and 0 implies there wasn’t a purchase.

    
###################################################################################################
# Importing required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# Reading the dataset into Python
data = pd.read_csv('D:/Hands on/16_Naive Bayes/Assignment/NB_Car_Ad.csv')

# Basic information of the dataset
data.info()

# Statistica values of the dataset
data.describe()

# Prints top five records
data.head()

# Columns of the dataset
data.columns

# Datatypes of the dataset
data.dtypes

# 1st moment business decession
data.mean()

data.median()

data.mode()

# 2nd moment business decession
data.var()

data.std()

# 3rd moment business decession
data.skew()

# 4th moment business decession
data.kurt()

# User ID columns nominal data and it is irrelavant
data.drop(['User ID'], axis = 1, inplace = True)

# Information of the dataset
data.info()

# Duplicates check
dup = data.duplicated()

# Sum of duplicates
dup.sum() 

# Removing duplicates
new_data = data.drop_duplicates()

# Sum of duplicates after removing duplicates
new_data.duplicated().sum()

# Checking null values
new_data.isna().sum()
data.isna().sum()

# subplots
new_data.plot(kind = 'box', subplots = True, sharey = False)
plt.show()

# Boxplot
plt.boxplot(new_data.EstimatedSalary)

# Creating dummy columns to the Gender feature
final_data = pd.get_dummies(new_data, columns = ['Gender'], drop_first = True)

# Information of the dataset
final_data.info()

# Changing the order of the features
final_data = final_data.loc[:, ['Age', 'EstimatedSalary', 'Gender_Male', 'Purchased']]

# Creating MinMaxScaler object
minmaxscale = MinMaxScaler()

# Executing and transforming MinMaxScaler on final_data dataset
final_data1  = pd.DataFrame(minmaxscale.fit_transform(final_data))

# Renaming the columns
final_data1.rename(columns = {0 : 'Age', 1 : 'EstimatedSalary', 2 : 'Gender_Male', 3 : 'Purchased'}, inplace = True)

# Information of the Dataset
final_data1.info()

# Statistical values of the dataset
final_data1.describe()

# Segregating the Input from the dataset 
X = final_data1.iloc[:, 0:3]

# Segregating the Output from the dataset
Y = final_data1.iloc[:, -1]

# Spliting the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state =1, test_size = 0.2)

# Creating a BernoulliNB object
classifier =  BernoulliNB(alpha = 5)

# Creating a model on train data
model = classifier.fit(x_train, y_train)

# Testing a model on test data
test_pred = model.predict(x_test)

# Cross table for original and predicted values
pd.crosstab(y_test, test_pred)

# confusion matrix
confusion_matrix(y_test, test_pred)

# Finding the test accuracy
test_acc = accuracy_score(y_test, test_pred)

# Testing the model on train data
train_pred = model.predict(x_train)

# Cross table for original and predicted values
pd.crosstab(y_train, train_pred)

# confusion matrix
confusion_matrix(y_train, train_pred)

# Finding the test accuracy
train_acc = accuracy_score(y_train, train_pred)
