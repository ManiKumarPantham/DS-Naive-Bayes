######################################################################################
Business Problem: In this case study, you have been given Twitter data collected from 
an anonymous twitter handle.With the help of a Naïve Bayes model, 
predict if a given tweet about a real disaster is real or fake.
########################################################################################
# Importing required libraries
import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import sklearn.metrics as skmet
from sklearn.naive_bayes import MultinomialNB
import joblib

# Reading the data into Python
tweet_data = pd.read_csv(r'D:/Hands on/16_Naive Bayes/Assignment/Disaster_tweets_NB.csv')

# Taking only required columns
data = tweet_data.iloc[:, 3:]

# Information of the dataset
data.info()

# Statistical calculations
data.describe()

# Print top five records
data.head()

# Columns of the datset
data.columns

# Data types of the dataset
data.dtypes

# Spliting the dataset into train and test
train, test = train_test_split(data, random_state = 0, test_size = 0.2, stratify = data[['target']])

# Function to split the words from the each record
def split_words(i):
    return [word for word in i.split(' ')]

# Creating a CountVectorizer object and executing it
vector = CountVectorizer(analyzer = split_words).fit(data.text)

# Transforming the data
vector_trans = vector.transform(data.text)

# Transforming the train data
train_trans = vector.transform(train.text)

# Transforming the test data
test_trans = vector.transform(test.text)

# Creating SMOTE object
smote = SMOTE(random_state = 100)

# Resampling the data
x_train, y_train = smote.fit_resample(train_trans, train.target)

# Counts of the each unique values
y_train.unique()
y_train.values.sum()   # Number of '1's
y_train.size - y_train.values.sum()  # Number of '0's
y_train.value_counts()

# Creating a MultinomialNB object
multinb = MultinomialNB(alpha = 2)

# Model builind on train data
model = multinb.fit(x_train, y_train)

# Test on the test data
test_pred = model.predict(test_trans)

# Cross table
pd.crosstab(test.target, test_pred)

# Testing accuracy
skmet.accuracy_score(test.target, test_pred)

# Testing on train data
train_pred = model.predict(train_trans)

# Cross table
pd.crosstab(train.target, train_pred)

# Train accuracy
skmet.accuracy_score(train.target, train_pred)

# Trying different aplha values
'''
k = []
l = []
for i in range(10, 20):
    
    hyper_mulinb = MultinomialNB(alpha = i)

    hyper_model = hyper_mulinb.fit(x_train, y_train)

    hyper_test_pred = hyper_model.predict(test_trans)

    pd.crosstab(test.target, test_pred)

    k.append(skmet.accuracy_score(test.target, test_pred))

    hyper_train_pred = hyper_model.predict(train_trans)

    pd.crosstab(train.target, train_pred)

    l.append(skmet.accuracy_score(train.target, train_pred))
    
print(k)

print(l)
'''
# Defining pipeline
pipe1 = make_pipeline(vector, smote, model)

# Executing the pipeling
processed = pipe1.fit(train.text.ravel(), train.target.ravel())

# Save into local system
joblib.dump(processed, 'processed')

# load the saved model for predictions
saved_model = joblib.load('processed')

#Predictions
test_pred1 = saved_model.predict(test.text.ravel())

# Evaluation on Test Data with Metrics
# Confusion Matrix
pd.crosstab(test.target, test_pred1)

# Test accuracy
skmet.accuracy_score(test.target, test_pred1)

#Predictions
train_pred1 = saved_model.predict(train.text.ravel())

# Evaluation on Test Data with Metrics
# Confusion Matrix
pd.crosstab(train.target, train_pred1)

# Train accuracy
skmet.accuracy_score(train.target, train_pred1)