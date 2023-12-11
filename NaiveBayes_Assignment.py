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

#########################################################################################
1.) Prepare a classification model using the Naive Bayes algorithm for the salary dataset. 
    Train and test datasets are given separately. Use both for model building.
    
#########################################################################################
# Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Reading a dataset into Python
sal_train = pd.read_csv('D:/Hands on/16_Naive Bayes/Assignment/SalaryData_Train.csv')

# Reading a dataset into Python
sal_test = pd.read_csv('D:/Hands on/16_Naive Bayes/Assignment/SalaryData_Test.csv')

# Appending two datasets into one dataset
data = sal_train.append(sal_test)

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe() 

# Datatypes of the dataset
data.dtypes

# Droping race columns from dataset as it is irrelavant
data.drop(['race'], inplace = True, axis = 1)

#First moment Business decession
data.mean()

data.median()

data.mode()

#Second moment business decession
data.var()

data.std()

#Third moment business decession
data.skew()

#Fourth moment business decession
data.kurt()

# Checking null values
data.isna().sum()

# Checking duplicate values
dup = data.duplicated()

# Sum of duplicate values
dup.sum()

# Dropping duplicate values
data = data.drop_duplicates()

# Checking duplicate values
dup = data.duplicated()

# Sum of duplicates
dup.sum()

# Correlation coefficient
data.corr()

# Pairplot
sns.pairplot(data)

# Chaning Salary column to 0 & 1( <= 50K --> 0, > 50K --> 1)
data['Salary'] = np.where(data.Salary == ' <=50K', 0, 1)

# Information of the Dataset
data.info()

# Selecting categorical features
cat_features = data.select_dtypes(include = 'object')

# Categorical features columns
cat_features.columns

# Selecting numerical features
num_features = data.select_dtypes(exclude = 'object')

# Numerical features columns
num_features.columns

# Boxplot
for i in num_features.columns[:]:
    sns.boxplot(num_features[i])
    plt.title('Boxplot for ' + str(i))
    plt.show()
    
# Winsorization
age_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['age'])
data['age'] = age_winsor.fit_transform(data[['age']])

eduno_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['educationno'])
data['educationno'] = eduno_winsor.fit_transform(data[['educationno']])

capgain_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['capitalgain'])
data['capitalgain'] = capgain_winsor.fit_transform(data[['capitalgain']])

caploss_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['capitalloss'])
data['capitalloss'] = caploss_winsor.fit_transform(data[['capitalloss']])


hrspweek_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['hoursperweek'])
data['hoursperweek'] = hrspweek_winsor.fit_transform(data[['hoursperweek']])

# Boxplot after Winsorization
for i in num_features.columns:
    sns.boxplot(num_features[i])
    plt.title('Boxplot for ' + str(i))
    plt.show()

# Pairplot
sns.pairplot(data)
plt.show()

# Normal QQ plot
for i in num_features.columns:
    sm.qqplot(num_features[i])
    plt.title('Normal QQ plot for ' + str(i))
    plt.show()

# Creating dummy variables
dummy_cat_features = pd.get_dummies(cat_features, columns = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'sex', 'native'], drop_first = True)

# Creating MixMaxSclaer and RobustScaler objects
robustscaler = RobustScaler()
minmaxscale = MinMaxScaler()

# Executing scaling 
num_features['age'] = pd.DataFrame(minmaxscale.fit_transform(num_features[['age']]))
num_features['educationno'] = pd.DataFrame(minmaxscale.fit_transform(num_features[['educationno']]))
num_features['capitalgain'] = pd.DataFrame(minmaxscale.fit_transform(num_features[['capitalgain']]))
num_features['capitalloss'] = pd.DataFrame(minmaxscale.fit_transform(num_features[['capitalloss']]))
num_features['hoursperweek'] = pd.DataFrame(minmaxscale.fit_transform(num_features[['hoursperweek']]))

# Prints top five records
num_features.head()

# Concatinating categorical and numerical features
new_data = pd.concat([num_features, dummy_cat_features], axis = 1)

# Inforamtion of the dataset
new_data.info()

# Statistical values of the dataset
new_data.describe()

# Count of each unique value
new_data.Salary.value_counts()

# Spliting train test datasets
train, test = train_test_split(new_data, test_size = 0.2, random_state = 0, stratify = new_data[['Salary']])

# creating SMOTE object
smote = SMOTE(random_state = 1)

# Resampling the dataset
x_train, y_train = smote.fit_resample(train, train.Salary)

# Count of each unique values
y_train.value_counts()

# creating MultinomialNB object
vanilla = MultinomialNB()

# Creating a model
model = vanilla.fit(x_train, y_train)

# Test on test data
test_pred = model.predict(test)

# Cross table
pd.crosstab(test.Salary, test_pred)

# Test accuracy
test_acc = np.mean(test.Salary == test_pred)

#confusion metrix
cm = confusion_matrix(test.Salary, test_pred)
cmplot = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['<=50K', '>50K'])
cmplot.plot()

# Test accuracy
test_acc1 = accuracy_score(test.Salary, test_pred)

# Test on Training data
train_pred = model.predict(train)

# crosstable
pd.crosstab(train.Salary, train_pred)

# Train accuracy
train_acc1 = accuracy_score(train.Salary, train_pred)

# Testing with different alpha values
'''
k = []
l = []
for i in range(2, 10):
    vanilla = MultinomialNB(alpha = i)
    model = vanilla.fit(x_train, y_train)
    test_pred = model.predict(test)
    k.append(accuracy_score(test.Salary, test_pred))
    
    train_pred = model.predict(train)
    l.append(accuracy_score(train.Salary, train_pred))
    
    
print(k)
print(l)
'''
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
