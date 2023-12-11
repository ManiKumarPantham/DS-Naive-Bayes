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