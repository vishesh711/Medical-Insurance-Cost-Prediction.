# Insurance Cost Prediction Model

## Overview

This project involves building a linear regression model to predict insurance charges based on various factors such as age, sex, BMI, number of children, smoking status, and region. The dataset used for this project is obtained from a CSV file and contains 1338 entries with 7 columns.

## Dataset

The dataset consists of the following columns:
- **age**: Age of the primary beneficiary
- **sex**: Gender of the primary beneficiary (male/female)
- **bmi**: Body mass index, providing an understanding of body weight relative to height
- **children**: Number of children/dependents covered by health insurance
- **smoker**: Smoking status of the primary beneficiary (yes/no)
- **region**: Residential area in the US (northeast, northwest, southeast, southwest)
- **charges**: Medical insurance charges billed by health insurance

## Data Preprocessing

The data preprocessing steps include:
1. Loading the data into a Pandas DataFrame.
2. Checking for null values and ensuring there are none.
3. Descriptive statistics of the dataset.
4. Visualizing the distribution of each feature.
5. Encoding categorical variables using numerical values.

## Data Visualization

We visualized the distribution of the following features:
- **Age Distribution**
- **Sex Distribution**
- **BMI Distribution**
- **Children Count**
- **Smoker Count**
- **Region Count**
- **Charges Distribution**

## Feature Encoding

We encoded categorical variables:
- **sex**: male (0), female (1)
- **smoker**: yes (0), no (1)
- **region**: southeast (0), southwest (1), northeast (2), northwest (3)

## Model Building

We used a linear regression model to predict the insurance charges. The steps involved:
1. Splitting the data into training and testing sets.
2. Training the linear regression model on the training set.
3. Evaluating the model using the R-squared value on both training and testing sets.

## Model Evaluation

The R-squared values obtained were:
- Training set: 0.7515
- Testing set: 0.7447

These values indicate that the model explains approximately 75% of the variance in the training set and 74% in the testing set.

## Predictions

We made predictions on new data and provided an example with the input data (31, 1, 25.74, 0, 1, 0), which corresponds to a 31-year-old female with a BMI of 25.74, no children, non-smoker, residing in the southeast region. The predicted insurance charge was USD 3760.08.

## Code

Here is the complete code for the project:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# loading the data from csv file to a Pandas DataFrame
insurance_dataset = pd.read_csv('/content/insurance.csv')
insurance_dataset.head()

# number of rows and columns
insurance_dataset.shape

# getting some informations about the dataset
insurance_dataset.info()

insurance_dataset.describe()

insurance_dataset.isnull().sum()

# distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()

# Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()

insurance_dataset['sex'].value_counts()

# bmi distribution
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()

# children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children')
plt.show()

insurance_dataset['children'].value_counts()

# smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('smoker')
plt.show()

insurance_dataset['smoker'].value_counts()

# region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('region')
plt.show()

insurance_dataset['region'].value_counts()

# distribution of charges value
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()

# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

# encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# loading the Linear Regression model
regressor = LinearRegression()

regressor.fit(X_train, Y_train)

# prediction on training data
training_data_prediction = regressor.predict(X_train)

# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)

# prediction on test data
test_data_prediction = regressor.predict(X_test)

# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)

input_data = (31,1,25.74,0,1,0)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)

print('The insurance cost is USD ', prediction[0])
```

## Conclusion

This project demonstrates the process of building a linear regression model to predict insurance charges. The model shows good predictive power with an R-squared value around 0.75. Further improvements can be made by exploring other machine learning algorithms and feature engineering techniques.
