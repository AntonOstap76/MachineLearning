'''Importing Dependencies'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

'''Data Collection'''
wine_dataset = pd.read_csv('winequality-red.csv')

# numbers of rows and columns in dataset
# print(wine_dataset.shape)

# first 5 row of the dataset
pd.set_option('display.max_columns', None)
print(wine_dataset.head())

# checking for missing values
# print(wine_dataset.isnull().sum())

'''Data Analysis and Visualization'''

# statistical measures of the dataset
# print(wine_dataset.describe())

# number of values for each quality
# sns.catplot(x='quality', data=wine_dataset, kind='count')

# # volatile acidity vs Quality
# plot = plt.figure(figsize=(5, 5))
# sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)

# #citric acid vs Quality
plot = plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='citric acid', data=wine_dataset)

'''Correlation'''
correlation = wine_dataset.corr()

# constructing a heat map to understand the  correlation between columns
plt.figure(figsize=(15, 15))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')

'''Data Preprocessing'''
# separate the data and Label
X = wine_dataset.drop('quality', axis=1)
# print(X)

# making Label binary
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value > 7 else 0)
# print(Y)

'''Train & Test Split'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# # showing  amount of original data, test data, and train data
# print(Y.shape, Y_test.shape, Y_train.shape)

'''Model Training'''
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

'''Model Evaluation'''
'''Accuracy Score'''
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f"Accuracy: {test_data_accuracy}")


"""Predictive System at work"""

def get_random_wine_parameteres():
    np.random.seed(42)
    indexes=X.shape[0]
    index=np.random.randint(indexes)
    random_wine_parameteres=np.array(X.iloc[index])
    return random_wine_parameteres

def predicting_if_wine_good():
    data_reshaped = get_random_wine_parameteres().reshape(1, -1)
    prediction = model.predict(data_reshaped)
    if (prediction[0] == 1):
        print('GOOD Quality Wine')
    else:
        print('BAD Quality Wine')

predicting_if_wine_good()

"""Visualizing the trees"""

plt.figure(figsize=(15, 7))
plot_tree(model, filled=True, feature_names=X.columns, max_depth=3, fontsize=7,
          class_names=['Bad Wine', 'Good Wine'])
plt.title('Train Tree')
plt.show()