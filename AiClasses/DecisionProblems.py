import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

wine_dataset = pd.read_csv('winequality-red.csv')


"""Data Analysis"""
print(wine_dataset.head())
print(wine_dataset.info())
high_quality_wine = wine_dataset[wine_dataset['quality'] > 6]
print(high_quality_wine.describe(), '\n')

#citric acid vs Quality
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='citric acid', data=wine_dataset)

#volatile acidity vs Quality
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)

#sulphates vs Quality
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='sulphates', data=wine_dataset)

#alcohol vs Quality
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='alcohol', data=wine_dataset)

#heatmap for features correlation
correlation=wine_dataset.corr()
plt.figure(figsize=(7,7))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True,
            annot_kws={'size':8}, cmap='Blues')


"""Decision Model"""
def classify_wine_quality(row):
    if row['quality']>6:
        return 1  # 1 - Good wine
    else:
        return 0  # 0 - Bad wine

def predict_wine_quality(row):
    if (row['alcohol'] > 11 and row['volatile acidity'] < 0.5 and row['citric acid'] > 0.2
            and row['sulphates']>0.6):
        return 1  # 1 - Good wine
    else:
        return 0  #0 - Bad wine

wine_dataset['quality_classified'] = wine_dataset.apply(classify_wine_quality, axis=1)
wine_dataset['predicted_quality'] = wine_dataset.apply(predict_wine_quality, axis=1)
wine_dataset.to_csv("winequality-red-with-predictions.csv", index=False)


"""Accuracy"""
correct_predictions = (wine_dataset['quality_classified'] ==
                       wine_dataset['predicted_quality']).sum()
total_predictions = len(wine_dataset)
accuracy = correct_predictions / total_predictions
print(f"Accuracy score:, {round(accuracy*100, 1)}%")

plt.show()
