import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import auc, roc_curve

import logging

trainMnist: pd.DataFrame = pd.read_csv("mnist_train.csv")
testMnist: pd.DataFrame = pd.read_csv("mnist_test.csv")

# print(trainMnist.head(5))
# print(testMnist.head(5))

X_train: np.ndarray = trainMnist.drop('label', axis=1).to_numpy()
y_train: np.ndarray = trainMnist['label'].to_numpy()
X_test: np.ndarray = testMnist.drop('label', axis=1).to_numpy()
y_test: np.ndarray = testMnist['label'].to_numpy()

# Set up logging
logging.basicConfig(filename='mnist_data_log.txt', level=logging.INFO)

# Logging train data information
logging.info("Train Data:")
logging.info(trainMnist.head())
logging.info(f"Number of samples in train data: {len(trainMnist)}")
logging.info(f"Structure of train data:\n{trainMnist.info()}")

# Logging test data information
logging.info("Test Data:")
logging.info(testMnist.head())
logging.info(f"Number of samples in test data: {len(testMnist)}")
logging.info(f"Structure of test data:\n{testMnist.info()}")


# Few sample handwritten digits with label description in the title
fig,ax=plt.subplots(2,5, figsize=(7,5))
for i, ax in enumerate(ax.flatten()):
    ax.imshow(X_train[i].reshape((28, 28)), cmap="gray")
    ax.set_title(y_train[i])
    fig.tight_layout()
plt.show()

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

''' Bernoulli distribution'''
modelBN = BernoulliNB()

fin_nb = modelBN.fit(X_train, y_train)

# confusion matrix for x test
predictionNB_1 = fin_nb.predict(X_test)
con_matrixNB = confusion_matrix(y_test, predictionNB_1)
print(con_matrixNB)
print(accuracy_score(predictionNB_1, y_test))

# confusion matrix for x_train
predictionNB_2 = fin_nb.predict(X_train)
con_matrixNB_2 = confusion_matrix(y_train, predictionNB_2)
print(con_matrixNB_2)
print(accuracy_score(predictionNB_2, y_train))

# heatmap for x test prediction
plt.imshow(con_matrixNB, cmap="inferno", interpolation="nearest")
plt.xlabel("Predictions 1")
plt.ylabel("Actual Values")
plt.show()

# heatmap for x train predictions
plt.imshow(con_matrixNB_2, cmap="inferno", interpolation="nearest")
plt.xlabel("Predictions 2")
plt.ylabel("Actual Values")
plt.show()

# y_predictProb = modelBN.predict_proba(X_test)
from sklearn.metrics import auc, roc_curve

# fpr, tpr, thresholds = roc_curve(y_test, y_predictProb[::, 1])
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, color='darkorange', label='ROC curve - Beroulli (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# # plt.show()


''' Gaussian distribution'''
modelGS = GaussianNB()
fin_nb = modelGS.fit(X_train, y_train)

# confusion matrix for x test
predictionGS_1 = fin_nb.predict(X_test)
con_matrixGS = confusion_matrix(y_test, predictionGS_1)
print(con_matrixGS)
print(accuracy_score(predictionGS_1, y_test))

# confusion matrix for x_train
predictionGS_2 = fin_nb.predict(X_train)
con_matrixGS_2 = confusion_matrix(y_train, predictionGS_2)
print(con_matrixGS_2)
print(accuracy_score(predictionGS_2, y_train))

# heatmap for x test prediction
plt.imshow(con_matrixGS, cmap="inferno", interpolation="nearest")
plt.xlabel("Predictions 1")
plt.ylabel("Actual Values")
plt.show()

# heatmap for x train predictions
plt.imshow(con_matrixGS_2, cmap="inferno", interpolation="nearest")
plt.xlabel("Predictions 2")
plt.ylabel("Actual Values")
plt.show()

# y_predictProb = modelGS.predict_proba(X_test)
# from sklearn.metrics import auc, roc_curve
#
# fpr, tpr, thresholds = roc_curve(y_test, y_predictProb[::, 1], pos_label="Gaussian")
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, color='darkred', label='ROC curve - Gaussian (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic - gaussian')
# plt.legend(loc="lower right")
# plt.show()


'''Error rate '''
def calculate_error_rate(conf_matrix):
    total_samples = np.sum(conf_matrix)
    correct_predictions = np.trace(conf_matrix)
    error_rate = 1 - (correct_predictions / total_samples)
    return error_rate


# Error rate for Bernoulli Naive Bayes
error_rate_BN_test = calculate_error_rate(con_matrixNB)
error_rate_BN_train = calculate_error_rate(con_matrixNB_2)
print("Bernoulli Naive Bayes:")
print(f"Test Error Rate: {error_rate_BN_test}")
print(f"Train Error Rate: {error_rate_BN_train}")

# Error rate for Gaussian Naive Bayes
error_rate_GS_test = calculate_error_rate(con_matrixGS)
error_rate_GS_train = calculate_error_rate(con_matrixGS_2)
print("Gaussian Naive Bayes:")
print(f"Test Error Rate: {error_rate_GS_test}")
print(f"Train Error Rate: {error_rate_GS_train}")
