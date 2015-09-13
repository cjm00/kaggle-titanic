import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold

def SexToNum(string):
    if string == 'male':
        return 1
    elif string == 'female':
        return 0

ports = ["S", "C", "Q"]
ports_col = ["EmbarkedS", "EmbarkedC", "EmbarkedQ"]

titanic_train = pandas.read_csv('data/train.csv')
titanic_test = pandas.read_csv('data/test.csv')

titanic_train["Sex"] = titanic_train['Sex'].map(SexToNum)
titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())

titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")

titanic_test["Sex"] = titanic_test['Sex'].map(SexToNum)
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

for port in ports_col:
    titanic_train[port] = pandas.Series([0] * len(titanic_train), index = titanic_train.index)
    titanic_test[port] = pandas.Series([0] * len(titanic_test), index = titanic_test.index)


for port, port_col in zip(ports, ports_col):
    titanic_train.loc[titanic_train["Embarked"] == port, port_col] = 1
    titanic_test.loc[titanic_test["Embarked"] == port, port_col] = 1



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "EmbarkedS", "EmbarkedC", "EmbarkedQ"]

algorithms = [RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4,min_samples_leaf=2),             LogisticRegression(random_state=1)]

kf = KFold(titanic_train.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic_train["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic_train[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic_train[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == titanic_train["Survived"]]) / len(predictions)
print(accuracy)
