import pandas
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score


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

alg = AdaBoostClassifier()
alg.fit(titanic_train[predictors], titanic_train["Survived"])

output = pandas.DataFrame(titanic_test["PassengerId"])
output["Survived"] = alg.predict(titanic_test[predictors])

print(output.head())

output.to_csv('kaggle.csv', index=False)
