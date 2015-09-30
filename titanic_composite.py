from __future__ import print_function
import pandas
import numpy as np
from sklearn.preprocessing import *
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import linear_model

def GetLastName(fullname):
    return fullname.split(',')[0]

def GetTitle(fullname):
    return fullname.split('.')[0].split(', ')[1]


titanic_train = pandas.read_csv("./data/train.csv")
titanic_test = pandas.read_csv("./data/test.csv")

titanic_train["FROM_TRAIN"] = 1
titanic_test["FROM_TRAIN"] = 0
titanic = pandas.concat([titanic_train, titanic_test])

age_model = ensemble.RandomForestRegressor(n_estimators=50)
age_predictors = ["Pclass", "Title", "Sex", "SibSp", "Parch", "Fare"]

model = ensemble.RandomForestClassifier(n_estimators=100)


sex_encoder = LabelEncoder()
fare_imputer = Imputer(strategy='median', axis=1)
title_encoder = LabelEncoder()


titanic["Sex"] = sex_encoder.fit_transform(titanic["Sex"])

titanic["Embarked"] = titanic["Embarked"].fillna('S')

titanic["Fare"] = fare_imputer.fit_transform(titanic["Fare"]).T

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"] 

titanic["LastName"] = titanic["Name"].apply(GetLastName)

titanic["Title"] = titanic["Name"].apply(GetTitle)

titanic["Title"] = titanic["Title"].replace(["Capt", "Col",  "Don", "Major", "Sir"], "Sir")
titanic["Title"] = titanic["Title"].replace(["Jonkheer", "Dona", "Mme", "the Countess", "Lady"], "Lady")
titanic["Title"] = titanic["Title"].replace(["Mlle", "Ms"], "Miss")

titanic["Mother"] = 0
titanic.loc[(titanic["Sex"] == 0) & (titanic["Parch"] > 0) & (titanic["Age"] > 18) & (titanic["Title"] != "Miss"), ["Mother"]] = 1

titanic["Child"] = 0
titanic.loc[(titanic["Parch"] > 0) & (titanic["Age"] <= 18), ["Child"]] = 1

titanic["Title"] = title_encoder.fit_transform(titanic["Title"])

age_model.fit(titanic[titanic["Age"].notnull()][age_predictors], titanic[titanic["Age"].notnull()]["Age"])

titanic.loc[titanic["Age"].isnull(),["Age"]] = age_model.predict(titanic[titanic["Age"].isnull()][age_predictors])

titanic = pandas.get_dummies(titanic, prefix=["Title", "Embarked"], columns=["Title","Embarked"])

titanic = titanic.drop(["Cabin", "Name", "Ticket", "LastName"], axis=1)

titanic_train = titanic[titanic["FROM_TRAIN"] == 1]
titanic_test = titanic[titanic["FROM_TRAIN"] == 0]

titanic_train = titanic_train.drop(["FROM_TRAIN"], axis=1)
titanic_test = titanic_test.drop(["FROM_TRAIN"], axis=1)

scores = cross_validation.cross_val_score(model, titanic_train.drop(["PassengerId", "Survived"], axis=1), titanic_train["Survived"], cv=6)


model.fit(titanic_train.drop(["PassengerId", "Survived"], axis=1), titanic_train["Survived"])
predictions = model.predict(titanic_test.drop(["PassengerId", "Survived"], axis=1))

print(titanic.loc[(titanic["Mother"] == 1) & (titanic["Survived"] == 0)])

"""
submission = pandas.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": predictions.astype(int)
    })

submission.to_csv("titanic_submission.csv", index=False)
"""
