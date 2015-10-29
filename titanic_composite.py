from __future__ import print_function
import pandas
import numpy as np
from sklearn.preprocessing import *
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import grid_search
from sklearn import linear_model
import xgboost as xgb

def GetLastName(fullname):
    return fullname.split(',')[0]

def GetTitle(fullname):
    return fullname.split('.')[0].split(', ')[1]


model = xgb.XGBClassifier(n_estimators=25)

#model_param_search = grid_search.GridSearchCV(model, parameters, cv=5)

titanic_train = pandas.read_csv("./data/train.csv")
titanic_test = pandas.read_csv("./data/test.csv")

titanic_train["FROM_TRAIN"] = 1
titanic_test["FROM_TRAIN"] = 0
titanic = pandas.concat([titanic_train, titanic_test])

age_model = ensemble.RandomForestRegressor(n_estimators=50)
age_predictors = ["Pclass", "Title", "SibSp", "Parch", "Fare"]


sex_encoder = LabelEncoder()
fare_imputer = Imputer(strategy='median', axis=1)
title_encoder = LabelEncoder()

familyID_encoder = LabelEncoder()

titanic["Embarked"] = titanic["Embarked"].fillna('S')

titanic["Fare"] = fare_imputer.fit_transform(titanic["Fare"]).T

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"] 

titanic["LastName"] = titanic["Name"].apply(GetLastName)

titanic["FamilyID"] = titanic["LastName"] + titanic["FamilySize"].astype(str)
print(titanic["FamilyID"].head())
titanic["FamilyID"] = familyID_encoder.fit_transform(titanic["FamilyID"])

titanic["Title"] = titanic["Name"].apply(GetTitle)
titanic["Title"] = titanic["Title"].replace(["Capt", "Col",  "Don", "Major", "Sir"], "Sir")
titanic["Title"] = titanic["Title"].replace(["Jonkheer", "Dona", "Mme", "the Countess", "Lady"], "Lady")
titanic["Title"] = titanic["Title"].replace(["Mlle", "Ms"], "Miss")

titanic["UpperClassWoman"] = 0
titanic.loc[(titanic["Sex"]==0) & (titanic["Parch"]==1),["UpperClassWoman"]] = 1 

titanic["LowerClassMale"] = 0
titanic.loc[(titanic["Sex"]==1) & (titanic["Parch"]==3),["LowerClassMale"]] = 1

titanic["Mother"] = 0
titanic.loc[(titanic["Sex"] == 0) & (titanic["Parch"] > 0) & (titanic["Age"] > 18) & (titanic["Title"] != "Miss"), ["Mother"]] = 1

titanic["Child"] = 0
titanic.loc[(titanic["Parch"] > 0) & (titanic["Age"] <= 18), ["Child"]] = 1

titanic["Title"] = title_encoder.fit_transform(titanic["Title"])

age_model.fit(titanic[titanic["Age"].notnull()][age_predictors], titanic[titanic["Age"].notnull()]["Age"])

titanic.loc[titanic["Age"].isnull(),["Age"]] = age_model.predict(titanic[titanic["Age"].isnull()][age_predictors])

titanic = pandas.get_dummies(titanic, prefix=["Title","Embarked","Sex"], columns=["Title","Embarked","Sex"])

titanic = titanic.drop(["Cabin", "Name", "Ticket", "LastName"], axis=1)

titanic_train = titanic[titanic["FROM_TRAIN"] == 1]
titanic_test = titanic[titanic["FROM_TRAIN"] == 0]

titanic_train = titanic_train.drop(["FROM_TRAIN"], axis=1)
titanic_test = titanic_test.drop(["FROM_TRAIN"], axis=1)

scores = cross_validation.cross_val_score(model, titanic_train.drop(["PassengerId", "Survived"], axis=1).as_matrix(), titanic_train["Survived"].as_matrix(), cv=6)


model.fit(titanic_train.drop(["PassengerId", "Survived"], axis=1).as_matrix(), titanic_train["Survived"].as_matrix())

print(scores)
print(scores.mean())

predictions = model.predict(titanic_test.drop(["PassengerId", "Survived"], axis=1).as_matrix())
predictions = pandas.Series(predictions)
submission = pandas.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": predictions.astype(int)
    })

printout=False
if printout:
    submission.to_csv("titanic_submission.csv", index=False)
