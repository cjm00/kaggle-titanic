from __future__ import print_function
import pandas
import numpy as np
from sklearn.preprocessing import *
from sklearn import cross_validation
from sklearn import svm
from sklearn import grid_search
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors

def GetLastName(fullname):
    return fullname.split(',')[0]

def GetTitle(fullname):
    return fullname.split('.')[0].split(', ')[1]


titanic_train = pandas.read_csv("../data/train.csv")
titanic_test = pandas.read_csv("../data/test.csv")

titanic_train["FROM_TRAIN"] = 1
titanic_test["FROM_TRAIN"] = 0
titanic = pandas.concat([titanic_train, titanic_test])


model = ensemble.RandomForestClassifier(n_estimators=80, criterion='entropy')

sex_encoder = LabelEncoder()

embark_encoder = LabelEncoder()
embark_imputer = Imputer(missing_values=0, strategy='most_frequent', axis=1)

age_imputer = Imputer(strategy='median', axis=1)
age_scaler = MinMaxScaler()

fare_imputer = Imputer(strategy='median', axis=1)
fare_scaler = MinMaxScaler()

family_scaler = MinMaxScaler()

titanic["Sex"] = sex_encoder.fit_transform(titanic["Sex"])

titanic["Embarked"] = titanic["Embarked"].fillna('S')


titanic["Fare"] = fare_imputer.fit_transform(titanic["Fare"]).T
titanic["Fare"] = fare_scaler.fit_transform(titanic["Fare"])

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"] 
titanic["FamilySize"] = family_scaler.fit_transform(titanic["FamilySize"].astype(float))

titanic["LastName"] = titanic["Name"].apply(GetLastName)

titanic["Title"] = titanic["Name"].apply(GetTitle)

titanic["Title"] = titanic["Title"].replace(["Capt", "Col",  "Don", "Major", "Sir"], "Sir")
titanic["Title"] = titanic["Title"].replace(["Jonkheer", "Dona", "Mme", "the Countess", "Lady"], "Lady")
titanic["Title"] = titanic["Title"].replace(["Mlle", "Ms"], "Miss")

#titanic = pandas.get_dummies(titanic, prefix=["Title", "Embarked"], columns=["Title","Embarked"])


titanic = titanic.drop(["Cabin", "Name", "Ticket", "LastName"], axis=1)

titanic_train = titanic[titanic["FROM_TRAIN"] == 1]
titanic_test = titanic[titanic["FROM_TRAIN"] == 0]

titanic_train = titanic_train.drop(["FROM_TRAIN"], axis=1)
titanic_test = titanic_test.drop(["FROM_TRAIN"], axis=1)



print(titanic["Title"].value_counts())

"""
submission = pandas.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": predictions.astype(int)
    })

submission.to_csv("titanic_svc_submission.csv", index=False)
"""
