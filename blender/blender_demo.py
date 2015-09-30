from __future__ import print_function
import pandas
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import *
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import naive_bayes
from sklearn import svm

class ModelTransformer(pipeline.TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pandas.DataFrame(self.model.predict(X))


predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

titanic = pandas.read_csv("../data/train.csv")
titanic_test = pandas.read_csv("../data/test.csv")


sex_encoder = LabelEncoder()

embark_encoder = LabelEncoder()
embark_imputer = Imputer(missing_values=0, strategy='most_frequent', axis=1)

age_imputer = Imputer(strategy='median', axis=1)

fare_imputer = Imputer(strategy='median', axis=1)


titanic["Sex"] = sex_encoder.fit_transform(titanic["Sex"])

titanic["Embarked"] = embark_encoder.fit_transform(titanic["Embarked"])
titanic["Embarked"] = embark_imputer.fit_transform(titanic["Embarked"]).T

titanic["Age"] = age_imputer.fit_transform(titanic["Age"]).T

titanic["Fare"] = fare_imputer.fit_transform(titanic["Fare"]).T

titanic_test["Sex"] = sex_encoder.transform(titanic_test["Sex"])

titanic_test["Embarked"] = embark_encoder.transform(titanic_test["Embarked"])
titanic_test["Embarked"] = embark_imputer.transform(titanic_test["Embarked"]).T

titanic_test["Age"] = age_imputer.transform(titanic_test["Age"]).T

titanic_test["Fare"] = fare_imputer.transform(titanic_test["Fare"]).T

extra = ModelTransformer(ExtraTreesClassifier(n_estimators=80,criterion='entropy'))
random_forest = ModelTransformer(RandomForestClassifier(n_estimators=50))
grad_boost = ModelTransformer(GradientBoostingClassifier(n_estimators=25, max_depth=5))
naive_bayes = ModelTransformer(naive_bayes.GaussianNB())
baby_svm = ModelTransformer(svm.SVC())

blender_1 = pipeline.make_union(extra, random_forest, grad_boost, naive_bayes, baby_svm)
blender_pipeline = pipeline.make_pipeline(blender_1, LogisticRegression())

blender_pipeline.fit(titanic[predictors], titanic["Survived"])

predictions = blender_pipeline.predict(titanic_test[predictors])

submission = pandas.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": predictions.astype(int)
    })

submission.to_csv("blender_submission.csv", index=False)
