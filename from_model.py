import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.multioutput import MultiOutputRegressor
import pickle as pcl
import numpy as np


def predict_logistic(model: LogisticRegression, encoder: OrdinalEncoder, df: pd.DataFrame):
    return pd.DataFrame(columns=["does person smoke specified tobacco"],
                        data=model.predict(np.column_stack((encoder.transform(df[['state_name', 'gender']]), df['age'].values))))


def predict_linear(model: LinearRegression, encoder: OrdinalEncoder, df: pd.DataFrame):
    res = model.predict(encoder.transform(df[['state_name', 'gender', 'tobacco']]))
    return pd.DataFrame(columns=["age"],
                        data=res.round())


def predict_multilinear(model: MultiOutputRegressor, encoder: OrdinalEncoder, df: pd.DataFrame):
    res = model.predict(encoder.transform(df[['state_name', 'tobacco']])).round()
    df = pd.DataFrame(columns=["age", "gender"], data=res)
    df["gender"] = df["gender"].apply(lambda a: "Male" if a == 1. else "Female")
    return df


def predict_tree(model: DecisionTreeClassifier, encoder, df: pd.DataFrame):
    en_x, en_y = encoder
    cat_col = ['state_name', 'gender']
    X_data = np.column_stack((en_x.fit_transform(df[cat_col]), df['age'].values))
    res = model.predict(X_data)
    return pd.DataFrame({'tobacco': en_y.inverse_transform(res.round().reshape(-1, 1)).flatten()})


predictors = {
    'Linear': predict_linear,
    'MultiLinear': predict_multilinear,
    'Logistic': predict_logistic,
    'DecisionTree': predict_tree
}

inputs = {
    'Linear': {'state_name': ['Georgia'], 'gender': ['Male'], 'tobacco': ['Smokeless Tobacco']},
    'MultiLinear': {'state_name': ['Alaska'], 'tobacco': ['Pipe']},
    'Logistic': {'state_name': ['Alabama', 'Georgia'], 'gender': ['Female', 'Male'], 'age': [33, 55]},
    'DecisionTree': {'state_name': ['Georgia', 'Alabama'], 'gender': ['Female', 'Male'], 'age': [21, 55]}
}


def from_saved_model(path: str):
    def wrap_pred(pred, model, fe):
        return lambda df: pred(model, fe, df)

    with open(path, 'rb') as file:
        (model_name, model, feature_encoder) = pcl.load(file)
        return wrap_pred(predictors[model_name], model, feature_encoder)


def test_saved_model():
    file_name = "decision_tree_model"
    path = f"models/{file_name}"
    predict = from_saved_model(path)
    res = predict(pd.DataFrame(inputs['DecisionTree']))
    print(res)


test_saved_model()
