import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.multioutput import MultiOutputRegressor
import pickle as pcl
import numpy as np


def predict_logistic(model: LogisticRegression, encoder: OrdinalEncoder, df: pd.DataFrame):
    return model.predict(np.column_stack((encoder.transform(df[['state_name', 'gender']]), df['age'].values)))


def predict_linear(model: LinearRegression, encoder: OrdinalEncoder, df: pd.DataFrame):
    return model.predict(encoder.transform(df[['state_name', 'gender', 'tobacco']]))


def predict_multilinear(model: MultiOutputRegressor, encoder: OrdinalEncoder, df: pd.DataFrame):
    return model.predict(encoder.transform(df[['state_name', 'tobacco']]))


predictors = {
    'Linear': predict_linear,
    'MultiLinear': predict_multilinear,
    'Logistic': predict_logistic
}


def from_saved_model(path: str):
    def wrap_pred(pred, model, fe):
        return lambda df: pred(model, fe, df)

    with open(path, 'rb') as file:
        (model_name, model, feature_endcoder) = pcl.load(file)
        return wrap_pred(predictors[model_name], model, feature_endcoder)


def test_saved_model():
    file_name = "multilinear_model"
    path = f"models/{file_name}"
    predict = from_saved_model(path)
    res = predict(pd.DataFrame({'state_name': ['Alaska'], 'tobacco': ['Pipe']}))
    print(res)


test_saved_model()
