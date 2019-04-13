from neo4j import GraphDatabase, basic_auth
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
import pickle as pcl
import numpy as np

db_location = "bolt://localhost:7687"
username = "neo4j"
password = "qwerty"

query = """
        MATCH (p:Person)-[sm:SMOKES]->(t:Tobacco), (p)-[lv:LIVES]->(s:State)
        RETURN s.name as state_name, p.gender as gender, p.age as age, t.name as tobacco
        """


def get_dataset(tx):
    db_res = tx.run(query)
    training_data = pd.DataFrame([r.values() for r in db_res], columns=db_res.keys())
    return training_data


def is_tobacco_of(tobacco: str):
    def go(val: str):
        return tobacco == val
    return go


def logistic():
    db = GraphDatabase.driver(db_location, auth=basic_auth(username, password))
    feature_encoder = OrdinalEncoder()

    with db.session() as session:
        df = session.read_transaction(get_dataset)
        y_data = df['tobacco'].apply(is_tobacco_of('Cigar'))
        X_data = np.column_stack((feature_encoder.fit_transform(df.filter(items=['state_name', 'gender'])), df['age'].values))

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.10, random_state=42)
        logistic_model = LogisticRegression(solver='lbfgs')
        logistic_model.fit(X_train, y_train)
        serz = pcl.dumps(("Logistic", logistic_model, feature_encoder))
        with open('./models/logistic_model', 'wb') as out:
            print("Serializing model to the file.. ")
            out.write(serz)

        y_pred = logistic_model.predict(X_test)
        res = len(y_pred[y_pred == y_test]) / len(y_pred)
        print(f"Accuracy: {res}")


def linear():
    db = GraphDatabase.driver(db_location, auth=basic_auth(username, password))
    feature_encoder = OrdinalEncoder()
    linear_model = LinearRegression()

    with db.session() as session:
        df = session.read_transaction(get_dataset)
        y_data = df['age'].values
        X_data = feature_encoder.fit_transform(df.filter(items=['state_name', 'gender', 'tobacco']))

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.10, random_state=42)

        linear_model.fit(X_train, y_train)
        serz = pcl.dumps(("Linear", linear_model, feature_encoder))
        with open('./models/linear_model', 'wb') as out:
            print("Serializing model to the file.. ")
            out.write(serz)

        y_pred = linear_model.predict(X_test)
        res = {f'Accuracy for {i} years': len(y_pred[np.abs(y_pred - y_test) < i]) / len(y_pred) for i in range(1, 10, 2)}
        print(res)


def multilinear():
    db = GraphDatabase.driver(db_location, auth=basic_auth(username, password))
    fe = OrdinalEncoder()
    multilinear_model = MultiOutputRegressor(estimator=GradientBoostingRegressor(random_state=42))

    with db.session() as session:
        df = session.read_transaction(get_dataset)
        Y_data = np.column_stack((df['age'].values, fe.fit_transform(df[['gender']])))
        X_data = fe.fit_transform(df[['state_name', 'tobacco']])

        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.10, random_state=42)

        multilinear_model.fit(X_train, Y_train)
        serz = pcl.dumps(("MultiLinear", multilinear_model, fe))
        with open('./models/multilinear_model', 'wb') as out:
            print("Serializing model to the file.. ")
            out.write(serz)


def decision_tree():
    db = GraphDatabase.driver(db_location, auth=basic_auth(username, password))
    fe_y = OrdinalEncoder()
    fe_x = OrdinalEncoder()
    dt_model = DecisionTreeClassifier()

    with db.session() as session:
        df = session.read_transaction(get_dataset)
        cat_columns = ['state_name', 'gender']
        df[cat_columns] = fe_y.fit_transform(df[cat_columns])
        Y_data = fe_y.fit_transform(df['tobacco'].values.reshape(-1, 1))
        X_data = np.column_stack((fe_x.fit_transform(df[cat_columns]), df['age'].values))

        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.10, random_state=42)

        dt_model.fit(X_train, Y_train)
        serz = pcl.dumps(("DecisionTree", dt_model, (fe_x, fe_y)))
        with open('./models/decision_tree_model', 'wb') as out:
            print("Serializing model to the file.. ")
            out.write(serz)


decision_tree()

