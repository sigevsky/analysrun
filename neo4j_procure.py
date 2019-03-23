from neo4j import GraphDatabase, basic_auth
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle as pcl

db_location = "bolt://localhost:7687"
username = "neo4j"
password = "qwerty"

query = """
        MATCH (p:Person)-[sm:SMOKES]->(t:Tobacco), (p)-[lv:LIVES]->(s:State)
        RETURN p.gender as gender, p.age as age, t.name as tobacco
        """


def get_dataset(tx):
    db_res = tx.run(query)
    training_data = pd.DataFrame([r.values() for r in db_res], columns=db_res.keys())
    return training_data


def categorize(df, columns):
    for c in columns:
        df[c] = df[c].astype("category").cat.codes
    return df


def is_tobacco_of(tobacco: str):
    def go(val: str):
        return tobacco == val
    return go


def main():
    db = GraphDatabase.driver(db_location, auth=basic_auth(username, password))

    with db.session() as session:
        df = session.read_transaction(get_dataset)
        df['tobacco'] = df['tobacco'].apply(is_tobacco_of('Smokeless Tobacco'))
        df = categorize(df, ['gender', 'tobacco'])
        X_data = df.filter(items=['gender', 'age'])
        y_data = df["tobacco"]

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.10, random_state=42)
        logistic_model = LogisticRegression(solver='lbfgs')
        logistic_model.fit(X_train, y_train)
        serz = pcl.dumps(logistic_model)
        with open('logistic_model', 'wb') as out:
            print("Serializing model to the file.. ")
            out.write(serz)

        y_pred = logistic_model.predict(X_test)
        res = len(y_pred[y_pred == y_test]) / len(y_pred)
        print(f"Accuracy: {res}")


main()
