from neo4j import GraphDatabase, basic_auth
import pandas as pd
from faker import Faker
from name_provider import Provider

db_location = "bolt://localhost:7687"
username = "neo4j"
password = "qwerty"


def form_data_set(df: pd.DataFrame):
    df = df[df["TopicType"] == "Tobacco Use – Survey Data"]
    df = df[df["MeasureDesc"] == "Smoking Status"]
    df = df[df["YEAR"] == "2014-2015"]
    df = df[df["Gender"] != "Overall"]
    df = df[df["Data_Value_Footnote_Symbol"] != "Data in these cells have been suppressed because of a small sample size"]
    df = df.drop_duplicates(subset=["LocationDesc", "TopicDesc", "Gender"])
    df["Tobacco"] = df["TopicDesc"].apply(lambda x: x.replace(" Use (Adults)", ""))
    return df


def save_states(tx, states: pd.DataFrame):
    for _, state in states.iterrows():
        tx.run("MERGE (State {name: $name, abbr: $abbr})",
               name=state["LocationDesc"], abbr=state["LocationAbbr"])


def save_tobacco(tx, tobaccos: pd.Series):
    for tobacco in tobaccos:
        tx.run("MERGE (Tobacco {name: $name})", name=tobacco)


def save_persons(tx, poll: pd.DataFrame, fake: Faker):
    amount = int(poll["Sample_Size"] / 50)
    for _ in range(amount):
        name = fake.female_name() if poll["Gender"] == "female" else fake.male_name()
        tx.run("""MATCH (s:State)
                  WHERE s.name = $state
                  CREATE (p:Person {name: $name, gender: $gender})
                  CREATE (p)-[:LIVES]->(s)""", state=poll["LocationDesc"], name=name, gender=poll["Gender"])
        if poll["Response"] != "Never":
            tx.run("""MATCH (p:Person), (t:Tobacco)
                      WHERE p.name = $name AND t.name=$tobac
                      CREATE (p)-[:SMOKED {status:$status}]->(t)""",
                   tobac=poll["Tobacco"], name=name, status=poll["Response"])


def main():
    fake = Faker()
    fake.add_provider(Provider)
    db = GraphDatabase.driver(db_location, auth=basic_auth(username, password))

    raw_df = pd.read_csv("./data/tobacco.csv")
    df = form_data_set(raw_df)

    tobaccos = df["TopicDesc"].unique()
    states = df[["LocationDesc", "LocationAbbr"]].drop_duplicates()

    with db.session() as session:
        session.write_transaction(save_states, states)
        session.write_transaction(save_tobacco, tobaccos)

        for i, poll in df.iterrows():
            print("Saved %d" % i)
            session.write_transaction(save_persons, poll, fake)


main()
