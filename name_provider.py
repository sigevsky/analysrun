from faker.providers import BaseProvider
import pandas as pd


class Provider(BaseProvider):

    male_names = pd.read_csv("./data/generator/male_names.csv")["name"].tolist()
    female_names = pd.read_csv("./data/generator/female_names.csv")["name"].tolist()
    surnames = pd.read_csv("./data/generator/surnames.csv")["surname"].tolist()

    def female_name(self):
        return self.random_element(self.female_names)

    def male_name(self):
        return self.random_element(self.male_names)

    def surname(self):
        return self.random_element(self.surnames)
