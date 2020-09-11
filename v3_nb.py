from dataset_manager_v3 import DatasetManager
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB
from sklearn.metrics import classification_report



if __name__ == "__main__":
        dm = DatasetManager()
        x_train, x_test, y_train, y_test = dm.get_seasonal_split(2018, 0, 3, 5, True, False, False)

        gaus = GaussianNB().fit(x_train, y_train)
        mult = MultinomialNB().fit(x_train, y_train)
        bern = BernoulliNB().fit(x_train, y_train)
        comp = ComplementNB().fit(x_train, y_train)
        
        print("GAUSSIAN")
        print(classification_report(y_train, gaus.predict(x_train)))
        print(classification_report(y_test, gaus.predict(x_test)))

        print("MULTINOMIAL")
        print(classification_report(y_train, mult.predict(x_train)))
        print(classification_report(y_test, mult.predict(x_test)))

        print("BERNOULLI")
        print(classification_report(y_train, bern.predict(x_train)))
        print(classification_report(y_test, bern.predict(x_test)))

        print("COMPLEMENT")
        print(classification_report(y_train, comp.predict(x_train)))
        print(classification_report(y_test, comp.predict(x_test)))
