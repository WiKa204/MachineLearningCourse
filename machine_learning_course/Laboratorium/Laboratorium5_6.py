import os
import time
import random
import numpy as np
import pandas as pd
import mlflow.sklearn
import missingno as msno
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from random import random, randint
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlflow import log_metric, log_param, log_artifacts

titanic = pd.read_csv("./_data/phpMYEkMl.csv")

def predict(x):
    return [1 if random.random() < 0.5 else 0 for i in np.array(x)]

# ZADANIE 1 Załaduj bazę danych Titanic. Wykorzystaj ładowanie jako pandas.DataFrame i
# np. meotdę info oraz describe. Sprawdź jakie wartości przyjmują poszczególne cechy.
def zad1():
    global titanic
    table = pd.DataFrame(data=titanic)
    # print(table)
    info = table.info(verbose=True)    # verbose=True - to print all info
    # print(info)
    opis = table.describe()
    print(opis)

# ZADANIE 2 Usuń  kolumny: boat, body, home.dest (dużo brakujących wartości).
def zad2():
    global titanic
    titanic.drop(['boat', 'body', 'home.dest', 'cabin', 'ticket', 'name'], axis=1, inplace=True)

    titanic = titanic.rename(columns={'pclass': 'TicketClass'})
    # print(titanic)

# ZADANIE 3 Podziel dane na zbiór treningowy i testowy.
def zad3():
    zad2()

    X_train, X_test, y_train, y_test = train_test_split(titanic.drop(columns=['survived'], axis=1), titanic['survived'],
                                        random_state=42, stratify=titanic['survived'], test_size=0.1)

# ZADANIE 4 Opracuj metodę, która dla danych testowych wyznacza szansę przeżycia
def zad4():
    print(f'Szansa na przeżycie poszczególnych osób wynosi : {predict(titanic)} ')

# ZADANIE 5 Przeszkuja bazę danych pod kątem brakujących wartości.
# Wyznacz ile dokładnie brakuje
# Spróbuj uzupełnić brakujące wartości.
def zad56_():
    zad2()
    X_train, X_test, y_train, y_test = train_test_split(titanic.drop(columns=['survived'], axis=1), titanic['survived'],
                                                        random_state=42, stratify=titanic['survived'], test_size=0.1)
    # print(titanic.loc[titanic['embarked'] == '?', 'name'])
    titanic.replace('?', np.NaN, inplace=True)
    # msno.bar(X_train, filter='?')
    # plt.show()
    # msno.matrix(X_train)
    # plt.show()
    titanic['embarked'].iloc[168] = 'S'
    titanic['embarked'].iloc[284] = 'S'
    titanic.drop(1225, inplace=True)
    titanic['age'] = pd.to_numeric(titanic['age'], downcast='float')
    # titanic.loc[titanic['TicketClass'], 'age'].hist()
    # titanic.loc[titanic["pclass"]==3, 'age'].hist()
    # plt.show()
    ages_titanic = titanic.groupby(['sex', 'TicketClass'])['age'].median().round(1)
    # print(ages_titanic)

    for row, passenger in titanic.loc[np.isnan(titanic['age'])].iterrows():
        titanic['age'].iloc[row] = ages_titanic[passenger.sex][passenger.TicketClass]
        # titanic['age'].iloc[row] = ages_titanic[passenger['sex']][passenger['TicketClass']]
        # print(passenger.sex)
        if np.isnan(titanic['age'].iloc[row]):
            print(ages_titanic[passenger.sex][passenger.TicketClass])
            # print(ages_titanic[passenger['sex']][passenger['TicketClass']])


    titanic.dropna(inplace=True)
    # msno.bar(titanic)
    # plt.show()

# ZADANIE 7 zamień wszystkie cechy na liczbowe.
    titanic['sex'].loc[titanic['sex'] == 'female'] = 1
    titanic['sex'].loc[titanic['sex'] == 'male'] = 0

    titanic['family_size'] = titanic['sibsp'] + titanic['parch']
    titanic.drop(['sibsp', 'parch', 'embarked'], axis=1, inplace=True)

    titanic['TicketClass'].loc[titanic['TicketClass'] == 2] = 0.5
    titanic['TicketClass'].loc[titanic['TicketClass'] == 3] = 0

    titanic['sex'] = titanic['sex'].astype(float)
    titanic['fare'] = titanic['fare'].astype(float)
    titanic['TicketClass'] = titanic['TicketClass'].astype(float)
    titanic['family_size'] = titanic['family_size'].astype(float)

# ZADANIE 8 wytrenuj wybrany klasyfikator i
# oceń go względem przygotowanego wcześniej rozwiązania bazowego.
    X_train, X_test, y_train, y_test = train_test_split(titanic.drop(columns=['survived'], axis=1), titanic['survived'],
                                                        random_state=42, stratify=titanic['survived'], test_size=0.1)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

#####################################################################################
#                               LAB 6
#####################################################################################

# ZADANIE 345 Wykorzystaj kod z jednych z poprzednich zajęć z trenowaniem
# i dodaj do niego autologowanie dla wykorzystywanej biblioteki.
# Sprawdź co dokładnie zostało zapisane w wyniku przeprowadzonego eksperymentu.
# Ustaw własną nazwę eksperymentu.
    mlflow.sklearn.autolog()
    clfs = [RandomForestClassifier(), SVC(), LinearRegression()]
    for clf in clfs:
        with mlflow.start_run(run_name=type(clf).__name__):
            start_time = time.perf_counter()
            clf.fit(X_train, y_train)
            duration = time.perf_counter() - start_time
            scr = clf.score(X_test, y_test)
            mlflow.log_metric('score', scr)
            mlflow.log_metric('duration', duration)



def main():
    # zad1()
    # zad2()
    # zad3()
    # zad4()
    zad56_()


if __name__ == '__main__':
    main()
    # LAB 6 ZADANIE 2
    for run in ['a', 'b', 'c']:
        with mlflow.start_run(run_name=run):
            # Log a parameter (key-value pair)
            log_param("param1", randint(0, 100))

            # Log a metric; metrics can be updated throughout the run
            log_metric("foo", random())
            log_metric("foo", random() + 1)
            log_metric("foo", random() + 2)

            # Log an artifact (output file)
            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            with open("outputs/test.txt", "w") as f:
                f.write("hello world!")
            log_artifacts("outputs")


