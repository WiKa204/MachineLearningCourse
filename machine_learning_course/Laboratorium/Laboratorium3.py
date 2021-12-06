import csv
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures


# ZADANIE 1 DO 1 Wykorzystaj problem dotyczący ładowania baterii
# wyświetl statystki danych - czy wyjaśniają dane i pozwalają dobrać właściwy model?
# jakiego modelu użyłeś na poprzednich zajęciach?
# wytrenuj modele bazujące na regresji liniowej, nieliniowej i drzewie decyzyjnym, porównaj wyniki.
# wykorzystaj bardziej złożony estymator typu las losowy.
def zad1():
    data = []
    with open('./_data/trainingdata.txt', 'r') as csv_f:
        csv_reader = csv.reader(csv_f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_reader:
            data.append(row)
    data = np.array(data)
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(x, y)
    pred = reg.predict(x)

    model = Pipeline([('poly', PolynomialFeatures(degree=10)),
                      ('linear', LinearRegression(fit_intercept=False))])
    model.fit(x, y)
    poly_pred = model.predict(x)

    plt.plot(x, y, 'ro')
    plt.plot(x, pred, 'bo')
    plt.plot(x, poly_pred, 'go')
    plt.show()

X, y = datasets.load_iris(return_X_y=True, as_frame=True)

# ZADANIE 2
# Załaduj zbiór danych i wyświetl informacje o nim. Załaduj jako pandas dataframe
# (argument as_frame=True) i wykorzystaj metodę describe.
# Czy możemy uzyskać jakieś wartościowe informacje?
def zad2():
    print(f'Describe: \n {X.describe()}')
    print(f'Head: \n {X.head()}')

# ZADANIE 3 i 4 Podziel wczytane dane na zbiór treningowy i testowy, w proporcjach 80%/20%.
# sprawdź, jakie jest procentowe rozłożenie poszczególnych klas w zbiorze treningowym i testowym.
# Dobrze by dystrybucje klas próbek w tych zbiorach były identyczne
# zmodyfikuj poprzedni kod tak, żeby dane po podziale spełniały ten warunek
def zad3_4():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train = X_train[['sepal length (cm)', 'sepal width (cm)']]
    X_test = X_test[['sepal length (cm)', 'sepal width (cm)']]

    print(y_train.value_counts() / len(y_train) * 100)

# ZADANIE 5 Wykorzystaj przedstawioną funkcjonalność do normalizacji danych.
# Wygeneruj wykres, którym sprawdzisz, jak wyglądają wartości danych po przeskalowaniu
# (ogranicz się do dwóch cech datasetu Iris).
# Jakie według Ciebie powinny być przedziały poszczególnych cech?
# Jakie są one w rzeczywistości (czy pokrywają się z Twoim wyobrażeniem)?
# Czy poszczególne cechy znajdują się na tej samej skali?
def zad5_6():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train = X_train[['sepal length (cm)', 'sepal width (cm)']]
    X_test = X_test[['sepal length (cm)', 'sepal width (cm)']]

    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.show()

    skaler = MinMaxScaler()
    # skaler = StandardScaler()

    skaler.fit(X_train)
    X_train = skaler.transform(X_train)

    plt.scatter(np.array(X_train)[:, 0],
                np.array(X_train)[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.show()

# ZADANIE 7 Zastosuj pipeline do aplikacji zrealizowane funkcji normalizacji.
# ZADANIE 8 Wytrenuj klasyfikator dla bazy danych Iris.
# ZADANIE 9 wizualizacja przestrzeni decyzyjnej
def zad789():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train = X_train[['sepal length (cm)', 'sepal width (cm)']]
    X_test = X_test[['sepal length (cm)', 'sepal width (cm)']]

    clf = Pipeline([
        ('skaler', MinMaxScaler()),
        ('svc', SVC())
    ])

    clf.fit(X_train, y_train)
    src = clf.score(X_test, y_test)
    print(f'score : {src}')
    plot_decision_regions(np.array(X_test), np.array(y_test), clf=clf, legend=1)

    print(clf.predict(X_test)[:5])
    plt.scatter(np.array(X_train)[:, 0], np.array(X_train)[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.show()

# ZADANIE 10 Napisz kod, który przeprowadzi trening na klasyfikatorach
# (LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier),
# przeprowadzi predykcję/sprawdzi dokładność na zbiorze testowym.
# Wyniki poszczególnych algorytmów zapisz w słowniku.
def zad10_11():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train = X_train[['sepal length (cm)', 'sepal width (cm)']]
    X_test = X_test[['sepal length (cm)', 'sepal width (cm)']]


    klasyfikatory = ["LogisticRegression", "SVC", "DecisionTreeClassifier", "RandomForestClassifier"]

    wyniki = dict()
    for classifier in klasyfikatory:
        clf = Pipeline([
            ('skaler', MinMaxScaler()),
            (classifier, globals()[classifier]())
        ])
        clf.fit(X_train, y_train)
        print(clf.score(X_test, y_test))

    print(globals()["SVC"])
    a = globals()["SVC"]()
    a.fit(X_train, y_train)
    print(a.score(X_test, y_test))

    # pętla przechodząca po klasyfikatorach i sprawdzająca, który sprawdza się najlepiej do danego problemu
    for classifier in klasyfikatory:
        a = globals()[classifier]()
        a.fit(X_train, y_train)
        print(f'Wynik dla {classifier} = {a.score(X_test, y_test)}')
        wyniki[classifier] = a.score(X_test, y_test)
        plot_decision_regions(np.array(X_test), np.array(y_test), clf=clf, legend=1)
        plt.scatter(np.array(X_train)[:, 0], np.array(X_train)[:, 1])
        plt.axvline(x=0)
        plt.axhline(y=0)
        plt.title(f'{classifier} Iris sepal features')
        plt.xlabel('sepal length (cm)')
        plt.ylabel('sepal width (cm)')
        plt.show()
    print(wyniki)

    with open('wyniki_lab3.json', 'w') as outfile:
        json.dump(wyniki, outfile)


def main():
    # zad1()
    # zad2()
    zad3_4()
    # zad5_6()
    # zad789()
    # zad10_11()


if __name__ == '__main__':
    main()
