import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import impute, svm, model_selection
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression

diabetes = pd.read_csv("./_data/dataset_37_diabetes.csv")
diabetes['class'].loc[diabetes['class'] == 'tested_positive'] = 1
diabetes['class'].loc[diabetes['class'] == 'tested_negative'] = 0
diabetes['class'] = diabetes['class'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(diabetes.drop(['class'], axis=1), diabetes['class'],
                                                    random_state=42, stratify=diabetes['class'], test_size=0.25)


# ZADANIE 1 załaduj bazę danych Pima Indians Diabetes Database
# wyświetl statystki danych przykładowe dane. Czy w zbiorze danych brakuje jakiś wartości?
# wyznacz wynik działania 3 wybranych metod klasyfikacjina surowych danych.
# Wykorzystaj podział na zbiór treningowy i testowy lub n-krotną walidację krzyżową.
# przedstaw graficznie wyniki uzyskane przez poszczególne klasyfikatory dla surowych danych
# i po odrzuceniu / uzupełniu brakujących rekordów różnymi metodami.
# porówna z wynikami Supervised Classification on diabetes na openml.
def zad1():
    results = dict()
    imputer_clf = [
        ('DTC', Pipeline([
            ('imputer', impute.KNNImputer()),
            ('skaler', MinMaxScaler()),
            ('DTC', DecisionTreeClassifier())
        ]))
    ]
    for name, clf in imputer_clf:
        clf.fit(X_train.dropna(), y_train[X_train.notna().all(axis=1)])
        results['imputer' + name] = clf.score(X_test.dropna(), y_test[X_test.notna().all(axis=1)])

    print(results)
    plt.bar(results.keys(), results.values())
    plt.show()


# ZADANIE 2 wyświetl histogram wartości dla cechy mass
# wyświetl wartości jako boxplot dla cechy mass
def zad2():
    global diabetes
    klasyfikatory = ["LogisticRegression", "SVC", "DecisionTreeClassifier", "RandomForestClassifier"]

    wyniki = dict()
    for classifier in klasyfikatory:
        clf = Pipeline([
            ('skaler', MinMaxScaler()),
            (classifier, globals()[classifier]())])
        clf.fit(X_train.dropna(), y_train[X_train.notna().all(axis=1)])
        print(f'Wynik dla {classifier} = {clf.score(X_test, y_test)}')

    X_train.drop(['insu'], axis=1, inplace=True)
    X_test.drop(['insu'], axis=1, inplace=True)

    for a in ["preg", "plas", "pres", "skin", "mass"]:  # ,"insu" - usunięcie niepotrzebnej wartości
        X_train[a].loc[X_train[a] == 0] = np.NaN
        print(a + ": ", X_train[a].isna().sum() / len(X_train) * 100)

    zscore = abs((diabetes - diabetes.mean()) / diabetes.std())
    print(zscore.head())
    exit()
    diabetes = diabetes.loc[~(zscore >= 3).any(axis=1)]


# ZADANIE 3 zwizualizuj rozkład cechy mass od plas
def zad345():
    global diabetes, X_train, X_test
    X_train.drop(['insu'], axis=1, inplace=True)
    X_test.drop(['insu'], axis=1, inplace=True)
    plt.scatter(diabetes['mass'], diabetes['plas'])
    plt.title('mass/plas')
    plt.xlabel('mass')
    plt.ylabel('plas')
    plt.show()

    # ZADANIE 5  wytrenuj IsolationFrest
    clf = IsolationForest()
    clf.fit(diabetes[['mass', 'plas']])
    print(clf.predict(diabetes[['mass', 'plas']].head()))
    plot_decision_regions(np.array(diabetes[['mass', 'plas']]), np.array(clf.predict(diabetes[['mass', 'plas']])),
                          clf=clf)

    for a in ["preg", "plas", "pres", "skin", "mass"]:  # ,"insu" - usunięcie niepotrzebnej wartości
        X_train[a].loc[X_train[a] == 0] = np.NaN
        print(a + ": ", X_train[a].isna().sum() / len(X_train) * 100)

    # ZADANIE 4 wyznacz wartość z-score, np. za pomocą scipy.stats.zscore
    # Usuń elementy, których z-score wynosi np. powyżej 3.
    # ponownie zwizualizuj dane.
    zscore = abs((diabetes - diabetes.mean()) / diabetes.std())
    diabetes = diabetes.loc[~(zscore >= 3).any(axis=1)]
    plt.scatter(diabetes['mass'], diabetes['plas'])
    plt.title('mass/plas')
    plt.xlabel('mass')
    plt.ylabel('plas')
    plt.show()


# ZADANIE 7 Przetestuj działanie GridSearchCV oraz RandomizedSearchCV
def zad78():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=42,
                                                        stratify=iris['target'], test_size=0.25)

    parameters = {
        'kernel': ('linear', 'rbf', 'sigmoid'),
        'C': [1, 10, 30]
    }
    clf = GridSearchCV(svm.SVC(), parameters, cv=10)
    clf.fit(X_train, y_train)

    pvt = pd.pivot_table(
        pd.DataFrame(clf.cv_results_),
        values='mean_test_score',
        index='param_kernel',
        columns='param_C'
    )
    wynik = clf.best_estimator_.score(X_test, y_test)

    ax = sns.heatmap(pvt)
    plt.show()

    # ZADANIE 8 Dodaj opcję zapisywania najlepszego modelu
    # dla  zadania  związanego z grid  search.
    with open("model.wzum_jest_superanckie", 'wb') as file:
        pickle.dump(clf.best_estimator_, file)

    print(clf.best_estimator_.score(X_test, y_test))

    # with open("model.wzum_jest_superanckie", 'rb') as file:
    #    clf = pickle.load(file)
    print(clf.score(X_test, y_test))

    with open('wyniki_lab4.json', 'w') as outfile:
        json.dump(wynik, outfile)


def zad():
    with open("model.wzum_jest_superanckie", 'rb') as file:
        clf = pickle.load(file)
    print(clf.score(X_test, y_test))


def main():
    # zad1()
    # zad2()
    # zad345()
    # zad78()
    zad()


if __name__ == '__main__':
    main()
