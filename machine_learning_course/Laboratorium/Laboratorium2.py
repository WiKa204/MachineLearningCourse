import csv
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report

# ZADANIE 1 TODO2 z Lab01
# Stwórz klasyfikator SVC i wytrenuj (metoda fit) go na pięciu zdjęciu z bazy danych.
# Przetestuj klasyfikator na tym samym zdjęciu (metoda predict), wyświetl wynik.
def zad1():
    digits = datasets.load_digits()

    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(X_train.shape, X_test.shape)
    # print(y_train.shape, y_test.shape)

    clf = SVC()
    clf.fit([X_train[0], X_train[1], X_train[2], X_train[3], X_train[4]],
            [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]])
    print(y_train[0:10])
    pred = clf.predict([X_train[3], X_train[4]])
    print(f'prediction: {pred}')

# ZADANIE 2 TODO5 z Lab01 i spróbuj wytrenować model. To samo zrób dla
# TODO6 i TODO7. Spróbuj różnych modeli.
def zad2():
    # TODO 5
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(X_test.shape)
    print(y_test.shape)
    clf = SVC()
    clf.fit(X_train, y_train)
    print(y_train[0:10])
    pred = clf.predict([X_train[3], X_train[4]])
    print(f'prediction: {pred}')
    # TODO 6
    X_mc, y_mc = make_classification(n_samples=1000, n_features=20, n_informative=2,
                               n_redundant=2, random_state=42)
    X_mctrain, X_mctest, y_mctrain, y_mctest = train_test_split(X_mc, y_mc, test_size=0.2, random_state=42, shuffle=False)
    print(X_mctest.shape)
    print(y_mctest.shape)
    clf = LinearSVC()
    clf.fit(X_mctrain, y_mctrain)
    print(y_mctrain[0:10])
    pred_mc = clf.predict([X_mctrain[3], X_mctrain[4]])
    print(f'prediction (make_classification): {pred_mc}')
    # TODO 7
    X_fetch, y_fetch = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
    print(X_fetch.shape)
    print(y_fetch.shape)
    X_fetch = MinMaxScaler().fit_transform(X_fetch)
    X_ftrain, X_ftest, y_ftrain, y_ftest = train_test_split(X_fetch, y_fetch, random_state=42, test_size=0.2)
    print(X_ftest.shape)
    print(y_ftest.shape)
    clf = LinearRegression()
    clf.fit(X_ftrain, y_ftrain)
    print(y_ftrain[0:10])
    pred_fetch = clf.predict([X_ftrain[3], X_ftrain[4]])
    print(f'prediction (fetch): {pred_fetch}')

# ZADANIE 3 Utworzyć własny zbiór uczacy, nauczyć klasyfikator
# min. 3 cechy, rozłożenie cech w 3D
def zad3():
    X = [[-30, 23, "duże"],
               [20, 15, "brak"],
               [10, 3, "małe"],
               [15, 8, "brak"],
               [1, 9, "średnie"],
               [23, 3, "brak"],
               [18, 12, "duże"],
               [17, 11, "małe"],
               [19, 19, "małe"],
               [25, 10, "średnie"],
               [-20, 0, "średnie"],
               [19, 14, "małe"],
               [12, 5, "małe"],
               [-26, 22, "średnie"],
                [19, 13, "średnie"]]
    y = [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0]
    ocena = {"brak": 0, "małe": 1, "średnie": 2, "duże": 3}
    X = [X[i][:2]+[ocena[X[i][2]]] for i in range(len(X))]
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  # , shuffle=False)
    svc = LinearSVC()  # SVC() #(verbose=True)
    svc.fit(X_train, y_train)
    src = svc.score(X_test, y_test)
    print(f'score : {src}')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    color = ['r', 'g']
    for row, label in zip(X_train, y_train):
        ax.scatter(row[0], row[1], row[2], marker='o', c=color[label])
    plt.show()

# ZADANIE 4 Wyświetl i zwizualizuj macierz pomyłek dla bazy danych digits (dla klasyfikatora DecisionTreeClassifier).
# Skorzystaj z funkcji plot_confusion_matrix aby wygenerować graficzną reprezentację macierzy pomyłek.
# Dodatkowo wyświetl kilkanaście przykładowych błędnych predykcji dla wartości 3 oraz 8.
def zad4():
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    # print(f'confusion_matrix: \n{confusion_matrix(y_test, y_pred)}')

    cmdtree = confusion_matrix(y_test, y_pred)
    cmdtree, dtree.score(X_test, y_test)
    print(cmdtree)
    print(dtree.score(X_test, y_test))
    for input, prediction, label in zip(X_test, y_pred, y_test):
        if label == 3 or label == 8:
            if prediction != label:
                print(input, ' has been classified as ', prediction, ' and should be ', label)
    plot_confusion_matrix(dtree, X_test, y_test)
    plt.show()

    print("Classification report for classifier %s:\n%s\n"
          % (dtree, classification_report(y_test, y_pred)))

# ZADANIE 5 Otwórz swój kod z TODO8 z poprzednich zajęć i:
# Wytrenuj model umożliwiający predykcję czasu pracy na baterii.
# Tym razem będzie to zadanie regresji. Na wykresie zaprezentuj wynik
# - rzeczywisty czas pracy dla danych testowych vs wynik predykcji.
# wyznacz metryki np. mean absolute error, mean squared error,
# r2 dla LinearRegression oraz DecisionTreeRegressor
def zad5_6():
    data = []
    with open('./_data/trainingdata.txt', 'r') as csv_f:
        csv_reader = csv.reader(csv_f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_reader:
            data.append(row)
    data = np.array(data)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)
    plt.scatter(X_train, y_train)
    plt.show()
    clf = Pipeline([
        ('poly', PolynomialFeatures(degree=8)),
        ('line', LinearRegression())
    ])
    clf = RandomForestRegressor()
    clf.fit(X_train, y_train)
    plt.scatter(X_train, clf.predict(X_train))
    plt.show()
    y_pred = clf.predict(X_test)
    print(mean_absolute_error(y_test, y_pred))
    print(mean_absolute_error(y_test, y_pred, multioutput='raw_values'))
    print(mean_squared_error(y_test, y_pred))
    print(r2_score(y_test, y_pred))
    print(r2_score(y_test, y_pred, multioutput='variance_weighted'))
    print(r2_score(y_test, y_pred, multioutput='uniform_average'))
    print(r2_score(y_test, y_pred, multioutput='raw_values'))


def main():
    # zad1()
    # zad2()
    # zad3()
    # zad4()
    zad5_6()


if __name__ == '__main__':
    main()