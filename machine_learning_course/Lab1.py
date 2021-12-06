import csv

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


def zad_1():
    digits = datasets.load_digits()
    # print(digits)
    # print(digits.target) # [0]
    # print(digits.target.shape)
    # print(digits.data.shape)
    # print(f'images:{digits.images.shape}')

    # print(f'DESCR:{digits.DESCR}')
    # print(f'target_names:{digits.target_names}')
    # print(f'target:{digits.target}')
    # print(f'images:{digits.images}')
    # print(f'data:{digits.data}')

    # plt.imshow(digits.images[0], cmap="gray_r")
    # plt.imshow(digits.images[-1], cmap="gray_r")
    # plt.show()

    elements = 5

    fig, axs = plt.subplots(len(digits.target_names), elements)

    for nr in range(len(digits.target_names)):
        for i in range(elements):
                                         # maska z numpy
            axs[nr][i].imshow(digits.images[digits.target==nr][i], cmap='gray_r')
            axs[nr][i].axis('off')
    # plt.show()

    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(X_train.shape, X_test.shape)
    # print(y_train.shape, y_test.shape)

    clf = SVC()
    # clf.fit([X_train[0]], [y_train[0]])
    print(y_train[0:10])
    #clf.fit(X_train, y_train)
    clf.fit([X_train[0], X_train[2]], [y_train[0], y_train[2]])
    src = clf.score([X_train[0], X_train[2]], [y_train[0], y_train[2]])
    pred = clf.predict([X_train[0], X_train[2]])
    print(f'prediction: {pred}')
    print([y_train[0], y_train[2]])
    print(f'score: {src}')


def zad_2():
    faces = datasets.fetch_olivetti_faces()

    X, y = faces.data, faces.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) #, shuffle=False)

    # print(X_train.shape, X_test.shape)
    # print(y_train.shape, y_test.shape)

    # elements = 5

    # fig, axs = plt.subplots(4, elements)

    # for nr in range(len(faces.target_names)):
    #     for i in range(elements):
    #         axs[nr][i].imshow(faces.images[faces.target == nr][i], cmap='gray_r')
    #         axs[nr][i].axis('off')
    # plt.show()

    svc = LinearSVC() # SVC() #(verbose=True)
    svc.fit(X_train, y_train)
    src = svc.score(X_test, y_test)
    print(f'score : {src}')


def zad_3(): # AND
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 0, 0, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict([[1, 0], [0, 1]]))  # Sprawdź sam(a) jakie będą wyniki dla innych danych wejściowych.

def zad_4(): # OR
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict([[1, 0], [0, 1], [ 1, 1]]))
    plot_tree(clf)
    plt.show()

def zad_5(): # todolab2-3
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
    # print(X)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # color = ['r', 'g']

    # for row, label in zip(X_train, y_train):
    #     ax.scatter(row[0], row[1], row[2], marker='o', c=color[label])
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  # , shuffle=False)

    # print(X_train.shape, X_test.shape)
    # print(y_train.shape, y_test.shape)

    svc = LinearSVC()  # SVC() #(verbose=True)
    svc.fit(X_train, y_train)
    src = svc.score(X_test, y_test)
    print(f'score : {src}')

def baterie():
    data = []
    with open('Laboratorium/_data/trainingdata.txt', 'r') as csv_f:
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

def main():
    # zad_1()
    # zad_2()
    # zad_3()
    # zad_5()
    # zad_4()
    baterie()


if __name__ == '__main__':
    main()


