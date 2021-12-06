import csv
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

#ZADANIE 1 utworzyć środowisko

#ZADANIE 2 Korzystając z biblioteki scikit-learn załadować dataset digits
digits = datasets.load_digits()
# Jakie dane przechowuje ta baza? Jaki jest ich format? Jakie są klasy?
# Skorzystać z dostępnych pól: DESCR, data, target, target_names, images.
# Jaka jest różnica pomiędzy danymi w data a images?
# Wyświetlić jedno ze zdjęć jako macierz numpy korzystając z biblioteki matplotlib.
# Wyświetlić po kilka (np. 5) elementów dla każdej z dostępnych klas.
def zad2():
    # print(digits) # cała zawartość (macierze i info)
    # print(f'DESCR : {digits.DESCR}') # dane o datasecie
    # print(f'data : {digits.data}') # spłaszczone macierze o 1 wymiar
    # print(f'target : {digits.target}') # ciąg przyporządkować
    # print(f'target names : {digits.target_names}') # nazwy klas
    # print(f'images : {digits.images}') # macierze

    # print(f'Size of: data = {digits.data.shape}, images = {digits.images.shape}')

    # plt.imshow(digits.images[0], cmap="gray_r")
    # plt.imshow(digits.images[-1], cmap="gray_r") # ostatnie
    # plt.show()

    elements = 5

    fig, axs = plt.subplots(len(digits.target_names), elements)

    for nr in range(len(digits.target_names)):
        for i in range(elements):
            # maska z numpy
            axs[nr][i].imshow(digits.images[digits.target == nr][i], cmap='gray_r')
            axs[nr][i].axis('off')
    plt.show()

# ZADANIE 3 Wykorzystaj funkcję train_test_split do podziału zbioru digits 80/20, 70/30
def zad3():
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

# ZADANIE 4 Korzystając z biblioteki scikit-learn załadować dataset Olivetti faces
# Jakie dane przechowuje ta baza? Jaki jest ich format? Jakie są klasy?
# Podzielić zbiór zdjęć wraz z przypisanymi im klasami na zbiory treningowy (80% wszystkich zdjęć) oraz testowy (20% wszystkich zdjęć).
# Podzielić zbiór zdjęć na zbiory treningowy i testowy w takich samych proporcjach, tak, że zbiory te będą rozłączne pod względem znajdujących się w nich osób.
# Wyświetlić zdjęcia osób ze zbioru testowego wraz z etykietami.
def zad4():
    faces = datasets.fetch_olivetti_faces()
    X, y = faces.data, faces.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(X_test.shape)
    print(y_test.shape)
    fig = plt.figure(figsize=(8, 8))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(80):
        ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(X_test.reshape(-1, 64, 64)[i], cmap='gray_r', interpolation='nearest')
        # label the image with the target value
        ax.text(0, 7, str(y_test[i]))
    plt.show()

# ZADANIE 5 Korzystając z biblioteki scikit-learn załadować wybraną przez siebie bazę danych
# Jakie dane przechowuje ta baza? Jaki jest ich format? Jakiego typu jest to problem?
# Czy niezbędne jest dodatkowe przetwarzania danych? Wyświetlić / wypisać przykładowe dane.
# Podzielić dane na podzbiór do uczenia oraz testowania.
def zad5():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    plt.figure(2, figsize=(8, 6))
    plt.clf()
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()

    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(X_test.shape)
    print(y_test.shape)

# ZADANIE 6 Korzystając z funkcji make_classification wygenerować nowy zbiór danych.
# Jakie dane przechowuje ta baza? Jaki jest ich format? Jakiego typu jest to problem?
# Czy niezbędne jest dodatkowe przetwarzania danych? Wyświetlić / wypisać przykładowe dane.
def zad6():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
                               n_redundant=2, random_state=42)
    for i in range(100):
        print(f'X = {X[i]}, y = {y[i]}')

# ZADANIE 7 Korzystając z platformy https://www.openml.org/ oraz funkcji fetch_openml  pobierz wybrany zbiór danych
# Jakie dane przechowuje ta baza? Jaki jest ich format? Jakiego typu jest to problem? Czy niezbędne jest dodatkowe
# przetwarzania danych? Wyświetlić / wypisać przykładowe dane.
def plot_digits(X, title):
    """Small helper function to plot 100 digits."""
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((16, 16)), cmap="Greys")
        ax.axis("off")
    fig.suptitle(title, fontsize=24)
    plt.show()

def zad7():
    X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                        random_state=0, train_size=1_000, test_size=100)
    plot_digits(X_test, "Uncorrupted test images")

# ZADANIE 8 Zapoznać się z problemem: battery/problem
# Pobrać zamieszczone tam dane (plik *training_data.txt).
# Wczytać dane (plik ma format csv).
# Podzielić dane na train / test.
def zad8():
    data = []
    with open('./_data/trainingdata.txt', 'r') as csv_f:
        csv_reader = csv.reader(csv_f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_reader:
            data.append(row)
    data = np.array(data)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)

def zadHelloWorld():
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 0, 0, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict([[1, 1]]))  # Sprawdź sam(a) jakie będą wyniki dla innych danych wejściowych.

# ZADANIE 9 stwórz bramkĘ OR w postaci klasyfikatora, realizującego tę funkcję logiczną.
# wykorzystaj metodę plot_tree do zobrazowania drzewa
def zad9_10():
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict([[1, 0], [0, 1], [1, 1]]))
    plot_tree(clf)
    plt.show()

def main():
    # zad2()
    # zad3()
    # zad4()
    # zad5()
    # zad6()
    # zad7()
    # zad8()
    # zadHelloWorld()
    zad9_10()


if __name__ == '__main__':
    main()
