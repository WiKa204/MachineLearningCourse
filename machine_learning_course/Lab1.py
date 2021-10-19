import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
    plt.show()

    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

def zad_2():
    faces = datasets.fetch_olivetti_faces()

    X, y = faces.data, faces.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    # elements = 5

    # fig, axs = plt.subplots(4, elements)

    # for nr in range(len(faces.target_names)):
    #     for i in range(elements):
    #         axs[nr][i].imshow(faces.images[faces.target == nr][i], cmap='gray_r')
    #         axs[nr][i].axis('off')
    # plt.show()

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

def main():
    # zad_1()
    # zad_2()
    # zad_3()
    zad_4()

if __name__ == '__main__':
    main()


