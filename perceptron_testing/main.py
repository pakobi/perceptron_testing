
import os
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from datetime import datetime

now = datetime.now()
print("now =", now)

# ddmmYY_HMS - string użyty jako nazwa pliku do zapisywania wyników testu
dt_string = now.strftime("%d%m%Y_%H%M%S")

# ======================================================================================================================
# - Czesc Pierwsza - START
#       Pobieranie danych z plików: mnist_test.csv, mnist_train.csv (biblioteka MNIST),
#       Przetwarzanie danych train i test do wektora data = (train_imgs, test_imgs, train_labels, test_labels)
#       i zachowanie przetworzonych zbiorów do pliku binarnego "pickled_mnist.pkl".
#       Plik binarny "pickled_mnist.pkl" będzie w cześci drugiej programu wykorzystywany do czytania danych ponieważ
#       pobieranie danych z niego jest szybsze niż z plików csv.
#       Procedura ta wykonywana jest tylko raz ponieważ jej celem jest przygotowanie danych do dalszej pracy,
#       Jeżeli plik pickled_mnist.pkl istnieje, to procedura ładowania jest pomijana co przyśpiesza działanie programu
# ======================================================================================================================

# Sprawdzam czy plik istnieje tak to ide dalej jeśli nie to importuje dane z plików csv do pliku binarnego
isDataFileImported = os.path.isfile('./pickled_mnist.pkl')
print("MNIST data set exists: ", isDataFileImported)

if (isDataFileImported != True ):
    print("... Ladowanie danych z plików csv")

    image_size = 28  # width and length
    no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    data_path = "/"
    # Pobranie danych zbioru trenującego
    train_data = np.loadtxt("mnist_train.csv", delimiter=",")
    print("... mnist_train.csv - jest zaladowany")
    # Pobranie danych zbioru testującego
    test_data = np.loadtxt("mnist_test.csv", delimiter=",")
    print("... mnist_test.csv - jest zaladowany")
    var = test_data[:10]

    fac = 0.99 / 255
    # Wydzielenie zbiorów trenującego oraz testowego w przetworzenie na liczby zmiennoprzecinkowe z zakresu 0 do 1 i usunięcie zer
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

    # Wydzielenie wektorów etykiet dla zbiorów trenującego i testowego

    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])
    # Return evenly spaced values within a given interval
    lr = np.arange(no_of_different_labels)

    # Przetwarzanie zbioru etykiet w prezentację typu float
    train_labels_one_hot = (lr==train_labels).astype(float)
    test_labels_one_hot = (lr==test_labels).astype(float)

    # Ponieważ nie chcemy używać ani 0 ani 1
    # dla etykiet w zbiorach trenującym i testowym zamieniamy na wartości z różne od 0 i 1.
    train_labels_one_hot[train_labels_one_hot==0] = 0.01
    train_labels_one_hot[train_labels_one_hot==1] = 0.99
    test_labels_one_hot[test_labels_one_hot==0] = 0.01
    test_labels_one_hot[test_labels_one_hot==1] = 0.99

    print("... Drukujemy obrazy dla dwóch liter w celu weryfikacji danych")
    # Drukowanie 2 obrazków w celu weryfikacji danych
    for i in range(2):
        img = train_imgs[i].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        plt.show()

    # Zachowanie danych do pliku binarnego w celu szybszego odczytu danych z pliku

    with open("pickled_mnist.pkl", "bw") as fh:
        data = (train_imgs,
                test_imgs,
                train_labels,
                test_labels)
        pickle.dump(data, fh)

    print("Dane zostały zachowane do pliku: pickled_mnist.pkl")

# ========================================================================================================
# Czesc Pierwsza - koniec
# Plik binarny pickled_mnist.pkl został stworzony i dane zostały zachowane do pliku
# ========================================================================================================
# Czesc Druga - start
#       Uczenie Sieci Neuronowej oraz przeprowadzanie testów dla zadanych przypadków testowych.
#       Pomiar Dopasowania dla zbiorów trenującego i testowego i zachowanie wyników do pliku o nazwie
#       skladajacej sie z daty i czasu '01022022_102042.txt'
# ========================================================================================================

# Dane wczytywane za pomocą pickle.load, znaczne przyśpieszenie programu niż ładowanie bezposrednio z csv
# We are able now to read in the data by using pickle.load. This is a lot faster than using loadtxt on the csv files:
print("Ladowanie danych z pliku binarnego: pickled_mnist.pkl")

with open("pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

X_train_img = data[0]
X_test_img = data[1]
y_train_labels = data[2]
y_test_labels = data[3]

'''
image_size = 28  # width and length
no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
'''
print("---------- Dane Zaladowane !!! ------------------------")
y_train_labels = np.ravel(y_train_labels, order='C')
print(y_train_labels[:10])

print(X_train_img[:784])

# Wyswietlanie obrazow ze zbioru treningowego
'''
for i in range(2):
    img = X_train_img[i].reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.show()
'''


def zachowajWynikdoPliku(context):
    with open(dt_string+".txt", "a") as fd_write:
        fd_write.write(context+"\n")

# MLPClassifier
# ======================================================================================================================
# FUNKCJA > multi_perceptron_hiddenLayerSize_test(params)
# ----------------------------------------------------------------------------------------------------------------------
# Wielowarstwowy Perceptron uczenie róznymi algorytmami, testowanie wyuczonego perceptronu
# Opis: Funkcja do trenowania i testowania Klasy multilayer perceptron
#       najpierw trenuje, tworzy siec w oparciu o zadane algorytmy  parametry, następnie drukuje dopasowania
#       odpowiednio dla zbioru trenującego i testowego
# ======================================================================================================================
#   params - Parametry Wejściowe
#
#   param[in]   testNumber = "Nazwa Testu" - incremetowana w zaleznosci od kolejnych ustawien
#   param[in]   data - vector - lista danych wejsciowych zawiera kolejno wektory danych:
#                       X_train_img, X_test_img, y_train_labels, y_test_labels
#   param[in]   hiddenLayerSize - lista podawana jako krotka zawierajaca ilosc warstw ukrytych oraz ilość wag w warstwie
#                         np.  ((30,),) - 1 wartswa zawieraja 30 wag(neuronów) w wartswie
#                         np.  ((5,), (30,), (200,)) - lista zawierajaca kolejno:
#                                                                   1 wartswa 5 wag, 1 wartswa 30 wag,
#                                                                   1 wartswa 200 wag
#                         np.  ((30,), (30, 30,), (30, 30, 30,)) - lista zawierajaca kolejno:
#                                                                   1 wartswa 30 wag,
#                                                                   2 warstwy po 30 wag w każdej
#                                                                   3 warstwy po 30 wag w każdej
#   param[in]   activationFunction = ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
#   param[in]   maxNumberOfIterations - liczba iteracji (przekazana jako krotka) trenujących siec neuronowa dla pojedynczej
#                               wartswy ukrytej i jej wag
#                               np. (5,) - 5 iteracji dla trenowania jednej warstwy z wynikiem dopasowania
#                                   (5, 30, 50,) - trenowanie pojedynczej warstwy kolejno: 5, 30, 50 iteracjami z wynikiem
#                                                   dopasowania dla każdego podzbioru iteracji
#   param[in]   optimizationAlgorytm - algorytmy optymalizacji:
#                                       'sgd'   - The Stochastic Gradient Descent
#                                       'adam'  - The Adam optimization algorithm is an extension to stochastic gradient descent
#                                       'lbfgs' - the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm
#                                        np. ('adam', 'sgd', 'lbfgs') podawane jako krotka
#   param[in]   learningRate(in) - współczynnik uczenia, przekazana jako wartość
# ----------------------------------------------------------------------------------------------------------------------
#   Wynik
#   Drukuje dopasowania:
#   np.    Zbior trenujacy dopasowanie: 0.780833
#          Zbior testowy   dopasowanie: 0.784900
# ======================================================================================================================

def multi_perceptron_hiddenLayerSize_test(testNumber,
                                          data,
                                          hiddenLayerSize,
                                          activationFunction,
                                          maxNumberOfIterations,
                                          optimizationAlgorytm = 'sgd',
                                          learningRate = 0.1):

    X_train = data[0]
    X_test = data[1]
    y_train = np.ravel(data[2], order='C')
    y_test = np.ravel(data[3], order='C')

    ind = 1
    for layers_size in hiddenLayerSize:
        for solverOptimization in optimizationAlgorytm:
            for activation in activationFunction:
                for iterations in maxNumberOfIterations:

                    print(testNumber + "." + str(ind))
                    zachowajWynikdoPliku(testNumber + "." + str(ind))
                    print("Algorythm Optymalizacji: %s , Funkcja aktywacji: %s , Liczba iteracji: %s" % (
                    str(solverOptimization), str(activation), str(iterations)))
                    zachowajWynikdoPliku("Algorythm Optymalizacji: %s , Funkcja aktywacji: %s , Liczba iteracji: %s" % (
                    str(solverOptimization), str(activation), str(iterations)))
                    print("Wielkości macierzy (ilośc wartsw:", len(layers_size), ", ilość neuronów: %s )" % (str(layers_size)))
                    zachowajWynikdoPliku("Wielkości macierzy (ilośc wartsw: %s, ilość neuronów: %s )" % (str(len(layers_size)), str(layers_size)))

                    mlp = MLPClassifier(
                        hidden_layer_sizes=layers_size,
                        activation=activation,
                        max_iter=iterations,
                        alpha=1e-4,
                        batch_size='auto',
                        solver=solverOptimization,
                        verbose=10,
                        random_state=1,
                        learning_rate_init=learningRate, # szybkość uczenia
                    )

                    ind = ind + 1

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
                        mlp.fit(X_train, y_train)


                    print("Zbior trenujacy dopasowanie: %f" % mlp.score(X_train, y_train))
                    zachowajWynikdoPliku("Zbior trenujacy dopasowanie: %f" % mlp.score(X_train, y_train))
                    print("Zbior testowy   dopasowanie: %f" % mlp.score(X_test, y_test))
                    zachowajWynikdoPliku("Zbior testowy   dopasowanie: %f" % mlp.score(X_test, y_test))


# ======================================================================================================================
#   Definicje Testów
# ----------------------------------------------------------------------------------------------------------------------
# TEST 1 -  Dla ilu warstw wynik jest najbardziej optymalny
#           Sprawdzenie dopasowania dla: jednej, dwóch i trzech warstw
num_TEST = "TEST 1"
num_hiddenLayerSize = ((30,), (30, 30,), (30, 30, 30,))
f_aktywacji = ('relu',)
f_optymalizacji = ('sgd',)
num_liczbaIteracji = (4,)
num_learningRate = 0.1

multi_perceptron_hiddenLayerSize_test(num_TEST, data, num_hiddenLayerSize, f_aktywacji, num_liczbaIteracji, f_optymalizacji, num_learningRate)

# TEST 2 -  Ile neuronów(wag) powinno być w jednej warstwie
#           Porównujemy po jednej warstwie z ilością 5, 30 i 200 wag

num_TEST = "TEST 2"
num_hiddenLayerSize = ((5,), (30,), (200,))
f_aktywacji = ('relu',)
f_optymalizacji = ('sgd',)
num_liczbaIteracji = (4,)
num_learningRate = 0.1

multi_perceptron_hiddenLayerSize_test(num_TEST, data, num_hiddenLayerSize, f_aktywacji, num_liczbaIteracji, f_optymalizacji, num_learningRate)

# TEST 3 -  Jaka ilość iteracji dla 1 warstwy daje najbardziej optymalne wyniki
#           Sprawdzamy dla wartości 5, 30, 50

num_TEST = "TEST 3"
num_hiddenLayerSize = ((30,),)
f_aktywacji = ('relu',)
f_optymalizacji = ('sgd',)
num_liczbaIteracji = (5, 30, 50,)
num_learningRate = 0.1

multi_perceptron_hiddenLayerSize_test(num_TEST, data, num_hiddenLayerSize, f_aktywacji, num_liczbaIteracji, f_optymalizacji, num_learningRate)

# TEST 4 -  Przy wykorzystaniu danych z poprzednich testów, szukamy najlepszego modelu przy użyciu różnych solverów, jak
#           i funkcji aktywacji

num_TEST = "TEST 4"
num_hiddenLayerSize = ((30,),)
f_aktywacji = ('identity', 'logistic', 'tanh', 'relu')
f_optymalizacji = ('adam', 'sgd', 'lbfgs')
num_liczbaIteracji = (30,)
num_learningRate = 0.1

multi_perceptron_hiddenLayerSize_test(num_TEST, data, num_hiddenLayerSize, f_aktywacji, num_liczbaIteracji, f_optymalizacji, num_learningRate)
