import numpy as np


def sigma(num):
    return 1/(1+np.exp(-num))
    
def readData():
    x_train = np.loadtxt("X_train.csv")
    y_train = np.loadtxt("Y_train.csv")
    x_test = np.loadtxt("X_test.csv")
    y_test = np.loadtxt("y_test.csv")

    return x_train, y_train, x_test, y_test

def createRandomWeights(inputSize, hiddenSize, outputSize):
    w1 = np.random.rand(hiddenSize, inputSize)
    b1 = np.random.rand(hiddenSize, 1)
    w2 = np.random.rand(outputSize, hiddenSize)
    b2 = np.random.rand(outputSize, 1)
    return w1, b1, w2, b2

def forwardPass(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = sigma(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = sigma(z2)

    return z1, z2, a1, a2

def backwardPass(x, z1, z2, a1, a2, y, w1, b1, w2, b2):
    dz2 = (a2 - y)
    dw2 = np.dot(dz2, a1.T)
    db2 = dz2
    dz1 = np.dot(w2.T, dz2) * sigma(z1) * (1-sigma(z1))
    dw1 = np.dot(dz1, x.T)
    db1 = dz1


    w1 = w1 - dw1
    w2 = w2 - dw2
    b1 = b1 - db1
    b2 = b2 - db2

    return w1, b1, w2, b2

def training(x_train, y_train, hiddenSize):
    w1, b1, w2, b2 = createRandomWeights(len(x_train[0]), hiddenSize, 1)
    for i in range(len(x_train)):
        z1, z2, a1, a2 = forwardPass(x_train[i], w1, b1, w2, b2)
        for i in range(10):
            w1, b1, w2, b2 = backwardPass(x_train[i], z1, z2, a1, a2, y_train[i], w1, b1, w2, b2)
    
    return w1, b1, w2, b2

def test(x_test, y_test, w1, b1, w2, b2, hiddenSize):
    sum = 0

    for i in range(len(y_test)):
        z1, z2, a1, a2 = forwardPass(x_test[i], w1, b1, w2, b2)
        print("A", a2, "Y", y_test[i])


def main():
    x_train, y_train, x_test, y_test = readData()

    hiddenSize = 2

    w1, b1, w2, b2 = training(x_train, y_train, hiddenSize)

    test(x_test, y_test, w1, b1, w2, b2, hiddenSize)

main()