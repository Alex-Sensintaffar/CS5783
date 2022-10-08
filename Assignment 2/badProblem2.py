from cmath import isclose
import numpy as np
from numpy import array
from matplotlib import pyplot as pt
import math
import pandas as pd

class Neuron:
    def __init__(self, value):
        self.value = value

    def setValue(self, value):
        self.value = value

    def getValue(self):
        return self.value

    
    
class Layer:
    def __init__(self, size=None, name=None):
        self.size = size
        self.name = name #input, hiden, output
        self.neuronArray = []
    
    def addToNeuronArray(self, newNeuronValue):
        self.neuronArray.append(newNeuronValue)

    def createInputLayer(self, data):
        self.size = len(data)
        for i in range(len(data)):
            self.addToNeuronArray(Neuron(data[i]))

    def createHiddenLayer(self, pastLayer, size, weight2DArray, biasArray):
        # i: cycles through each neuron in current layer
        # j: cycles through each neuron in past layer
        self.size = size
        for i in range(size):
            sum = 0
            for j in range(pastLayer.getSize()):
                sum += pastLayer.getNeuronValue(j) * weight2DArray[i][j]
            value = activationFunction(sum)
            self.addToNeuronArray(Neuron(value))

    def createOutputLayer(self, pastLayer, size, weight2DArray, biasArray):
        self.size = size
        for i in range(size):
            sum = 0
            for j in range(pastLayer.getSize()):
                sum += pastLayer.getNeuronValue(j) * weight2DArray[i][j]
            # value = activationFunction(sum)
            self.addToNeuronArray(Neuron(sum))

    def getSize(self):
        return self.size

    def getNeuronValue(self, neuronIndex):
        return self.neuronArray[neuronIndex].getValue()

    def getAllNeuronValues(self):
        tempArray = []
        for i in range(len(self.neuronArray)):
            tempArray.append(self.neuronArray[i].getValue())
        return np.array(tempArray)

def readData():
    x_train = np.loadtxt("X_train.csv")
    y_train = np.loadtxt("Y_train.csv")
    x_test = np.loadtxt("X_test.csv")
    y_test = np.loadtxt("y_test.csv")

    return x_train, y_train, x_test, y_test

# TODO update create layers to handel np.arrays()
# TODO create the training algorithem and implement it

# def sigma(num):
#     return 1/(1+math.exp(num * -1))

# def activationFunction(input):
#     sigmaFunction = np.vectorize(sigma)
#     return sigmaFunction(input)

def activationFunction(input):
    return 1/(1+np.exp(-input))

def training(
    x_train, y_train, hiddenLayerSize):
    alpha = 1
    
    inputLayerSize = len(x_train[0])
    # hiddenLayerSize = 5
    outputLayerSize = 1

    # W1
    hiddenWeight2DArray = np.random.randn(hiddenLayerSize, inputLayerSize)
    # B1
    hiddenBiasArray = np.random.randn(hiddenLayerSize, 1)
    # W2
    outputWeight2DArray = np.random.randn(outputLayerSize, hiddenLayerSize)
    # B2
    outputBiasArray = np.random.randn(outputLayerSize, 1)

    lossArray = []
    # for i in range(1):
    for i in range(len(x_train)):
        inputLayer = Layer()
        inputLayer.createInputLayer(x_train[i])

        hiddenLayer = Layer()
        hiddenLayer.createHiddenLayer(inputLayer, hiddenLayerSize, hiddenWeight2DArray, hiddenBiasArray)

        outputLayer = Layer()
        outputLayer.createOutputLayer(hiddenLayer, outputLayerSize, outputWeight2DArray, outputBiasArray)

        output = outputLayer.getAllNeuronValues()
        # print(output)

        # W2
        W2pt1 = (output - y_train[i]) * hiddenLayer.getAllNeuronValues()
        W2pt1.shape = (len(hiddenLayer.getAllNeuronValues()), 1)
        W2 = np.array(outputWeight2DArray) - alpha * W2pt1.T
        # print(W2)

        # B2
        # print(np.array(hiddenBiasArray))
        B2 = np.array(hiddenBiasArray) - alpha * (output - y_train[i])
        # print(B2)

        # W1
        # print(np.array(hiddenWeight2DArray))
        Z1 = np.array(hiddenWeight2DArray) * x_train[i] + hiddenBiasArray
        W1 = np.array(hiddenWeight2DArray) - alpha * (np.array(outputWeight2DArray).T * activationFunction(Z1) * (1-activationFunction(Z1)) * (output - y_train[i]) * np.array(x_train[i]).T)
        # print(W1)

        # B1
        # print(np.array(hiddenBiasArray))
        B1 = np.array(hiddenBiasArray) - alpha * (np.array(outputWeight2DArray).T * activationFunction(Z1) * (1-activationFunction(Z1)) * (output - y_train[i]))
        # print(B1[0])

        # lossArray.append(-((1-y_train[i])*math.log(1-output[0])+y_train[i]*math.log(output[0])))
        lossArray.append(((output[0]-y_train[i])**2))
        outputWeight2DArray = W2
        outputBiasArray = B2[0]

        hiddenWeight2DArray = W1
        hiddenBiasArray = B1[0]

    print(lossArray)
    return hiddenWeight2DArray, hiddenBiasArray, outputWeight2DArray, outputBiasArray

def test(x_test, y_test, w1, b1, w2, b2, hiddenLayerSize):

    sum = 0
    for i in range(len(x_test)):
        inputLayer = Layer()
        inputLayer.createInputLayer(x_test[i])

        hiddenLayer = Layer()
        hiddenLayer.createHiddenLayer(inputLayer, hiddenLayerSize, w1, b1)

        outputLayer = Layer()
        outputLayer.createOutputLayer(hiddenLayer, 1, w2, b2)

        output = outputLayer.getAllNeuronValues()

        print("output:", output[0], "\ny_test:", y_test[i])

        if math.isclose(output[0], y_test[i]):
            sum += 1
    
    accuracy = sum / len(x_test)

    print(accuracy)


def main():

    x_train, y_train, x_test, y_test = readData()

    hiddenLayerSize = 3

    w1, b1, w2, b2 = training(x_train, y_train, hiddenLayerSize)

    test(x_test, y_test, w1, b1, w2, b2, hiddenLayerSize)




main()

