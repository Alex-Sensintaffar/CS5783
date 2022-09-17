import numpy as np
from matplotlib import pyplot as plt
import math

def GraphOfInputData():
    xTest = np.load('x_test.npy')
    yTest = np.load('y_test.npy')
    xTrain = np.load('x_train.npy')
    yTrain = np.load('y_train.npy')

    fig, axs = plt.subplots(2)
    axs[0].scatter(xTest, yTest)
    axs[0].set_title("Test Data")
    axs[1].scatter(xTrain, yTrain)
    axs[1].title.set_text("Training Data")

    plt.show()


def equation(x):
    x_Out = np.zeros(len(x))
    for i in range(len(x)):
        temp = .5*x[i]
        x_Out[i] = (temp**3 + 5*math.sin(5*temp))+5.5
    return x_Out

def main():
    x = np.load('x_train.npy')
    y = np.load('y_train.npy')

    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')


    y_calc = equation(x_test)

    plt.scatter(x_test, y_test)
    plt.scatter(x_test, y_calc)
    plt.title("Question 1")
    plt.legend(["Test Data", "Calculated from Equation"])
    plt.show()

main()