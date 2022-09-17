import numpy as np
from matplotlib import pyplot as plt
import math

def equation(w, x):
    x_Out = np.zeros(len(x))
    for i in range(len(x)):
        temp = .5*x[i]
        x_Out[i] = -w * (temp**3 + math.sin(5*temp))+5.5
    return x_Out

def lossFunction(w, x, y):
    return np.sum((equation(w, x) - y)**2)

def gradientFunction(w, x, y):
    gradient = np.sum(2*equation(1, x) * (equation(w, x) - y))
    return gradient

def gradientDescent(wGuess, alpha, iterations, x, y):
    wValues = [wGuess]

    for i in range(iterations):
        gradient = gradientFunction(wValues[-1], x, y)
        wValues.append(wValues[-1] - alpha * gradient)
    
    return wValues


def run():
    x = np.load('x_train.npy')
    y = np.load('y_train.npy')

    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')

    newWValues = gradientDescent(-1, .0001, 50, x, y)
    
    xRange = np.arange(x_test.min(), x_test.max(), .01)
    yOutput = equation(newWValues[-1], xRange)
    # print(newWValues[-1])
    
    fig, axs = plt.subplots(1)

    axs.plot(xRange, yOutput)
    axs.scatter(x_test, y_test, color="orange")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(["Model", "Test Data"])
    plt.title("Question 3")
    # axs.scatter(x, y, color="green")
    plt.show()

run()
