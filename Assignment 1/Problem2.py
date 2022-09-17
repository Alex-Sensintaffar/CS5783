import numpy as np
import pandas as pd


def leastSquare(actual, forcast):
    sum = 0
    for i in range(len(actual)):
        sum += (actual[i] - forcast[i])**2

    return sum/len(actual)

def regressionLine(x, y):
    #output = b0 + b1*x
    # b1 = r sy/sx
    # b0 = avrage(y) - b1*avrage(x)
    # Sx = sqrt(sum(xi-avrage(x))**2)/(len(x)-1)
    # r = (1/n-1) * sum((xi-avrage(x))/sx)(yi-avrage(y))/sy)
    for i in range(len(x)):
        xAvg = np.sum(x)/len(x)
        yAvg = np.sum(y)/len(y)

    xsum = 0
    ysum = 0

    for i in range(len(x)):
        xsum += (x[i] - xAvg)**2
        ysum += (y[i] - yAvg)**2

    sx = ((xsum)/(len(x)))**.5
    sy = ((ysum)/(len(y)))**.5

    sum = 0

    for i in range(len(x)):
        sum += (x[i] - xAvg)/sx * (y[i] - yAvg)/sy

    r = (1/(len(x))) * sum

    b1 = r * sy/sx
    b0 = yAvg - b1*xAvg

    yCalc = np.zeros(len(y))
    for i in range(len(x)):
        yCalc[i] = b0 + b1 * x[i]

    # print("xAvg", xAvg)
    # print("yAvg", yAvg)
    # print("sx", sx)
    # print("sy", sy)
    # print("r", r)
    # print("b1", b1)
    # print("b0", b0)

    return yCalc

def meanSquaredError(x, y):
    # try:
    yForcast = regressionLine(x, y)
    MSE = leastSquare(y, yForcast)
    # except:
        # MSE = "nan"
    # print(MSE)
    return(MSE)


def part1():
    file = pd.read_csv (r'Assignment1_Q2_Data.csv')
    y = pd.DataFrame(file, columns= ["Price (1000$)"]).values

    # print(y.values)

    columnsNameListing = []
    columnsDataListing = []

    for (columnName, columnData) in file.iteritems():
        columnsNameListing.append(columnName)
        columnsDataListing.append(columnData.values)

    # print(columnsNameListing)
    # print(columnsDataListing)

    leastSquareResults = []

    for i in range(1, len(columnsNameListing)-1):
        x = columnsDataListing[i]
        # print(x)
        leastSquareResults.append(meanSquaredError(x, y))

    # print(leastSquareResults)

    for i in range(1, len(columnsNameListing)-1):
        print(columnsNameListing[i], leastSquareResults[i-1])

part1()