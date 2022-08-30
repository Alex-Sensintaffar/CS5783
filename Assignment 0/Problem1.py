from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np


file = open("test.txt", 'r')

data = {}
for line in file:
    # print(line)
    for i in line:
        if i.lower() in data:
            data[i.lower()] = data[i.lower()] + 1
        else:
            data[i.lower()] = 1
# print(data)

dataLetter = []
dataFreq = []
dataNorm = []
dataHistNorm = {}

sum = 0
for i in data:
    dataLetter.append(i)
    dataFreq.append(data[i])
    sum += data[i]

for i in dataFreq:
    dataNorm.append(i / sum)

# print(dataLetter)
# print(dataFreq)
for i in range(len(dataNorm)):
    dataHistNorm[dataLetter[i]] = dataNorm[i]

print("Normalized:", dataHistNorm)
print("Unnormalized:", data)

fig, ax = plt.subplots(1, 2)
fig.suptitle("Assignment 0, Problem 1")
fig.set_size_inches(10, 7)

ax[0].bar(dataLetter, dataNorm)
ax[0].title.set_text("Histogram with Normalized Data")
ax[0].set_ylabel("Noralized Frequency")
ax[0].set_xlabel("Letter")

ax[1].bar(dataLetter, dataFreq)
ax[1].title.set_text("Histogram with Unnormalized Data")
ax[1].set_ylabel("Frequency")
ax[1].set_xlabel("Letter")


plt.show()