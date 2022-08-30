import cv2
import numpy as np

p = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

def distance(a, b):
    x1 = (a[0] - b[0])**2
    x2 = (a[1] - b[1])**2
    x3 = (a[2] - b[2])**2

    out = (x1 + x2 + x3)**.5
    return out

def minDistance(pixel):
    p = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    dist = []

    for i in range(3):
        dist.append(distance(pixel, p[i]))

    min = 10000
    minNum = -1
    for i in range(len(dist)):
        if dist[i] < min:
            min = dist[i]
            minNum = i
        
    return minNum

img = cv2.imread('input.png')
output1 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
output2 = img

minDistance(img[50, 25])

for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        num = minDistance(img[i, j])
        output1[i, j] = p[num]

cv2.imwrite("./output1.png", output1)

print(output2.shape[0])
center0 = output2.shape[0] / 2
center1 = output2.shape[1] / 2
for i in range(0, output2.shape[0]):
    if i >= center0 - 25 and i <= center0 + 25:
        for j in range(0, output2.shape[1]):
            if j >= center1 - 25 and j <= center1 + 25:
                output2[i, j] = [0, 0, 0]


cv2.imwrite("./output2.png", output2)


