import numpy as np

# 3 Tacos, 3 Burritos, $11.25
# 4 Tacos, 2 Burritos, $10.00

a = np.array([[3, 3], [4, 2]])
b = np.array([11.25, 10])

print(np.linalg.solve(a,b))