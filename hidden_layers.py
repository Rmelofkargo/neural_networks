import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]

inputs = [0,2,-1,3.3,-2.7,1.1,20,.2,-100]

output = []

for i in inputs:
    output.append(max(0,i))
    """if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
    """

print(output)