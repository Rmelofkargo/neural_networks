import math

layer_outputs = [4.8, 1.21, 2.398]

E = math.e

exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)


print(exp_values)