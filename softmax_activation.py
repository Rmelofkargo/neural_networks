import math
import numpy as np

#Exponential
layer_outputs = [4.8, 1.21, 2.398]

E = math.e

exp_values = np.exp(layer_outputs)

print(exp_values)

#Normalization

norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values))

