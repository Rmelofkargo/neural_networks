inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

#dot product
import numpy as np
output = np.dot(inputs, weights[0]) + biases[0]
print('dot product neuron: ', output)

layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

"""
para la capa con más de una neurona, se necesita poner en el producto punto
primero el array de arrays, por eso dará un array de productos punto.
Tambien evita el error de shape en numpy
"""
output = np.dot(weights, inputs) + biases
print('dot product layer: ', output)