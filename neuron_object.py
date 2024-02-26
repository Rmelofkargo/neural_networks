import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

#inputs deben normalizarse entre -1 y 1
#weights debe normalizarse entre -0.1 y 0.1
#biasese deben normalizarse  a 0, pero se puede propagar a todas las neuranos (dead network)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) #se evita la trasposición
        self.biases = 0.1 * np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU():
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2, 5)
#layer2 = Layer_Dense(5, 2)# tiene que ser del shape de la capa anterior
activation1 = Activation_ReLU()
layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
#layer2.forward(layer1.output)
#print(layer2.output)