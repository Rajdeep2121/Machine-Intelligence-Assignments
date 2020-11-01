import numpy as np 
import math

'''To set randomness value for checking'''
# np.random.seed(0)

class NN:
	def __init__(self, layers=[2, 2, 1]):	
		self.layers = layers
		# self.activation = 'sigmoid'
		self.weights = []
		self.biases = []

		for i in range(len(self.layers) -1):
			self.weights.append(np.random.randn(self.layers[i+1], self.layers[i]))	# dim(weights) = (next layer x curr layer)
			self.biases.append(np.zeros([self.layers[i+1], 1]))	# dim(biases) = (next layer x 1)
			
	def sigmoid(self, X):
		return 1/(1+ np.exp(-X))

	def forward(self, X):
		layerInput = []
		for i in X:
			layerInput.append(i)
		layerOutput = []
		layerImInputs = [layerInput]	

		for i in range(len(self.weights)):
			layerOutput.append(self.weights[i].dot(layerInput) + self.biases[i])	# (weights dot product with layer input) + biases
			layerInput = self.sigmoid(layerOutput[-1])		# applying activation function to layer outputs
			layerImInputs.append(layerInput)
		return layerOutput, layerImInputs
	
	def train(self, X, y):
		layerOutput, layerImInputs = self.forward(X)
		return layerOutput[-1][0][0]

	def errorOutput(self, y_hat, y):
		return 0.5 * (y_hat - y)**2

inputs = [3, 5]
output = 0.6
check = NN()
y_hat = float(check.train(inputs, output))
error = check.errorOutput(y_hat, output)
print("y_hat: ", y_hat)
print("Error: ", error)



