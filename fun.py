from typing import Callable, TypeVar
import numpy as np


class Neuron:
	def __init__(self, weights: list[float], bias:float):
		self.weights = weights
		self.bias = bias

	T = TypeVar('T')

	def classify(self, inputs: list[float], classifier: Callable[[float], T]) -> T:
		z = zip(inputs, self.weights)
		products = []
		for (x, w) in z:
			products.append(x * w)

		y = sum(products) + self.bias

		return classifier(y)


class MP(Neuron):
	pass


class Perceptron(Neuron):
	def __init__(self, n: int):
		w = np.random.rand(n)
		bias = np.random.rand(1)
		super().__init__(weights=list(w), bias=float(bias))

	def train(self, inputs: list[list[float]], target: list[float], alpha: float, precision: float, max_iter=100):
		def output(inputs: list[float]) -> float:
			z = zip(inputs, self.weights)
			products = []
			for (x, w) in z:
				products.append(x * w)
			y = sum(products) + self.bias

			return y

		actual = [output(inputs[i]) for i in range(len(inputs))]
		i = 0
		error = np.subtract(target, actual)
		while i < max_iter and np.abs(error).max() >= precision:
			for (x, expected) in zip(inputs, target):
				actual = output(x)
				delta = expected - actual
				np.add(self.weights, [alpha * delta * x[i] for i in range(len(x))])
				self.bias += alpha * delta
			actual = [output(inputs[i]) for i in range(len(inputs))]
			error = np.subtract(target, output(actual))
			i += 1


class PerceptronWithoutLearning(Neuron):
	def __init__(self, n: int, bias:float):
		w = [1.0] * n
		super().__init__(weights=w, bias=bias)
