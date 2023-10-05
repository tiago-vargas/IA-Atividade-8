from typing import Callable, TypeVar


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
	def __init__(self, n: int, bias:float):
		w = [1.0] * n
		super().__init__(weights=w, bias=bias)
