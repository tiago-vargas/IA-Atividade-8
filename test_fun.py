from fun import *


def test_mp_neuron_positive():
	w = [0.02, -0.2, 0.03, -0.09]
	neuron = Neuron(weights=w, bias=-1)

	x: list[float] = [10, -20, -8, 2]
	g = lambda y: 'A' if y > 0 else 'B'
	class_ = neuron.classify(inputs=x, classifier=g)

	# (10 * 0.02) + (-20 * -0.2) + (-8 * 0.03) + (2 * -0.09) - 1 == 2.78 > 0
	assert class_ == 'A'


def test_mp_neuron_negative():
	w = [-0.2, -0.2, 0.3, -0.09]
	neuron = Neuron(weights=w, bias=-1)

	x: list[float] = [100, -20, -8, 2]
	g = lambda y: 'A' if y > 0 else 'B'
	class_ = neuron.classify(inputs=x, classifier=g)

	# (100 * -0.2) + (-20 * -0.2) + (-8 * 0.3) + (2 * -0.09) - 1 == -19.58 <= 0
	assert class_ == 'B'
