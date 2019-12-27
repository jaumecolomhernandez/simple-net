import numpy as np

class Dense:
    def __init__(
        self,
        n_inputs,
        n_outputs,
        activation
    ):
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.weights = np.random.sample(size=(self.n_inputs+1, self.n_outputs))-.5
        self.activation = activation

    def forward_prop(self, input_data):
        x = np.concatenate((input_data, np.ones((1,1))),axis=1)
        return self.activation.calc(x @ self.weights)
