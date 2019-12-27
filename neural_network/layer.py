import numpy as np

class Dense:
    def __init__(
        self,
        n_inputs,
        n_outputs,
        activation
    ):  
        # Declaration parameters
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.weights = np.random.sample(size=(self.n_inputs+1, self.n_outputs))-.5
        self.activation = activation
        
        self.learning_rate = 0.001

        # Placeholders for input and output (later used)
        self.x = np.zeros((1, self.n_inputs + 1))
        self.y = np.zeros((1, self.n_outputs))
    def forward_prop(self, input_data):
        " "
        self.x = np.concatenate((input_data, np.ones((1,1))),axis=1)
        v = self.x @ self.weights
        self.y = self.activation.calc(v)
        return self.y
    
    def back_prop(self, de_dy):
        " "
        dy_dv = self.activation.calc_d(self.y)
        dy_dw = self.x.transpose() @ dy_dv
        de_dw = de_dy * dy_dw
        self.weights -= de_dw * self.learning_rate
        de_dx = (de_dy * dy_dv) @ self.weights.transpose()
        return de_dx[:,:-1]   # The last value corresponds to the bias (constant 1)
