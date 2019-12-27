import numpy as np

class Network:
    def __init__(
        self,
        model=None,
        data_range=(0,1)
    ):  
        # Base params
        self.layers = model
        self.iter_train = int(1e8)
        self.iter_eval = int(1e6)

        # Efficiency for data normalization!
        self.data_range = data_range
        self.mean = np.mean(self.data_range)
        self.rang = self.data_range[1]-self.data_range[0]

    def forward_prop(self, input_data):
        """ Does forward propagation 
            On an interesting side, creating a 2D (1xN) array
            makes the results cleaner.         
        """
        input_data = input_data[np.newaxis,:]
        output = self.layers[0].forward_prop(input_data)
        return output.ravel() # Heavy use of ravel (be careful)

    # Dumb methods to train and eval the network
    def train(self, training_set):
        for i in range(self.iter_train):
            sample = self.normalize(next(training_set()).ravel())
            output = self.forward_prop(sample)
            print(output)
        return

    def evaluate(self, evaluation_set):
        for i in range(self.iter_eval):
            sample = self.normalize(next(evaluation_set()).ravel())
            print(sample)
        return
    
    # Inputs to the net should be between [-0.5, 0.5] cause gradient descent
    def normalize(self, input_data):
        return (input_data-self.mean)/self.rang

    def denormalize(self, input_data):
        return input_data*self.rang + self.mean