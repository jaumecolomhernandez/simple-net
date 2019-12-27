import numpy as np
import matplotlib.pyplot as plt
import os


class Network:
    """ Dead simple implementation of a neural network 
        TBC
    """
    def __init__(
        self,
        model=None,
        error_class=None,
        data_range=(0,1)
    ):  
        # Base params
        self.layers = model
        self.error = error_class
        self.iter_train = int(1e8)
        self.iter_eval = int(1e6)

        # Efficiency for data normalization!
        self.data_range = data_range
        self.mean = np.mean(self.data_range)
        self.rang = self.data_range[1]-self.data_range[0]

        # Parameters for error accounting and plot generation
        self.error_history = [] # TODO: Convert to numpy array
        self.error_bins = 1000
        self.report_max = 0     # As we will be using log error max is zero
        self.report_min = -3    
        self.report_folder = "outputs"
        self.report_name = "history.png"
        self.report_interval = int(1e5)

        try: os.mkdir(self.report_folder)   # Create the folder in case it does not exist
        except: pass

    def forward_prop(self, input_data):
        """ Does forward propagation 
            On an interesting side, creating a 2D (1xN) array
            makes the results cleaner.         
        """
        y = input_data[np.newaxis,:]
        for layer in self.layers:
            y = layer.forward_prop(y)
        return y.ravel() # Heavy use of ravel (be careful)

    # Dumb methods to train and eval the network
    def train(self, training_set):
        " Trains the network given a dataset "
        for i in range(self.iter_train):
            # Take sample from dataser, flatten it and normalize
            sample = self.normalize(next(training_set()).ravel())

            # Forward propagate
            output = self.forward_prop(sample)

            # Compute errors
            error_array = self.error.calc(sample, output) # 2X2 size
            error_d_array = self.error.calc_d(sample, output)
            error = np.mean(error_array**2)**0.5

            # Error visualization
            self.error_history.append(error)
            if (i+1)%self.report_interval == 0:
                current_error = self.report()
                print(f"Report|Iteration:{i+1}|Error:{current_error}")
        return

    def evaluate(self, evaluation_set):
        " Evaluates the network given a dataset (of unseen data) "
        for i in range(self.iter_eval):
            # Take sample from dataser, flatten it and normalize
            sample = self.normalize(next(training_set()).ravel())

            # Forward propagate
            output = self.forward_prop(sample)

            # Compute errors
            error_array = self.error.calc(sample, output) # 2X2 size
            error = np.mean(error_array**2)**0.5

            # Error visualization
            self.error_history.append(error)
            if (i+1)%self.report_interval == 0:
                current_error = self.report()
        return
    
    # Inputs to the net should be between [-0.5, 0.5] cause gradient descent
    def normalize(self, input_data):
        " Normalizes the data to [-0.5,0.5] "
        return (input_data-self.mean)/self.rang

    def denormalize(self, input_data):
        " Denormalizes the data back to the original form "
        return input_data*self.rang + self.mean

    def report(self):
        " Generates and export the error "
        n_bins = int(len(self.error_history) // self.error_bins)

        # Create history with binned errors
        # TODO: We could store this so it doesn't need to be recalculated
        smoothed_history = []   
        for i_bin in range(n_bins):
            smoothed_history.append(np.mean(self.error_history[
                i_bin * self.error_bins:
                (i_bin + 1) * self.error_bins
            ]))

        # Epsilon added just in case error 0 
        error_history = np.log10(np.array(smoothed_history) + 1e-10)

        # Set bounds for the plot
        ymin = np.minimum(self.report_min, np.min(error_history))
        ymax = np.maximum(self.report_max, np.max(error_history))

        # Plot plumbing
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(error_history)
        ax.set_xlabel(f"x{self.error_bins} iterations")
        ax.set_ylabel("log error")
        ax.set_ylim(ymin, ymax)
        ax.grid()
        fig.savefig(os.path.join(self.report_folder, self.report_name))
        plt.close()

        # Returns the last log10 calculated error for the CLI logging
        return error_history[-1]