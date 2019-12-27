from neural_network.network import Network
from neural_network.layer import Dense
import neural_network.activation as activations

import dataset_gen as data

training, evaluation = data.generate_sets()

# Sample the data
sample = next(training())
n_pix = sample.shape[0] * sample.shape[1]

# Number of neurons per layer
hidden = [6,5,5]
n_nodes = [n_pix] + hidden + [n_pix]

# Create model structure
layers = []
for i in range(len(n_nodes)-1):
    layers.append(Dense(n_nodes[i], n_nodes[i+1], activations.tanh))

autoencoder = Network(model=layers, data_range=(0, 1))

# Run dumb methods
autoencoder.train(training)
autoencoder.train(evaluation)

