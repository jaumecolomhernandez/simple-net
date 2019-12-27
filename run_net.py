from neural_network.network import Network

import dataset_gen as data
training, evaluation = data.generate_sets()

# Sample the data
sample = next(training())
n_pix = sample.shape[0]*sample.shape[1]
n_nodes = [n_pix, n_pix]

autoencoder = Network(model=None)
autoencoder.train(training)
autoencoder.train(evaluation)

# Get infinite dataaaa
while 1:
    print(next(training()).ravel())
    