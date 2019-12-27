from neural_network.network import Network

import dataset_gen as data
training, evaluation = data.generate_sets()

# Sample the data
sample = next(training())
n_pix = sample.shape[0]*sample.shape[1]
n_nodes = [n_pix, n_pix]

autoencoder = Network(
    model=None,
    data_range=(0,1)
    )

# Run dumb methods
autoencoder.train(training)
autoencoder.train(evaluation)

    