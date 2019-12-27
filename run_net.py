import dataset_gen as data
training, evaluation = data.generate_sets()

# Sample the data
sample = next(training())
n_pix = sample.shape[0]*sample.shape[1]
n_nodes = [n_pix, n_pix]

# Get infinite dataaaa
while 1:
    print(next(training()).ravel())
    