import numpy as np

def generate_sets():
    " Returns two generators that return the same (although not called the same) "
    examples = [
        np.array([
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ]),
        np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ]),
        np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]),
        np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]),
        np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]),
        np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]),
        np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]),
        np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]),
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]),
        np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]),
        np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]),
        np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ]),
        np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]),
        np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]),
        np.array([
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ]),
        np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]),
        np.array([
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]),
        np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ]),
    ]

    def training_set():
        while True:
            yield examples[np.random.choice(len(examples))]
            

    def evaluation_set():
        while True:
            yield examples[np.random.choice(len(examples))]

    return training_set, evaluation_set