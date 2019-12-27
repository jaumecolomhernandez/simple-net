import numpy as np

class tanh:
    @staticmethod
    def calc(v):
        return np.tanh(v)

    @staticmethod
    def calc_d(v):
        return 1 - np.tanh(v)**2

class logistic:
    @staticmethod
    def calc(v):
        return 1/(1+np.exp(-v))
    
    @staticmethod
    def calc_d(v): 
        return calc(v) * (1 - calc(v))

class relu:
    @staticmethod
    def calc(v):
        return np.maximum(0, v)
    
    @staticmethod
    def calc_d(v):
        if v > 0:
            return 1
        else:
            return 0