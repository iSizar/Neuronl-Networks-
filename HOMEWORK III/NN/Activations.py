import numpy as np
from numpy import exp

def sigmoid(z) -> np.array:
    ret_vec = map(lambda x: 1./(1 + np.exp(-x)), z)
    return np.array(list(ret_vec))


def relu(z):
    return z if z > 0 else 0


def print_f(z):
    strg = ""
    for x in z[0]:
        strg = strg + ' ' + format(x, ".3f")
    print(strg)


# calculate the softmax of a vector
def softmax(vector):
    e = exp(vector)
    return e / e.sum()


activation_dict = {
    'sigmoid': sigmoid,
    'relu': relu,
    'softmax': softmax
}
