import numpy as np
from copy import deepcopy


def perceptron(input_shape: int) -> np.array:
    w = np.random.rand(input_shape)
    return np.zeros(input_shape)  # np.random.rand(input_shape) w /len(w)  #


def add_bias_padding(data_set: np.array) -> np.array:
    t_data = deepcopy(data_set)
    new_train_features = [np.concatenate((x, np.array([1])), axis=0) for x in t_data[0]]
    new_train_features = np.array(new_train_features)
    return np.array(list(zip(new_train_features, t_data[1])))
