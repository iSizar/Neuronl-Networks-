import numpy as np
from abc import ABC


class Layer(ABC):
    def __int__(self):
        self._size = None
        self._use_bias = False
        self._activation = None

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, val: int):
        self.size = val

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, val: int):
        self.activation = val

    @property
    def use_bias(self):
        return self._use_bias

    @use_bias.setter
    def use_bias(self, val: bool):
        self.use_bias = val


class Input(Layer):
    def __init__(self, input_size: int):
        self._size = input_size


class Dense(Layer):
    def __init__(self, size: int, activation: str = 'sigmoid', use_bias: bool = True):
        self._size = size
        self._activation = activation
        self._use_bias = use_bias


class CONV:
    def __init__(self, channel_size: int, shape_kernel: tuple, activation='sigmoid'):
        self.shape = shape_kernel
        self.channel_size = channel_size
        self.activation = activation
