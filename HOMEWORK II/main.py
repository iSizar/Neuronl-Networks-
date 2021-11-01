import pickle
import gzip
import numpy as np
from copy import deepcopy
from Perceptron import perceptron, add_bias_padding
from config import *


def activation(z: int) -> int:
    ret_val = z
    if not ADALINE_USED:
        ret_val = 1 if z > 0 else 0
    return ret_val


def validate(mlp: np.array, validation_set: np.array) -> float:
    nr_incorrect_classified = 0
    for x, t in validation_set:
        z = int(np.argmax([activation(mlp[i].dot(x)) for i in range(len(mlp))]))
        if t != z:
            nr_incorrect_classified += 1

    return nr_incorrect_classified/len(validation_set)


def online_training(mlp: np.array, training_set: np.array, validation_set: np.array):
    nr_iteration = 1
    eps = l_rate
    while nr_iteration <= MAX_ITERATIONS:
        train_error = 0
        np.random.shuffle(training_set)
        for x, t in training_set:
            z = int(np.argmax([activation(mlp[i].dot(x)) for i in range(len(mlp))]))
            if t != z:
                mlp[z] -= x * eps
                mlp[t] += x * eps
                train_error += 1
        train_error /= len(training_set)
        validation_error = validate(mlp, validation_set)
        print(f"{nr_iteration}: Training acc = {1-train_error}, validation acc = {validation_error}; eps = {eps}")
        eps = max(0.2/nr_iteration, l_rate-0.03*nr_iteration)
        nr_iteration += 1


def crate_batches(batch_count: int, all_data: np.array) -> list:
    ret_list = []
    batch_size = len(all_data) // batch_count
    idx = 1
    while (idx * batch_size) < len(all_data):
        ret_list.append(all_data[(idx-1)*batch_size: min(idx*batch_size, len(all_data))])
        idx += 1
    return ret_list


def batch_training(mlp: np.array, training_set: np.array, validation_set: np.array):
    nr_iteration = 1
    l_rate = 0.3
    MAX_ITERATIONS = 200
    eps = l_rate
    while nr_iteration <= MAX_ITERATIONS:
        train_error = 0
        np.random.shuffle(training_set)
        train_batches = crate_batches(100, training_set)
        deltas = np.zeros((len(train_batches), 10, 28*28+1))
        for i, batch in enumerate(train_batches):
            for x, t in batch:
                z = int(np.argmax([activation(mlp[i].dot(x)) for i in range(len(mlp))]))
                if t != z:
                    deltas[i][z] -= x * eps
                    deltas[i][t] += x * eps
                    train_error += 1

        for d in deltas:
            for j in range(10):
                mlp[j] += d[j] * eps

        train_error /= len(training_set)
        validation_error = validate(mlp, validation_set)
        print(f"{nr_iteration}: Training acc = {1-train_error}, validation acc = {1-validation_error}; eps = {eps}")
        eps = max(0.3/nr_iteration, l_rate-0.002*nr_iteration)
        nr_iteration += 1


def mini_batch_training(mlp: np.array, training_set: np.array, validation_set: np.array):
    nr_iteration = 1
    eps = l_rate
    last_acc = 0.0
    MAX_ITERATIONS = 50
    while nr_iteration <= MAX_ITERATIONS:
        train_error = 0
        np.random.shuffle(training_set)
        train_batches = crate_batches(100, training_set)
        for i, batch in enumerate(train_batches):
            delta = np.zeros((10, 28*28+1))
            for x, t in batch:
                z = int(np.argmax([activation(mlp[i].dot(x)) for i in range(len(mlp))]))
                if t != z:
                    delta[z] -= x * eps
                    delta[t] += x * eps
                    train_error += 1

            for j in range(10):
                mlp[j] += delta[j] * eps

        train_error /= len(training_set)
        validation_error = validate(mlp, validation_set)
        print(f"{nr_iteration}: Training acc = {1-train_error}, validation acc = {1-validation_error}; eps = {eps}")
        if last_acc > 1-train_error:
            eps = max(0.1/nr_iteration, eps-0.03)
        last_acc = 1-train_error
        nr_iteration += 1


def evaluate(mlp: np.array, testing_set: np.array):
    nr_incorrect_classified = 0
    for x, t in testing_set:
        z = int(np.argmax([activation(mlp[i].dot(x)) for i in range(len(mlp))]))
        if t != z:
            nr_incorrect_classified += 1

    print(f"### Test acc = {1-nr_incorrect_classified/len(testing_set)} ###")


with gzip.open("dataset/mnist.pkl.gz", 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

    train_set, valid_set, test_set = add_bias_padding(train_set), add_bias_padding(valid_set), add_bias_padding(test_set)

    MLP = np.array([perceptron(28*28 + 1) for _ in range(10)])
    # print(MLP.shape)
    # online_training(MLP, train_set, valid_set)
    # evaluate(MLP, test_set)

    #mini_batch_training(MLP, train_set, valid_set)
    #evaluate(MLP, test_set)

    batch_training(MLP, train_set, valid_set)
    evaluate(MLP, test_set)
