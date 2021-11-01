import numpy as np
import time
import matplotlib.pyplot as plt
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show
from copy import deepcopy


class Neuron:
    def __init__(self, train_data, label_data):
        a = np.random.randint(0, 101)
        b = np.random.randint(0, 101)
        c = np.random.randint(0, 101)
        self.weights = np.array([a, b, c])
        self.train_data = train_data
        self.label_data = label_data
        self.neighbors = None

    def msr(self, w=None) -> float:
        if w is None:
            w = self.weights
        error = 0.0
        for idx in range(len(self.train_data)):
            y_predict = self.train_data[idx].dot(w)
            y = self.label_data[idx]
            if y * y_predict < 0:
                error = error + (y-y_predict)**2
        error /= len(train_data)
        return error

    def total_failed(self, w=None) -> float:
        if w is None:
            w = self.weights
        error = 0.0
        for idx in range(len(self.train_data)):
            y_predict = self.train_data[idx].dot(w)
            y = self.label_data[idx]
            if y * y_predict < 0:
                error += 1
        return error

    def update_neighbors(self, step_size: float, eps: int = 1) -> []:
        """
        This function will adjust the weights in respect to step_size and learning rate eps
        :param step_size: The amount by witch the weights+bias will be adjusted
        :param eps: This parameter will decrease over time
        :return: a list of possible new values for weights and bias
        """
        ret_list = []
        possible_vals_per_coordinate = [[] for _ in range(len(self.weights))]
        for i in range(len(self.weights)):
            possible_vals_per_coordinate[i].append(self.weights[i] - step_size*eps)
            possible_vals_per_coordinate[i].append(self.weights[i])
            possible_vals_per_coordinate[i].append(self.weights[i] + step_size*eps)

        current_lvl = 0
        idx = 0
        current_list = []
        stack = []
        while len(ret_list) != len(possible_vals_per_coordinate[0]) ** len(possible_vals_per_coordinate):
            if current_lvl < len(possible_vals_per_coordinate) - 1:
                current_list.append(possible_vals_per_coordinate[current_lvl][idx])
                stack.append(idx)
                current_lvl += 1
                idx = 0
            else:
                for i in range(len(possible_vals_per_coordinate[current_lvl])):
                    l_to_ad = deepcopy(current_list)
                    l_to_ad.append(possible_vals_per_coordinate[current_lvl][i])
                    ret_list.append(np.array(l_to_ad))
                    # print(l_to_ad)
                current_list.pop()
                idx = stack.pop() + 1
                current_lvl -= 1
                while idx == len(possible_vals_per_coordinate[current_lvl]) and current_lvl > 0:
                    current_lvl -= 1
                    current_list.pop()
                    idx = stack.pop() + 1
        self.neighbors = np.array(ret_list)

    def get_stochastic_update(self):
        """
        This method will randomly choose a new weight form the neighbors_list
        :return:
        """
        random_idx = np.random.randint(0, len(self.neighbors))
        return self.neighbors[random_idx]

    def get_steepest_ascent_update(self):
        """

        :return:
        """
        t = time.time()
        aux_array = self.neighbors
        idx = np.argmin(np.array(list(map(self.get_error, aux_array)), dtype=np.float))
        ret_l = self.neighbors[idx]
        # print(type(ret_l))
        # print(f"steepest_ascent_update took {time.time() - t} sec.")
        return ret_l

    def predict(self, x_y):
        ret_array = np.zeros(len(x_y))
        w = self.weights[:2]
        b = self.weights[2]
        ret_array = (x_y.dot(w) + b)>0
        return ret_array

    def get_error(self, w=None):
        return self.msr(w)


def hill_climbing(x_data, y_data, t_max, eps, step):
    neuron = Neuron(x_data, y_data)
    T_MAX = t_max
    TARGET = 0.005
    found_sol = False
    t = 0

    neuron.update_neighbors(step, eps)
    best_weights = None
    best_score = None
    current_error = None
    e = eps
    while t < T_MAX:
        current_error = neuron.get_error()
        if current_error < TARGET:
            print(f'Found solution in {t} iterations. : a,b,c ='
                  f'{neuron.weights[0], neuron.weights[1],neuron.weights[2]}'
                  f'\n error = {neuron.get_error()}')
            found_sol = True
            break
        else:
            new_w = neuron.get_steepest_ascent_update()
            new_err = neuron.get_error(w=new_w)
            if new_err < current_error:
                print(f"Ascending from {current_error} ==> {new_err}")
                neuron.weights = new_w
                neuron.update_neighbors(step, e)

            elif new_err > current_error:
                print("how???")
                raise Exception("no way....")
            else:
                if not best_score or best_score > current_error:
                    ''' Reinitialized the search alg'''
                    best_score = current_error
                    best_weights = neuron.weights
                    neuron = Neuron(x_data, y_data)
                    current_error = neuron.get_error()
                    neuron.update_neighbors(step, e)
        t += 1
        e = eps/t
    if best_score is not None and current_error > best_score:
        neuron.weights = best_weights

    if not found_sol:
            print(f'Did not found solution in {t} iterations. : a,b,c ='
                  f'{neuron.weights[0], neuron.weights[1],neuron.weights[2]}'
                  f'\n get_error = {neuron.get_error()}')

    return neuron


if __name__ == "__main__":
    train_data = np.zeros((100, 3))
    label_data = np.zeros((100, 1))

    for idx in range(50):
        train_data[idx] = np.array([np.random.randint(0, 46), np.random.randint(0, 101), 1])
        label_data[idx] = -1

    for idx in range(50, 100):
        train_data[idx] = np.array([np.random.randint(55, 101), np.random.randint(0, 101), 1])
        label_data[idx] = 1

    model = hill_climbing(t_max=200, x_data=train_data, y_data=label_data, eps=7, step=10)

    failed_to_classified = model.total_failed()

    x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
    y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
    h = .06 # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h),)
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    ax.axis('off')

    # Plot also the training points
    ax.scatter(train_data[:50, 0], train_data[:50, 1], c=label_data[:50], cmap="Blues")
    ax.scatter(train_data[50:, 0], train_data[50:, 1], c=label_data[50:], cmap="plasma")
    ax.set_title(f"Total miss classified = {failed_to_classified}")
    separator_fct = lambda x: (-model.weights[0]*x - model.weights[1]) / model.weights[2]

    plt.show()
