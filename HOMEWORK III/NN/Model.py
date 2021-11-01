import numpy as np
from NN.Layer import *
from NN.Activations import *
from NN.Utils import *


class SGD:
    def __init__(self):
        self.w = []
        self.b = []
        self.activ = []

    def add(self, layer: Layer):
        shape = None
        activation = None
        use_bias = False
        if len(self.w) == 0:
            if type(layer) is not Input:
                raise Exception(f" Wrong Input Layer type received {type(layer)}")
            shape = (1, layer.size)

        else:
            shape = (self.w[-1].shape[1], layer.size)
            activation = activation_dict[layer.activation]
            use_bias = layer.use_bias

        weights = np.random.randn(*shape) / np.sqrt(1 / shape[0])  # np.zeros(shape)
        biases = None
        if use_bias:
            biases = (np.random.randn(layer.size)).reshape(1, layer.size) / np.sqrt(1 / shape[0])
        else:
            biases = np.zeros((1, layer.size))

        self.w.append(weights)
        self.b.append(biases)
        self.activ.append(activation)

    def feed_forward(self, x):
        zs = [x.reshape((1, len(x)))]
        ys = [x.reshape((1, len(x)))]
        for (w_is, b_is), f in zip(zip(self.w[1:], self.b[1:]), self.activ[1:]):
            zs.append(ys[-1].dot(w_is) + b_is)
            ys.append(f(zs[-1]))
        return zs, ys

    def fit(self, x_data, y_data, max_epochs: int, eps: float,
            rlambda: float = 5.0, gamma_mom: float = 0.4, gamma_rmsprop: float = 0.4,
            valid_x_data=None, valid_y_data=None):
        print(f"######## fit start ########")

        train_data = np.array(list(zip(x_data, y_data)))
        start_eps = eps
        for epoch in range(max_epochs):
            eps = max(start_eps - (epoch+1) * 0.1, start_eps / (epoch+1))
            print(f"######## Current iteration {epoch + 1}, eps = {eps}, momentum = {gamma_mom}, rmsprop = {gamma_rmsprop} ########")

            np.random.shuffle(train_data)
            batches = crate_batches(10, train_data)
            nr_wrong_classified = 0
            for batch in batches:
                delta_w = [np.zeros(w.shape, dtype=np.float32) for w in self.w]
                vt      = [np.zeros(w.shape, dtype=np.float32) for w in self.w]
                st_w    = [np.zeros(w.shape, dtype=np.float32) for w in self.w]
                delta_b = [np.zeros(b.shape, dtype=np.float32) for b in self.b]
                st_b    = [np.zeros(b.shape, dtype=np.float32) for b in self.b]

                for x, target in batch:
                    t = np.zeros(self.w[-1].shape[1])
                    t[target] = 1
                    _, ys = self.feed_forward(x)
                    # print('old', ys[-1])
                    y = ys[-1][0]
                    predict = np.argmax(ys[-1])
                    if predict != target:
                        nr_wrong_classified += 1

                    # y[predict] = 1
                    prev_delta = y - t  # y * (np.ones(t.shape) - y) * (t - t)

                    idx = len(ys) - 1
                    while idx > 0:
                        delta_l = prev_delta

                        prev_delta = ys[idx-1] * \
                                     (np.ones(len(ys[idx-1])) - ys[idx-1]) * \
                                     delta_l.dot(self.w[idx].T)

                        # gradients
                        dC_dw = delta_l * ys[idx-1].T
                        dC_db = delta_l

                        # RMSProp
                        # st_w[idx] = gamma_rmsprop * st_w[idx] + (1 - gamma_rmsprop) * dC_dw * dC_dw
                        # st_b[idx] = gamma_rmsprop * st_b[idx] + (1 - gamma_rmsprop) * dC_db * dC_db

                        # new leaning rates:
                        # eps_w = np.full(delta_w[idx].shape, eps) / np.sqrt(st_w[idx] + 0.0000001)
                        # eps_b = np.full(delta_b[idx].shape, eps) / np.sqrt(st_b[idx] + 0.0000001)

                        # momentum
                        vt[idx] = gamma_mom * vt[idx] + eps / len(batch) * dC_dw

                        # without rmsprop
                        delta_w[idx] -= (vt[idx] + eps * rlambda / len(x_data) * self.w[idx])
                        delta_b[idx] -= (eps/len(x_data) * dC_db)

                        # with rmsprop
                        # delta_w[idx] -= (vt[idx] + eps_w * rlambda / len(x_data) * self.w[idx])
                        # delta_b[idx] -= (eps_b / len(x_data) * dC_db)

                        # ONLINE trainning
                        # self.w[idx] = (1 - (eps * rlambda) / len(x_data)) * self.w[idx] - eps/len(batch) * dC_dw
                        # self.b[idx] = self.b[idx] - eps/len(x_data) * dC_db

                        idx -= 1

                for idx in range(1, len(self.w)):
                    self.w[idx] += (delta_w[idx])
                    self.b[idx] += (delta_b[idx])

                del delta_w[:]
                del delta_w
                del delta_b[:]
                del delta_b
                del st_b[:]
                del st_b
                del st_w[:]
                del st_w
                del vt[:]
                del vt

            training_acc = 1 - nr_wrong_classified / len(x_data)
            valid_acc = self.acc(valid_x_data, valid_y_data)
            print(f"######## Training acc: {training_acc} ------ Validation acc: {valid_acc}  ########")

        print(f"######## fit end  ########")

    def test(self, test_x_data: np.array, test_y_data: np.array):
        print(f"####### Test {self.acc(test_x_data, test_y_data)}")

    def acc(self, test_x_data: np.array, test_y_data: np.array):
        nr_wrong = 0
        for x, t in zip(test_x_data, test_y_data):
            _, ys = self.feed_forward(x)
            # print("ACC VAL:", ys[-1])
            if np.argmax(ys[-1]) != t:
                nr_wrong += 1
        return 1 - nr_wrong / (len(test_x_data))
