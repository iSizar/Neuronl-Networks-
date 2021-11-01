import pickle
import gzip
from config import *
from NN.Model import *
from NN.Layer import *
from NN.Activations import *

if __name__ == "__main__":
    with gzip.open("../HOMEWORK II/dataset/mnist.pkl.gz", 'rb') as fd:
        train_set, valid_set, test_set = pickle.load(fd, encoding='latin')
        model = SGD()
        model.add(Input(784))
        model.add(Dense(100))
        model.add(Dense(10, activation='softmax'))
        print(softmax(np.array([[1, 1, 3, 4]])))
        print(np.array([[1, 1, 3, 4]]) + 1)

        '''
        model.w.append(np.zeros((1, 2), dtype=np.float32))
        model.w.append(np.zeros((2, 2), dtype=np.float32))
        model.w.append(np.zeros((2, 1), dtype=np.float32))

        model.b.append(np.zeros((1, 2), dtype=np.float32))
        model.b.append(np.zeros((1, 2), dtype=np.float32))
        model.b.append(np.zeros((1, 1), dtype=np.float32))

        model.activ.append(sigmoid)
        model.activ.append(sigmoid)
        model.activ.append(sigmoid)


        model.w[1] = np.array([[-3., 6.], [1., -2.]])
        model.w[2] = np.array([[8.], [4.]])
        '''
        model.fit(train_set[0], train_set[1], max_epochs=max_epoch, eps=eps,
                  valid_x_data=valid_set[0], valid_y_data=valid_set[1])

        # x = np.array([2, 3, 4],  dtype=np.float32).reshape(1, 3)
        # w = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.float32)
        # print(x.dot(w))
        # d = np.array([[1, 2, 3, 4]], dtype=np.float32)
        # print(d.dot(w.T))
        '''
        model.fit(np.array([[2, 6]]), np.array([[0]]), max_epochs=3, eps=0.5,
                  valid_x_data=np.array([[2, 6]]), valid_y_data=np.array([[0]]))
        
        print(model.w)
        for wt in model.w:
            print(wt)
        '''
        model.test(test_set[0], test_set[1])


