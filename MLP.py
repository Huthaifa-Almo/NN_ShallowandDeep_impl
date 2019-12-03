import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class MLP:
    # initialize the MLP parameters
    def __init__(self, layers_dim, lambd, learning_rate=0.01, num_iterations=3000, print_cost=False):
        self.layers_dim = layers_dim   # number of layers and number of neurons on each
        self.lambd = lambd  # regularization parameter
        self.learning_rate = learning_rate  # the hyperparameter that used on updating the gradients
        self.num_iterations = num_iterations  # the number of epochs that will be used
        self.print_cost = print_cost  # optional printing cost after specific number of iteration

    # initialize the weights and biases
    def initialize(self, layer_dims):
        parameters = {}
        L = len(layer_dims)
        for l in range(1, L):
            # create each layers weights with random numbers and biases with zeros
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(1/(layer_dims[l - 1]))
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        return parameters

    # sigmoid activation function
    def sigmoid(self, z):
        A = 1 / (1 + np.exp(-z))
        activation_cache = A.copy()
        return A, activation_cache

    # relu activation function
    def relu(self, z):
        A = z * (z > 0)
        activation_cache = z
        return A, activation_cache

    # forward activation
    def activation_forward(self, A_prev, W, b, activation):
        if activation == "sigmoid":
            linear_cache = (A_prev, W, b)
            A, activation_cache = self.sigmoid(np.dot(W, A_prev) + b)
        elif activation == "relu":
            linear_cache = (A_prev, W, b)
            A, activation_cache = self.relu(np.dot(W, A_prev) + b)
        cache = (linear_cache, activation_cache)
        return A, cache

    # model forward
    def model_forward(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2
        for l in range(1, L):
            A_prev = A
            A, cache = self.activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
            caches.append(cache)
        AL, cache = self.activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
        caches.append(cache)
        return AL, caches

    # computing cost function
    def compute_cost(self, AL, Y, parameters, lambd):
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))
        cost = np.squeeze(cost)

        L = len(parameters) // 2
        regularization = 0

        for l in range(L):
            regularization += np.sum(np.square(parameters["W" + str(l + 1)]))

        L2_regularization_cost = (lambd / (2 * m)) * regularization
        cost = cost + L2_regularization_cost
        return cost

    # sigmoid derivative
    def sigmoid_backward(self, dA, activation_cache):
        return dA * (activation_cache * (1 - activation_cache))

    # relu derivative
    def relu_backward(self, dA, activation_cache):
        return dA * (activation_cache > 0)

    # backward activation
    def activation_backward(self, dA, cache, activation, lambd):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            A_prev, W, b = linear_cache
            m = A_prev.shape[1]
            dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambd/m) * W
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            A_prev, W, b = linear_cache
            m = A_prev.shape[1]
            dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambd/m) * W
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    # backward model
    def model_backward(self, AL, Y, caches, lambd):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.activation_backward(dAL,
                                                                                                        current_cache,
                                                                                                        "sigmoid",
                                                                                                        lambd)
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.activation_backward(grads["dA" + str(l + 1)],
                                                                      current_cache,
                                                                      "relu",
                                                                      lambd)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        return grads

    # updating weights and biases
    def update_parameters(self, parameters, grads, learning_rate):
        L = len(parameters) // 2
        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        return parameters

    # fitting the model
    def fit(self, X, Y):
        costs = []
        parameters = self.initialize(layers_dim)
        for i in range(0, self.num_iterations):
            AL, caches = self.model_forward(X, parameters)
            cost = self.compute_cost(AL, Y, parameters, self.lambd)
            grads = self.model_backward(AL, Y, caches, self.lambd)
            parameters = self.update_parameters(parameters, grads, self.learning_rate)
            if self.print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if self.print_cost and i % 100 == 0:
                costs.append(cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()

        return parameters

    # predicting function
    def predict(self, X, y, parameters):
        m = X.shape[1]
        p = np.zeros((1, m), dtype=np.int)
        a3, caches = self.model_forward(X, parameters)
        for i in range(0, a3.shape[1]):
            if a3[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        print(str(np.mean((p[0, :] == y[0, :]))))
        return p


#######################################################################################################################
#################################### END OF THE MODEL / START THE EXAMPLE #############################################
#######################################################################################################################

# create a sample dataset
def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    return X, Y


# load the sample dataset
X, Y = load_planar_dataset()
# split dataset into test and train
x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.4,
    random_state=0)
# define number of layers and the number of neuron on each layers
layers_dim = np.array([2, 256, 1])
# create an instance from the model
# try different hyberparameters and monitor the results
model = MLP(layers_dim, lambd=0.7, learning_rate=0.03, num_iterations=10000, print_cost=True)
# fitting the model with the training dataset
parameters = model.fit(x_train.T, y_train.T)
# predict the result using train and test dataset
pred_train = model.predict(x_train.T, y_train.T, parameters)
pred_test = model.predict(x_test.T, y_test.T, parameters)
