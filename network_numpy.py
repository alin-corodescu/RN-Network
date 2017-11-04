import numpy as np


def split_into_batches(train_set):
    pass


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Network:
    weights = [np.array([])
        , np.array([[np.random.normal() for i in range(784)] for j in range(100)])
        , np.array([[np.random.normal() for i in range(100)]] for j in range(10))]

    biases = [
        np.array([]),
        np.array([np.random.normal() for i in range(100)])
        , np.array([np.random.normal() for i in range(10)])]

    y = [
        np.array([0 for i in range(784)]),
        np.array([0 for i in range(100)]),
        np.array([0 for i in range(10)])
    ]

    z = [
        np.array([0 for i in range(784)]),
        np.array([0 for i in range(100)]),
        np.array([0 for i in range(10)])
    ]

    errors = [
        np.array([0 for i in range(784)]),
        np.array([0 for i in range(100)]),
        np.array([0 for i in range(10)])
    ]

    d_weights = [np.array([])
        , np.array([[0 for i in range(784)] for j in range(100)])
        , np.array([[0 for i in range(100)]] for j in range(10))]

    d_biases = [
        np.array([]),
        np.array([0 for i in range(100)])
        , np.array([0 for i in range(10)])]

    batch_size = 1

    def train(self, train_set, iterations, learn_rate, batch_size):
        self.batch_size = batch_size
        for iteration in range(iterations):
            #   Split into mini batches
            batches = split_into_batches(train_set)
            for batch in batches:
                # TODO : reset every neuron's weight and bias delta accumulated during the last run
                for entry in batch:
                    # Grab the inputs and feed them through the network
                    inputs = np.array([entry[0]])
                    target = entry[1]
                    self.y[0] = inputs
                    # Feed forward the entry
                    self.classifyInstance()

                    # Compute the error for the last layer
                    self.computeErrorLastLayer(target)

                    # Backpropagate the error through the network
                    # Store the temporary results to d_weighs and d_biases
                    self.backpropagate()

                    #         The current batch has finished
                self.commitChanges()

    def classifyInstance(self):
        for i in range(1,3):
            for j in range(len(self.z[i])):
                self.z[i][j] = np.dot(self.weights[i][j], self.y[i-1]) + self.biases[i][j]
                self.y[i][j] = sigmoid(self.z[i][j])
        # todo make the last layer use softmax function

    def computeErrorLastLayer(self, target):
        for i in range(10):
            t = int(target == i)
            self.errors[-1][i] = self.y[-1][i] - t

    def commitChanges(self):
        self.d_weights = np.multiply(self.d_weights, 1.0/self.batch_size)
        self.d_biases = np.multiply(self.d_biases, 1.0/self.batch_size)
        self.weights = np.add(self.weights, self.d_weights)
        self.biases = np.add(self.biases, self.d_biases)
        self.d_weights = [np.array([])
            , np.array([[0 for i in range(784)] for j in range(100)])
            , np.array([[0 for i in range(100)]] for j in range(10))]

        self.d_biases = [
            np.array([]),
            np.array([0 for i in range(100)])
            , np.array([0 for i in range(10)])]


if __name__ == '__main__':
    # Create an empty network with 784 inputs, 100 neurons in the hidden layer and 10 in the output layer
    network = Network(784, 100, 10)
    network.train([], 30, 3)
