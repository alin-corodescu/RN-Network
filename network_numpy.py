import gzip
import random
import pickle
import numpy as np
import math

def split_into_batches(train_set, batch_size):
    random.shuffle(train_set)
    batches = [train_set[k: k + batch_size] for k in range(0,len(train_set),batch_size)]
    return batches


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derived(z):
    return sigmoid(z)*(1-sigmoid(z))


def softmax(zs):
    sum = np.sum(np.exp(zs))
    return np.exp(zs)/sum


class Network:
    weights = [np.array([])
        , np.array([np.random.normal(0, 1.0/math.sqrt(784),size=784)  for j in range(100)])
        , np.array([np.random.normal(0, 1.0/math.sqrt(100),size=100) for j1 in range(10)])
    ]

    biases = [
        np.array([]),
        np.random.normal(size=100),
        np.random.normal(size=10)]

    y = [
        np.array([0.0 for i4 in range(784)]),
        np.array([0.0 for i5 in range(100)]),
        np.array([0.0 for i6 in range(10)])
    ]

    z = [
        np.array([0.0 for i7 in range(784)]),
        np.array([0.0 for i8 in range(100)]),
        np.array([0.0 for i9 in range(10)])
    ]

    errors = [
        np.array([0.0 for i0 in range(784)]),
        np.array([0.0 for i11 in range(100)]),
        np.array([0.0 for i12 in range(10)])
    ]

    d_weights = [np.array([])
        , np.array([[0.0 for i13 in range(784)] for j4 in range(100)])
        , np.array([[0.0 for i14 in range(100)] for j5 in range(10)])
        ]

    d_biases = [
        np.array([]),
        np.array([0.0 for i15 in range(100)])
        , np.array([0.0 for i16 in range(10)])]

    batch_size = 1
    learn_rate = 3

    regularization_rate = 0.1

    n = None

    def train(self, train_set, iterations, learn_rate, batch_size, regularization_rate):
        self.n = len(train_set)//batch_size
        self.regularization_rate = regularization_rate
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        for iteration in range(iterations):
            #   Split into mini batches
            batches = split_into_batches(train_set, batch_size)
            for batch in batches:
                for tuple in batch:
                    inputs = tuple[0]
                    target = tuple[1]
                    # Grab the inputs and feed them through the network
                    self.y[0] = np.array(inputs)
                    # Feed forward the entry
                    self.classifyInstance()

                    # Compute the error for the last layer
                    self.computeErrorLastLayer(target)

                    # Backpropagate the error through the network
                    # Store the temporary results to d_weighs and d_biases
                    self.backpropagate()

                #         The current batch has finished
                self.commitChanges()
            print("Iteration %d done" % iteration)
            correct = 0
            incorrect = 0
            for entry in train_set:
                inputs = entry[0]
                target = entry[1]
                digit = self.classifyDigit(inputs)
                if digit == target:
                    correct+=1
                else:
                    incorrect+=1

            print("[TRAIN]Correct instances : ", correct)
            print("[TRAIN]Incorrect instances : ", incorrect)
            print("[TRAIN]Precision :", correct / (correct + incorrect))

    def classifyDigit(self, inputs):
        self.y[0] = np.array(inputs)
        self.classifyInstance()
        #todo check why y[0] contained 355 for example
        results = list(self.y[-1])
        recognised_digit = results.index(max(results))

        return recognised_digit

    def classifyInstance(self):
        for i in range(1,3):
            for j in range(len(self.z[i])):
                self.z[i][j] = np.dot(self.weights[i][j], self.y[i-1]) + self.biases[i][j]
            if i == 2:
                # self.y[i] = sigmoid(self.z[i])
                self.y[i] = softmax(self.z[i])
            else:
                self.y[i] = sigmoid(self.z[i])

    def computeErrorLastLayer(self, target):
        for i in range(10):
            t = int(target == i)
            self.errors[-1][i] = self.y[-1][i] - t

    def commitChanges(self):
        self.d_weights = np.multiply(self.d_weights, self.learn_rate/self.batch_size)
        self.d_biases = np.multiply(self.d_biases, self.learn_rate/self.batch_size)
        self.weights = np.add(np.multiply(self.weights, (1.0 - self.learn_rate * self.regularization_rate/ self.n)), -self.d_weights)
        self.biases = np.add(self.biases, -self.d_biases)
        self.d_weights = [np.array([])
            , np.array([[0.0 for i13 in range(784)] for j4 in range(100)])
            , np.array([[0.0 for i14 in range(100)] for j5 in range(10)])
            ]

        self.d_biases = [
            np.array([]),
            np.array([0.0 for i in range(100)])
            , np.array([0.0 for i in range(10)])]

    def backpropagate(self):
#         The error is computed for the last layer
        for i in range(1,-1,-1):
            self.errors[i] = np.dot(self.errors[i+1], self.weights[i+1]) * sigmoid_derived(self.z[i])
            # Compute the difference in weights and bias
            # todo maybe find a way to do this in numpy directly
            for j in range(len(self.d_weights[i+1])):
                self.d_weights[i+1][j] = np.add(self.d_weights[i+1][j], np.multiply(self.y[i], self.errors[i+1][j]))
            self.d_biases[i+1] = np.add(self.d_biases[i+1], self.errors[i+1])

        pass


if __name__ == '__main__':

    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    # Create an empty network with 784 inputs, 100 neurons in the hidden layer and 10 in the output layer
    network = Network()
    # print(zip(train_set[0], train_set[1]))
    reduced_train_set = list(zip(train_set[0],train_set[1]))
    reduced_train_set = reduced_train_set[:len(reduced_train_set)]

    network.train(reduced_train_set, 1, 0.5, 5, 0.1)

    correct = 0
    incorrect = 0
    for inp,target in zip(test_set[0],test_set[1]):
        digit = network.classifyDigit(inputs = inp)
        if digit == target:
            correct+=1
        else:
            incorrect+=1

    print("[TEST]Correct instances : ", correct)
    print("[TEST]Incorrect instances : ", incorrect)
    print("[TEST]Precision :", correct/(correct + incorrect))