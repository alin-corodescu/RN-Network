import numpy as np

def split_into_batches(train_set):
    pass


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class Neuron :
    w = np.array([])
    b = 0
    z = 0
    output = 0
    error = 0
    b_staged_change = 0
    w_staged_change = 0
    change_count = 0
    target = 0
    activation_function = sigmoid

    def __init__(self, weight_count = 0):
        self.w = np.array([np.random.normal() for i in range(weight_count)])
        # todo check if this is how bias is supposed to be initialized
        self.b = np.random.normal()

    def activate(self, inputs):
        self.z = np.dot(self.w, inputs) + self.b
        self.output = self.activation_function(self.z)

    def computeError(self, actual):
        t = int(actual == self.target)
        self.error = self.output * (1 - self.output) * (self.output - t)



class Network :
    layer_count = 0
    layers = []

    def __init__(self, *args):
        self.layer_count = len(args)
        for size in args:
            if len(self.layers):
                self.layers.append([Neuron(len(self.layers[-1])) for i in range(size)])
            else:
                self.layers.append([Neuron() for i in range(size)])

        for i in range(len(self.layers[-1])):
            neuron = self.layers[-1][i]
            neuron.target = i
            # Todo : make last layer to use softmax activation function

    def classifyInstance(self, entry):
        # Initialize output from the first neurons to be equal to the input values
        for neuron, value in zip(self.layers[0], entry[0]):
            neuron.output = value


        # For each subsequent layers, activate the neurons
        for i in range(1, self.layer_count):
            inputs = [neuron.output for neuron in self.layers[i - 1]]
            for neuron in self.layers[i]:
                neuron.activate(inputs)

    def train(self, train_set, iterations, learn_rate):
        for iteration in range(iterations):
        #   Split into mini batches
            batches = split_into_batches(train_set)
            for batch in batches:
                # TODO : reset every neuron's weight and bias delta accumulated during the last run
                for entry in batch:
                    # Check if this is actually the target digit
                    target = entry[1]
                    # Feed forward the entry
                    self.classifyInstance(entry)

                    # Compute the error for the last layer
                    self.computeErrorLastLayer(target)

                    # Backpropagate the error through the network
                    # The individual neurons will hold the information about how their weights and biases should be adjusted
                    self.backpropagate()

        #         The current batch has finished
                self.commitChanges()

        pass
    # Computes the error for each neuron in the last layer
    def computeErrorLastLayer(self, actual):
        for i in range(len(self.layers[-1])) :
            neuron = self.layers[-1][i]
            neuron.computeError(actual)

    def backpropagate(self):
        pass

    def commitChanges(self):
        pass



if __name__ == '__main__':
    # Create an empty network with 784 inputs, 100 neurons in the hidden layer and 10 in the output layer
    network = Network(784,100,10)
    network.train([] ,30 ,3 )