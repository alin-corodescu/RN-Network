import numpy as np

def split_into_batches(train_set):
    pass


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# a = np.array([0,1,2])
#
# print(sigmoid(a))
#
# # self.errors[i] = np.multiply(self.weights[i + 1], self.errors[i + 1]) * sigmoid_derived(self.z[i])
#

#
# print(np.dot(w,d) *z)
#
# print(np.multiply([[1,2],[3,4]],[1,2]))

# self.d_weights[i + 1] = np.add(self.d_weights[i + 1], np.dot(self.y[i], self.errors[i + 1]))

w = np.array([[[1,2],[3,4]]])
d = np.array([1,2])
y = np.array([5,6])
print(w[0][0])
print(np.multiply(y,d[0]))
w = np.add(w, -np.multiply(y,d) * 0.5)
print(w)