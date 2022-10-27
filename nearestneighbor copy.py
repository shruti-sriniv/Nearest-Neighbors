import numpy as np
from numpy.linalg import norm
# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# numpy vector z of d rows, 1 column
# Output: label (+1 or -1)
def run(X,y,z):
    c = 0
    b = norm(z - X[0])
    n, d = np.shape(X)
    z = np.transpose(z)
    for t in range(1, n):
        if norm(z - X[t]) < b:
            c = t
            b = norm(z - X[t])
    label = y[c]
    return label