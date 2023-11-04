import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv('dataset03.csv').values
X = X[:, 1:]
y = np.loadtxt('y3.txt')

def find_weights(X, y, d=None):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if d is not None:
        s = s[:d]
        U = U[:, :d]
        Vt = Vt[:d, :]
    S_inv = np.linalg.inv(np.diag(s))
    X_pseudo_inverse = Vt.T @ S_inv @ U.T
    w = X_pseudo_inverse @ y
    return w

errors = []
d_values = range(1, X.shape[1] + 1)
for d in d_values:
    w = find_weights(X, y, d)
    y_pred = X @ w
    error = np.linalg.norm(y - y_pred)
    errors.append(error)

plt.plot(d_values, errors, marker='o')
plt.xlabel('Number of Singular Values (d)')
plt.ylabel('Least Squares Error')
plt.title('Least Squares Error vs. Number of Singular Values Used')
plt.grid(True)
plt.show()
