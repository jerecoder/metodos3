import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import seaborn as sns

X = np.loadtxt('dataset03.csv', delimiter=',')

U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
def reduce_dimensionality(X, VT, d):
    return X.dot(VT[:d, :].T)

dimensions = [2, 4, 6, 20]

reduced_data = {d: reduce_dimensionality(X, VT, d) for d in dimensions}

print(reduced_data)

sigma = 1
def gaussian_kernel(x_i, x_j, sigma):
    distance_squared = np.sum((x_i - x_j) ** 2)
    return np.exp(-distance_squared / (2 * sigma ** 2))

n = len(reduced_data[2])
d_to_plot = 2
similarity_matrix = np.zeros((n, n))
for x in range(n):
    for y in range(n):
        similarity_matrix[x][y] = gaussian_kernel(reduced_data[d_to_plot][x], reduced_data[d_to_plot][y], sigma)

similarity_df = pd.DataFrame(similarity_matrix)
sns.heatmap(similarity_df, cmap='viridis')
plt.show()
