import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt

# Load the data
data_df = pd.read_csv('dataset03.csv')
data = data_df.values[:, 1:]  # Exclude the first column if it's an identifier

# Center the data
mean_data = np.mean(data, axis=0)
centered_data = data - mean_data

# Compute the Singular Value Decomposition (SVD) of the centered data
U, S, Vt = np.linalg.svd(centered_data)

# Define the similarity function
def similarity_function(xi, xj, sigma):
    return np.exp(-np.sum((xi - xj)**2) / (2 * sigma**2))

# Define a function to reduce the dimensionality using SVD components
def reduce_data(U, S, Vt, d):
    return np.dot(U[:, :d], np.diag(S[:d]))

# Define the sample size and a random seed for reproducibility
sample_size = 1999  # Adjust based on available memory and dataset size
np.random.seed(0)

# If the dataset is large, take a random sample to avoid memory issues
if data.shape[0] > sample_size:
    indices = np.random.choice(data.shape[0], sample_size, replace=False)
    sampled_data = data[indices, :]
else:
    sampled_data = data

# Choose a sigma value for the similarity function
# This can be a fraction of the mean pairwise distance or based on domain knowledge 
#Eligo este sigma como la mediana por un tema de heuristica.
sigma = np.mean(pdist(centered_data)) / 2

# Perform PCA and Random Projections for different dimensions and calculate similarities
d_values = [2, 4, 6, 20, 50 ,centered_data.shape[1]]
for d in d_values:
    # Calculate epsilon for the current dimension d
    epsilon = np.sqrt(8 * np.log(2000) / d)
    # print(sampled_data.shape[0])
    print(f"For dimension {d}, epsilon is approximately {epsilon:.4f}")

    # PCA Reduction
    pca_reduced_data = reduce_data(U[indices, :], S, Vt, d)
    pca_distances = pdist(pca_reduced_data)
    pca_similarities = np.exp(-pca_distances**2 / (2 * sigma**2))

    # Random Projections
    transformer = GaussianRandomProjection(n_components=d, random_state=0)
    rp_reduced_data = transformer.fit_transform(sampled_data)
    rp_distances = pdist(rp_reduced_data)
    rp_similarities = np.exp(-rp_distances**2 / (2 * sigma**2))

    # Compare similarities
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(pca_similarities, bins=100, alpha=0.7, label=f'PCA d={d}')
    plt.legend()
    plt.title('PCA Similarities Distribution')

    plt.subplot(1, 2, 2)
    plt.hist(rp_similarities, bins=100, alpha=0.7, label=f'RP d={d}')
    plt.legend()
    plt.title('Random Projection Similarities Distribution')

    plt.suptitle(f'Comparison of Similarities for d={d}')
    plt.show()
 


# Calculate the squared loadings for each feature across all components
loadings_squared = Vt**2  # Square Vt because it represents loadings here

# Multiply by the corresponding squared singular values
weighted_squared_loadings = loadings_squared * S[:, np.newaxis]**2

# Sum for each feature (across all principal components)
feature_importances = np.sum(weighted_squared_loadings, axis=0)

# Order features by their importances
ordered_feature_indices = np.argsort(feature_importances)[::-1]  # Descending order

# Ordered feature importances
ordered_feature_importances = feature_importances[ordered_feature_indices]

# Ensure there's variance among the importances
importance_variance = np.var(ordered_feature_importances)

# Plot the ordered feature importances to check if they show variance
plt.figure(figsize=(15, 5))
plt.bar(range(len(ordered_feature_importances)), ordered_feature_importances)
plt.xlabel('Feature index (ordered by importance)')
plt.ylabel('Feature importance')
plt.title('Feature Importances Ordered by Weight with Variance Check')
plt.show(), importance_variance


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data
# data_df = pd.read_csv('dataset03.csv')

# # Convert the dataframe to a numpy array
# data = data_df.values[:,1:]

# # Check if there are any NaN values and replace them with the mean of the respective column
# if np.isnan(data).any():
#     column_means = np.nanmean(data, axis=0)
#     data = np.where(np.isnan(data), column_means, data)

# # Center the data
# mean_data = np.mean(data, axis=0)
# centered_data = data - mean_data

# # Compute the Singular Value Decomposition (SVD) of the centered data
# U, S, Vt = np.linalg.svd(centered_data)

# # Define a function to reduce the dimensionality using SVD components
# def reduce_data(U, S, Vt, d):
#     return np.dot(U[:, :d], np.diag(S[:d]))

# # Reduce the dimensionality for each d in [2, 4, 6, 20, p]
# d_values = [2, 4, 6, 20, centered_data.shape[1]]
# reduced_data_dict = {d: reduce_data(U, S, Vt, d) for d in d_values}




# print(list(map(int, S)))
# # Visualize the 2D projection of the data
# plt.scatter(reduced_data_dict[2][:, 0], reduced_data_dict[2][:, 1], alpha=0.5)
# plt.title("2D Projection of Data")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.show()

print(ordered_feature_indices)