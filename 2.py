import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load the data
data_df = pd.read_csv('dataset03.csv')
data = data_df.values[:, 1:]  # Exclude the first column if it's an identifier

# Center the data
mean_data = np.mean(data, axis=0)
data = data - mean_data

# Compute the Singular Value Decomposition (SVD) of the centered data
U, S, Vt = np.linalg.svd(data)

print(S)
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
sigma = np.mean(pdist(data)) / 2

# Perform PCA and Random Projections for different dimensions and calculate similarities
d_values = [2, 4, 6, 20, 50 ,data.shape[1]]
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
 


# Load the label data from y3.txt
y_data = pd.read_csv('y3.txt', header=None)

# Split the label data into a training set (70%) and a test set (30%)
y_train, y_test = train_test_split(y_data, test_size=0.3, random_state=0)

# Split the X data into train and test sets using the same indices as y
X_train, X_test = data[y_train.index], data[y_test.index]

# Center the X data (only using the training data for calculating the mean to avoid data leakage)
mean_X_train = np.mean(X_train, axis=0)
X_train_centered = X_train
X_test_centered = X_test  # Also center the test data using the training mean

# Compute SVD of the centered training data
U_train, S_train, Vt_train = np.linalg.svd(X_train_centered, full_matrices=False)

# Define a function to perform linear regression and calculate errors
def calculate_errors_for_dimension(U, S, Vt, X_train, y_train, X_test, y_test, d):
    # Reduce data using the top d SVD components
    X_train_reduced = np.dot(U[:, :d], np.diag(S[:d]))
    X_test_reduced = np.dot(X_test, Vt[:d].T)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train_reduced, y_train)

    # Predict and calculate training error
    y_train_pred = model.predict(X_train_reduced)
    train_error = mean_squared_error(y_train, y_train_pred)

    # Predict and calculate test error
    y_test_pred = model.predict(X_test_reduced)
    test_error = mean_squared_error(y_test, y_test_pred)

    return train_error, test_error
# Calculate errors for different dimensions
dimensions = [2,4,6,20,50,106]
train_errors = []
test_errors = []

for d in dimensions:
    train_error, test_error = calculate_errors_for_dimension(U_train, S_train, Vt_train, 
                                                            X_train_centered, y_train, 
                                                            X_test_centered, y_test, d)
    train_errors.append(train_error)
    test_errors.append(test_error)

# Plotting the errors
plt.figure(figsize=(12, 6))
plt.plot(dimensions, train_errors, marker='o', linestyle='-', color='blue', label='Train Error')
plt.plot(dimensions, test_errors, marker='s', linestyle='--', color='red', label='Test Error')
plt.xlabel('Dimensions')
plt.ylabel('Mean Squared Error')
plt.title('Train and Test Errors for Different Dimensions using SVD')
plt.legend()
plt.grid(True)
plt.show()


 

# Compute the squared loadings for the V matrix (features x components)
loadings_squared = Vt ** 2  # Vt is already components x features for SVD in numpy, so no need to transpose

# Weight them by the squared singular values (which represent the variance explained by each PC)
# Since singular values are by component, we multiply across rows, and S needs to be reshaped to multiply correctly
weighted_squared_loadings = loadings_squared * (S.reshape(-1, 1) ** 2)

# Now let's plot the weighted squared loadings for each feature, summing across all components
feature_contributions = np.sum(weighted_squared_loadings, axis=0)


# Sum for each feature (across all principal components)
feature_importances = np.sum(weighted_squared_loadings, axis=0)


print(feature_importances)
# Order features by their importances
ordered_feature_indices = np.argsort(feature_importances)[::-1]  # Descending order

# Ordered feature importances
ordered_feature_importances = feature_importances[ordered_feature_indices]


print(ordered_feature_indices+1)

# Ensure there's variance among the importances
importance_variance = np.var(ordered_feature_importances)


# Plot
plt.figure(figsize=(14, 7))
plt.bar(x=range(feature_contributions.shape[0]), height=feature_contributions, color='skyblue')
plt.title('Weighted Squared Loadings per Feature')
plt.xlabel('Feature index')
plt.ylabel('Weighted Squared Loadings')
plt.yscale('log')  # Using a log scale due to large range of values
plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_df = pd.read_csv('dataset03.csv')

data = data_df.values[:,1:]

if np.isnan(data).any():
    column_means = np.nanmean(data, axis=0)
    data = np.where(np.isnan(data), column_means, data)

mean_data = np.mean(data, axis=0)
centered_data = data - mean_data

U, S, Vt = np.linalg.svd(centered_data)

def reduce_data(U, S, Vt, d):
    return np.dot(U[:, :d], np.diag(S[:d]))

d_values = [2, 4, 6, 20, centered_data.shape[1]]
reduced_data_dict = {d: reduce_data(U, S, Vt, d) for d in d_values}




print(list(map(int, S)))
plt.scatter(reduced_data_dict[2][:, 0], reduced_data_dict[2][:, 1], alpha=0.5)
plt.title("2D Projection of Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()