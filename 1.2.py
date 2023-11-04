import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(folder, filename))
            img_data = np.array(img)
            images.append(img_data)
    return images

# Load all images from the folder "dataset_imagenes"
folder_path = "dataset_imagenes"
all_images = load_images_from_folder(folder_path)

# Convert all images to vectors
image_vectors = [img.ravel() for img in all_images]

# Stack all vectors into a matrix
image_matrix = np.vstack(image_vectors)

U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)

def dTail(matrix, U, S, Vt, d):
    U_d = U[:, -d:]
    S_d = np.diag(S[-d:])
    Vt_d = Vt[-d:, :]
    return np.dot(U_d, np.dot(S_d, Vt_d))

def dPrincipal(matrix, U, S, Vt, d):
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    return np.dot(U_d, np.dot(S_d, Vt_d))

def reduce_dimensionality(X, VT, d):
    return X.dot(VT[:d, :].T)


d = min(image_matrix.shape[0], 6)
reduced_matrix = dPrincipal(image_matrix, U, S, Vt, d)

def reconstruct_image(vector, shape):
    return vector.reshape(shape)

# Original image dimensions
original_shape = all_images[0].shape

# Reconstruct the reduced images
reconstructed_images = [reconstruct_image(row, original_shape) for row in reduced_matrix]

# Display original and reconstructed images together in a big plot
n = len(all_images)
fig, axes = plt.subplots(n//4, 8, figsize=(20, 5*(n//4)))

for i in range(0, n, 4):
    row_idx = i//4
    # First set: Original and Reconstructed
    axes[row_idx, 0].imshow(all_images[i], cmap='gray')
    axes[row_idx, 0].axis('off')

    axes[row_idx, 1].imshow(reconstructed_images[i], cmap='gray')
    axes[row_idx, 1].axis('off')

    # Second set: Original and Reconstructed
    axes[row_idx, 2].imshow(all_images[i+1], cmap='gray')
    axes[row_idx, 2].axis('off')

    axes[row_idx, 3].imshow(reconstructed_images[i+1], cmap='gray')
    axes[row_idx, 3].axis('off')

    # Third set: Original and Reconstructed
    axes[row_idx, 4].imshow(all_images[i+2], cmap='gray')
    axes[row_idx, 4].axis('off')

    axes[row_idx, 5].imshow(reconstructed_images[i+2], cmap='gray')
    axes[row_idx, 5].axis('off')

    # Fourth set: Original and Reconstructed
    axes[row_idx, 6].imshow(all_images[i+3], cmap='gray')
    axes[row_idx, 6].axis('off')

    axes[row_idx, 7].imshow(reconstructed_images[i+3], cmap='gray')
    axes[row_idx, 7].axis('off')

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()
