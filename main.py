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

# Cargar todas las imágenes de la carpeta "dataset_imagenes"
folder_path = "dataset_imagenes"
all_images = load_images_from_folder(folder_path)

# Convert all images to vectors
image_vectors = [img.ravel() for img in all_images]

# Stack all vectors into a matrix
image_matrix = np.vstack(image_vectors)

U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)

def dPrincipal(matrix, U, S, Vt, d):
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    return np.dot(U_d, np.dot(S_d, Vt_d))

def dTail(matrix, U, S, Vt, d):
    U_d = U[:, -d:]   # Get the last d columns
    S_d = np.diag(S[-d:])  # Get the smallest d singular values
    Vt_d = Vt[-d:, :]  # Get the last d rows
    return np.dot(U_d, np.dot(S_d, Vt_d))


d = min(image_matrix.shape[0], 16)  # Ensure d isn't larger than the number of images
reduced_matrix = dTail(image_matrix, U, S, Vt, d)

def reconstruct_image(vector, shape):
    return vector.reshape(shape)

# Original image dimensions
original_shape = all_images[0].shape

# Reconstruct the reduced images
reconstructed_images = [reconstruct_image(row, original_shape) for row in reduced_matrix]

# Display original and reconstructed images side by side
for original, reconstructed in zip(all_images, reconstructed_images):
    plt.figure(figsize=(5, 2))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

    plt.show()
