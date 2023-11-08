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

folder_path = "dataset_imagenes"
all_images = load_images_from_folder(folder_path)

image_vectors = [img.ravel() for img in all_images]
image_matrix = np.vstack(image_vectors)

mean_vector = image_matrix.mean(axis=0)
centered_matrix = image_matrix - mean_vector

U, S, Vt = np.linalg.svd(centered_matrix, full_matrices=False)

def dTail(matrix, U, S, Vt, d):
    U_d = U[:, -d:]
    S_d = np.diag(S[-d:])
    Vt_d = Vt[-d:, :]
    return np.dot(U_d, np.dot(S_d, Vt_d))

def dPrincipal(matrix, U, S, Vt, d):
    return U[:, :d] @ np.diag(S[:d]) @ Vt[:d, :]

def reduce_dimensionality(X, VT, d):
    return X.dot(VT[:d, :].T)


d = min(image_matrix.shape[0], 2)
reduced_matrix = dPrincipal(image_matrix, U, S, Vt, d)

def reconstruct_image(vector, shape):
    return vector.reshape(shape)

original_shape = all_images[0].shape

reconstructed_images = [reconstruct_image(row, original_shape) for row in reduced_matrix]

n = len(all_images)
fig, axes = plt.subplots(n//4, 8, figsize=(20, 5*(n//4)))

for i in range(0, n, 4):
    row_idx = i//4
    axes[row_idx, 0].imshow(all_images[i], cmap='gray')
    axes[row_idx, 0].axis('off')

    axes[row_idx, 1].imshow(reconstructed_images[i], cmap='gray')
    axes[row_idx, 1].axis('off')

    axes[row_idx, 2].imshow(all_images[i+1], cmap='gray')
    axes[row_idx, 2].axis('off')

    axes[row_idx, 3].imshow(reconstructed_images[i+1], cmap='gray')
    axes[row_idx, 3].axis('off')

    axes[row_idx, 4].imshow(all_images[i+2], cmap='gray')
    axes[row_idx, 4].axis('off')

    axes[row_idx, 5].imshow(reconstructed_images[i+2], cmap='gray')
    axes[row_idx, 5].axis('off')

    axes[row_idx, 6].imshow(all_images[i+3], cmap='gray')
    axes[row_idx, 6].axis('off')

    axes[row_idx, 7].imshow(reconstructed_images[i+3], cmap='gray')
    axes[row_idx, 7].axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()
