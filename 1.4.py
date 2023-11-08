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


def find_minimal_d(image_matrix, max_error=0.1):
    mean_image = np.mean(image_matrix, axis=0)
    image_matrix = image_matrix - mean_image
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    
    original_norm = np.linalg.norm(image_matrix, 'fro')
    
    error = original_norm
    d = 0
    
    while error > max_error * original_norm and d < len(S):
        d += 1
        S_d = np.zeros((d, d))
        np.fill_diagonal(S_d, S[:d])
        reconstruction = U[:, :d].dot(S_d).dot(Vt[:d, :])
        error = np.linalg.norm(image_matrix - reconstruction, 'fro')
    return d, error, reconstruction, Vt[:d, :]

np.random.seed(0)
image_matrix = all_images[0]

d, error, reconstructed_image, portal= find_minimal_d(image_matrix)

projected_images = [img @ portal.T @ portal for img in all_images]

n = len(all_images)
fig, axes = plt.subplots(n//4, 8, figsize=(20, 5*(n//4)))

for i in range(0, n, 4):
    row_idx = i//4
    axes[row_idx, 0].imshow(all_images[i], cmap='gray')
    axes[row_idx, 0].axis('off')

    axes[row_idx, 1].imshow(projected_images[i], cmap='gray')
    axes[row_idx, 1].axis('off')

    axes[row_idx, 2].imshow(all_images[i+1], cmap='gray')
    axes[row_idx, 2].axis('off')

    axes[row_idx, 3].imshow(projected_images[i+1], cmap='gray')
    axes[row_idx, 3].axis('off')

    axes[row_idx, 4].imshow(all_images[i+2], cmap='gray')
    axes[row_idx, 4].axis('off')

    axes[row_idx, 5].imshow(projected_images[i+2], cmap='gray')
    axes[row_idx, 5].axis('off')

    axes[row_idx, 6].imshow(all_images[i+3], cmap='gray')
    axes[row_idx, 6].axis('off')

    axes[row_idx, 7].imshow(projected_images[i+3], cmap='gray')
    axes[row_idx, 7].axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


frobenius_errors = [np.linalg.norm(all_images[i] - projected_images[i], 'fro') for i in range(len(all_images))]

print(frobenius_errors[0])
frobenius_errors_excluding_first = frobenius_errors[1:]

median_error = np.median(frobenius_errors_excluding_first)

print("Median Frobenius error (excluding first image):", median_error)
