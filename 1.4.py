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
    return d, error, reconstruction

np.random.seed(0)
image_matrix = all_images[0]

d, error, reconstructed_image = find_minimal_d(image_matrix)

plt.imshow(image_matrix, cmap='gray')
plt.title(f'Reconstructed Image with d={d}')
plt.colorbar()
plt.tight_layout()
plt.show()
