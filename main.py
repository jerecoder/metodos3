import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):  # puedes agregar más extensiones si es necesario
            img = Image.open(os.path.join(folder, filename))
            img_data = np.array(img)
            images.append(img_data)
    return images

# Cargar todas las imágenes de la carpeta "dataset_imagenes"
folder_path = "parapelotudear"
all_images = load_images_from_folder(folder_path)

# Si quieres convertir todas las imágenes en vectores:
image_vectors = [img.ravel() for img in all_images]

# Para apilar todos los vectores en una matriz:
image_matrix = np.vstack(image_vectors)

U, S, Vt = np.linalg.svd(image_matrix, )

print(S)

def reduce_dimension(matrix, U, S, Vt, d):
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]

    return np.dot(U_d, np.dot(S_d, Vt_d))

d = min(image_matrix.shape[0], 12)  # Esto garantiza que d no sea mayor que el número de imágenes
reduced_matrix = reduce_dimension(image_matrix, U, S, Vt, d)


def reconstruct_image(vector, shape):
    return vector.reshape(shape)

# Dimensiones originales de las imágenes
original_shape = all_images[0].shape

# Reconstruye las imágenes reducidas
reconstructed_images = [reconstruct_image(row, original_shape) for row in reduced_matrix]

# Muestra las imágenes originales y reconstruidas lado a lado
for original, reconstructed in zip(all_images, reconstructed_images):
    plt.figure(figsize=(5, 2))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title("Reconstruida")
    plt.axis('off')

    plt.show()
