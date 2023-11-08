import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(folder, filename))
            img_data = np.array(img)
            images.append(img_data)
    return images

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

def compute_similarity_for_d(matrix, U, S, Vt, d):
    reduced_matrix = dPrincipal(matrix, U, S, Vt, d)
    return cosine_similarity(reduced_matrix)

def prepare_data_for_plotting(all_similarities, pairs):
    data = []
    for d, similarities in all_similarities.items():
        for pair in pairs:
            i, j = pair
            similarity = similarities[i][j]
            data.append([d, f"Image {i} & Image {j}", similarity])
    return pd.DataFrame(data, columns=['d', 'Pair', 'Similarity'])

folder_path = "dataset_imagenes"
all_images = load_images_from_folder(folder_path)

image_vectors = [img.ravel() for img in all_images]

image_matrix = np.vstack(image_vectors)

U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)

similarities_10d = compute_similarity_for_d(image_matrix, U, S, Vt, 16)

plt.figure(figsize=(10, 8))
sns.heatmap(similarities_10d, annot=True, cmap='YlGnBu', cbar=True)
plt.xlabel('Imagen')
plt.ylabel('Imagen')
plt.show()
