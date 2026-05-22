import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100

# Cargar CIFAR-100
(x_train, y_train), (_, _) = cifar100.load_data()

# Seleccionar imagen aleatoria
idx = np.random.randint(0, len(x_train))

# Normalizar a [0,1]
image = x_train[idx].astype(np.float32) / 255.0

# Valores de sigma
sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]

# Crear figura 2x3
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

# Imagen original
axes[0].imshow(image, interpolation='nearest')
axes[0].set_title("Original")
axes[0].axis("off")

# Imágenes con ruido
for i, sigma in enumerate(sigmas):

    # Generar ruido gaussiano
    noise = np.random.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape
    )

    # Añadir ruido
    noisy_image = image + noise

    # Clip a rango válido
    noisy_image = np.clip(noisy_image, 0.0, 1.0)

    # Mostrar
    axes[i + 1].imshow(noisy_image, interpolation='nearest')
    axes[i + 1].set_title(f"$\\sigma$ = {sigma}")
    axes[i + 1].axis("off")

plt.tight_layout()
plt.show()

# Mostrar clase
print("Clase CIFAR-100:", y_train[idx][0])