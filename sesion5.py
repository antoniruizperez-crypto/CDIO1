from PIL import Image
import numpy as np

imagen = Image.open("Imagenes/foto1prueba.jpg").convert("RGB")
datos = np.array(imagen).astype(np.float32)

r = datos[:, :, 0]
g = datos[:, :, 1]
b = datos[:, :, 2]


# -----------------------------------------
# 1. BRILLO
# -----------------------------------------

brillo = (r + g + b) / 3


# -----------------------------------------
# 2. SATURACIÓN
# -----------------------------------------

maximo = np.max(datos, axis=2)
minimo = np.min(datos, axis=2)

saturacion = maximo - minimo


# -----------------------------------------
# 3. DETECTAR MAR
# -----------------------------------------
#
# El mar normalmente es más oscuro que las nubes.
#
# Esta condición es SOLO un primer filtro.
#

mar = (
    (brillo < 80)
)


# -----------------------------------------
# 4. DETECTAR NUBES
# -----------------------------------------
#
# Las nubes:
# - son relativamente brillantes
# - tienen poca saturación
#

nubes = (
    (brillo > 80) &
    (saturacion < 80) &
    (~mar)
)


# -----------------------------------------
# 5. PORCENTAJE
# -----------------------------------------

porcentaje = np.mean(nubes) * 100

print(f"Porcentaje detectado: {porcentaje:.2f}%")


# -----------------------------------------
# 6. MÁSCARA VISUAL
# -----------------------------------------

resultado = np.zeros_like(datos, dtype=np.uint8)

resultado[nubes] = [255, 0, 0]

Image.fromarray(resultado).save("nubes_detectadas.jpg")

print("Se ha creado: nubes_detectadas.jpg")

print("Tamaño:", imagen.size)
print("Modo:", imagen.mode)