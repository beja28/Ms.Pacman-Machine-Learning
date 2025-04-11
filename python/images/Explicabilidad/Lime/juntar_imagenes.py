from PIL import Image

# Carga las dos imágenes
img1 = Image.open('LIME_mlp_trained_model_2025-03-12_(165,).pkl_2025-04-09_19-21-20.png')
img2 = Image.open('LIME_mlp_trained_model_2025-03-12_(165,).pkl_2025-04-09_19-30-57.png')

# Ajustamos al mismo tamaño (por ejemplo, al tamaño de la primera)
# Si prefieres redimensionar ambas a un tamaño específico, puedes cambiar esto
img2 = img2.resize(img1.size)

# Parámetro de separación
espaciado = 50

# Tamaño de cada imagen (ya iguales)
w, h = img1.size

# Crear nueva imagen con espacio en medio
combined = Image.new('RGB', (w * 2 + espaciado, h), color=(255, 255, 255))  # fondo blanco

# Pegar imágenes con separación
combined.paste(img1, (0, 0))
combined.paste(img2, (w + espaciado, 0))

# Guardar imagen final
combined.save('imagen_combinada_con_espacio.png')