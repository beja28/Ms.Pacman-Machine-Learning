import pandas as pd
import os


"""
EL DATASET DEBE CONTENER 38 POSICIONES DE INTERSECCIONES DISTINTAS
"""

# Obtener la ruta de la carpeta 'codigo' donde está 'main.py'
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta que sube dos niveles desde 'codigo' y entra en 'DataSets'
dataset_path = os.path.join(directorio_actual, '..', 'DataSets', '03_gameStatesData.csv')

# Normalizar la ruta para evitar problemas con distintos sistemas operativos
dataset_path = os.path.normpath(dataset_path)

df = pd.read_csv(dataset_path)


valores_unicos = sorted(set(df['pacmanCurrentNodeIndex'].unique()))

 
print(len(valores_unicos))
