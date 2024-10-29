import pandas as pd
import os

#df = pd.read_csv('01_gameStatesData.csv')


# Obtener la ruta de la carpeta 'codigo' donde está 'main.py'
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta que sube dos niveles desde 'codigo' y entra en 'DataSets'
dataset_path = os.path.join(directorio_actual, '..', 'DataSets', 'gameData_PRUEBAS.csv')

# Normalizar la ruta para evitar problemas con distintos sistemas operativos
dataset_path = os.path.normpath(dataset_path)

df = pd.read_csv(dataset_path)


valores_unicos = sorted(set(df['pacmanCurrentNodeIndex'].unique()))

 
print(valores_unicos)
