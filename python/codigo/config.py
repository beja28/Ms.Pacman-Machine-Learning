import os

class Config:
    def __init__(self):
        # --- CREAR PATHS USANDO RUTAS RELATIVAS ---

        # Obtener la ruta de la carpeta 'codigo' donde está 'main.py'
        directorio_actual = os.path.dirname(os.path.abspath(__file__))

        # Construir la ruta que sube dos niveles desde 'codigo' y entra en 'DataSets'
        dataset_path = os.path.join(directorio_actual, '..', '..', 'DataSets', '01_gameStatesData.csv')

        # Normalizar la ruta para evitar problemas con distintos sistemas operativos
        dataset_path = os.path.normpath(dataset_path)

        # Construir la ruta hacia la carpeta 'Redes_Entrenadas'
        path_trained = os.path.join(directorio_actual, 'Redes_Entrenadas')

        # Normalizar la ruta para evitar problemas con distintos sistemas operativos
        path_trained = os.path.normpath(path_trained)
