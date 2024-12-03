import pandas as pd
from sklearn.preprocessing import OneHotEncoder


columns_to_encode = ['pacmanLastMoveMade', 'ghost1LastMove', 'ghost2LastMove', 'ghost3LastMove', 'ghost4LastMove']
boolean_col = ['pacmanWasEaten', 'ghost1Eaten', 'ghost2Eaten', 'ghost3Eaten', 'ghost4Eaten']

categories = {
        "pacmanLastMoveMade": ["DOWN", "LEFT", "NEUTRAL", "RIGHT", "UP"],
        "ghost1LastMove": ["DOWN", "LEFT", "NEUTRAL", "RIGHT", "UP"],
        "ghost2LastMove": ["DOWN", "LEFT", "NEUTRAL", "RIGHT", "UP"],
        "ghost3LastMove": ["DOWN", "LEFT", "NEUTRAL", "RIGHT", "UP"],
        "ghost4LastMove": ["DOWN", "LEFT", "NEUTRAL", "RIGHT", "UP"],
    } 

# Función de preprocesamiento del CSV con mapeo de 'PacmanMove' a números
def preprocess_csv(path):
    df = pd.read_csv(path)
    
    # Mapeo de la columna 'PacmanMove' a números
    move_mapping = {
        'UP': 0,
        'DOWN': 1,
        'LEFT': 2,
        'RIGHT': 3,
        'NEUTRAL': 4
    }
    
    # Aplicamos el mapeo
    df['PacmanMove'] = df['PacmanMove'].map(move_mapping)
    
    #print(df)
    encoder = OneHotEncoder(categories=[categories[col] for col in columns_to_encode], sparse_output=False, drop=None)
    
    one_hot_encoded = encoder.fit_transform(df[columns_to_encode])
        
    # Convertimos a DataFrame
    encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns_to_encode))
    
    
    # Variables booleanas a 0 y 1
    df[boolean_col] = df[boolean_col].astype(int)
    
    # Concatenamos el DataFrame codificado con el resto de columnas
    df_final_encoded = pd.concat([df.drop(columns=columns_to_encode), encoded_df], axis=1)
    
    # df_final_encoded.to_csv('D:/Documentos/diego/Universidad/4 Curso/pruebaCsv.csv', index =False)

        
    # Dividimos el dataframe por intersecciones

    grouped_df = df_final_encoded.groupby(['pacmanCurrentNodeIndex'])

    return grouped_df

def preprocess_game_state(game_state, path):
    
    
    # Leer el CSV y obtener los nombres de las columnas
    df_columns = pd.read_csv(path, nrows=0)
    columns_csv = df_columns.columns.tolist()
    columns_csv = columns_csv[1:] # La primera columna (pacmanMove) sobra
    

    # Guardo cada elemento en un vector
    data = game_state.split(',')

    if len(columns_csv) != len(data):
        print("Distinto numero de datos que de columnas")

    data_dict = {}
    # Llenar el diccionario con los nombres de las columnas como claves
    for i in range(len(columns_csv)):
        data_dict[columns_csv[i]] = data[i]
        
    state = convert_types(data_dict) # Devuelve los datos parseados
    
    subset_df = pd.DataFrame([state])
    
    encoder = OneHotEncoder(categories=[categories[col] for col in columns_to_encode], sparse_output=False, drop=None)

    one_hot_encoded = encoder.fit_transform(subset_df[columns_to_encode])
    
    
    # Convertimos a DataFrame
    encoded_state = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns_to_encode))
    
    #encoded_state.to_csv('D:/Documentos/diego/Universidad/4 Curso/pruebaGS.csv', index =False)
    
    
    # Casteo las variables booleanas
    for key in boolean_col:
        if key in state:
            state[key] = int(state[key])
    
    # Crear un nuevo diccionario sin las columnas que fueron codificadas
    filtered_state = {}

    # Iterar sobre cada clave-valor en state
    for k, v in data_dict.items():
        # Agregar al nuevo diccionario solo si la clave no está en columns_to_encode
        if k not in columns_to_encode:
            filtered_state[k] = v

    # Si hay claves duplicadas entre los dos diccionarios, se sobreescriben los valores de encoded_state sobre filtered_state
    combined_state = {**filtered_state, **encoded_state}
        
    print(list(combined_state.keys()))
    return list(combined_state.values())

    
# Función para convertir tipos de datos
def convert_types(data_dict):

    # Convertir las claves booleanas
    for key in boolean_col:
        if key in data_dict:
            data_dict[key] = bool(data_dict[key])

    # Convertir otras claves a enteros
    for key in data_dict:
        if key not in boolean_col and key not in columns_to_encode:
            data_dict[key] = int(data_dict[key])

    return data_dict