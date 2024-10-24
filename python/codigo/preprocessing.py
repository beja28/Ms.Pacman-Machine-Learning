import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Función de preprocesamiento del CSV con mapeo de 'PacmanMove' a números
def preprocess_csv(path):
    df = pd.read_csv(path)
    
    # Mapeo de la columna 'PacmanMove' a números
    move_mapping = {
        'UP': 0,
        'DOWN': 1,
        'LEFT': 2,
        'RIGHT': 3,
        'STOP': 4
    }
    
    # Aplicamos el mapeo
    df['PacmanMove'] = df['PacmanMove'].map(move_mapping)
    
    # One-hot encoding para variables categóricas
    encoder = OneHotEncoder(sparse_output=False)
    columns_to_encode = ['pacmanLastMoveMade', 'ghost1LastMove', 'ghost2LastMove', 'ghost3LastMove', 'ghost4LastMove']
    one_hot_encoded = encoder.fit_transform(df[columns_to_encode])
    
    # Convertimos a DataFrame
    encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns_to_encode))
    
    # Variables booleanas a 0 y 1
    boolean_col = ['pacmanWasEaten', 'ghost1Eaten', 'ghost2Eaten', 'ghost3Eaten', 'ghost4Eaten']
    df[boolean_col] = df[boolean_col].astype(int)
    
    # Concatenamos el DataFrame codificado con el resto de columnas
    df_final_encoded = pd.concat([df.drop(columns=columns_to_encode), encoded_df], axis=1)
    
    # Variables independientes (X) y dependientes (Y)
    X = df_final_encoded.drop(columns=['PacmanMove']) 
    Y = df_final_encoded['PacmanMove']
    
    return X, Y