import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder



path = 'D:/Documentos/diego/universidad/4 Curso/TFG/Redes Neuronales/gameData.csv'

df = pd.read_csv(path) # nos lo devuelve con variables categóricas

# Creamos un objeto OneHotEncoder
encoder = OneHotEncoder(sparse_output=False) # HAY QUE PROBAR SI ES MEJOR TRUE (matriz dispersa) O FALSE (matriz densa)

columns_to_encode = ['pacmanLastMoveMade', 'ghost1LastMove', 'ghost2LastMove', 'ghost3LastMove', 'ghost4LastMove',] # son las columnas que quiero codificar

one_hot_encoded = encoder.fit_transform(df[columns_to_encode])

# hay que convertir la salida a un DataFrame de Pandas
encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns_to_encode))

# convertir las variables categoricas (True/False) en 0 y 1
boolean_col = ['pacmanWasEaten', 'ghost1Eaten', 'ghost2Eaten', 'ghost3Eaten', 'ghost4Eaten']
df[boolean_col] = df[boolean_col].astype(int) # convierto en 0 y 1

# Concatenar el DataFrame codificado con las columnas restantes del DataFrame original
df_final_encoded = pd.concat([df.drop(columns=columns_to_encode), encoded_df], axis=1) # elimino las columnas categoricas originales

#print(df_final_encoded.head())

# Guardar el DataFrame combinado en un archivo CSV
# output_path = 'D:/Documentos/diego/universidad/4 Curso/TFG/Redes Neuronales/gameData_preprocesado.csv'
# df_final_encoded.to_csv(output_path, index=False)

# print(f"\nEl DataFrame preprocesado se ha guardado en: {output_path}")

# Convertimos el dataframe a un tensor de Pytorch
tensor_data = torch.tensor(df_final_encoded.values, dtype=torch.float32)


