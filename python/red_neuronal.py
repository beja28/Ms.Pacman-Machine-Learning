import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


########
#no funciona
# from google.colab import drive
# drive.mount('/content/drive')

# path = '/content/drive/MyDrive/gamedata.csv'

path = 'D:/Documentos/diego/universidad/4 Curso/TFG/Redes Neuronales/gameData.csv'

df = pd.read_csv(path) # nos lo devuelve con variables categóricas

# Creamos un objeto OneHotEncoder
encoder = OneHotEncoder(sparse_output=False) # HAY QUE PROBAR SI ES MEJOR TRUE (matriz dispersa) O FALSE (matriz densa)

columns_to_encode = ['pacmanLastMoveMade', 'ghost1LastMove', 'ghost2LastMove', 'ghost3LastMove', 'ghost4LastMove', 'NextPacmanMove'] # son las columnas que quiero codificar

one_hot_encoded = encoder.fit_transform(df[columns_to_encode])

# hay que convertir la salida a un DataFrame de Pandas
encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns_to_encode))

# convertir las variables categoricas (True/False) en 0 y 1
boolean_col = ['pacmanWasEaten', 'ghost1Eaten', 'ghost2Eaten', 'ghost3Eaten', 'ghost4Eaten']
df[boolean_col] = df[boolean_col].astype(int) # convierto en 0 y 1

# Concatenar el DataFrame codificado con las columnas restantes del DataFrame original
df_final_encoded = pd.concat([df.drop(columns=columns_to_encode), encoded_df], axis=1) # elimino las columnas categoricas originales

X = df_final_encoded.drop(columns=['NextPacmanMove']) 
Y = df_final_encoded['NextPacmanMove'] # cambiar a la variable de futuro movimiento

# dividimos en entrenamiento 70% y test 30% 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # ver si afecta el valor del random_state

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32)

n_features = X.shape[1] #numero de caracteristicas
n_classes = len(Y) # suponemos que el num. de clases va a ser 5 (5 posibles movimientos)
n_hidden = 34 #numero de neuronas en la capa oculta. Para sacar este numero hacemos 2/3 del num. de caracteristicas + num. de clases de salida

simple_model = nn.Sequential(nn.Linear(n_features, n_hidden),
                      nn.ReLU(),
                      nn.Linear(n_hidden, n_classes))

#Función de pérdida y optimizador
learning_rate = 0.01  # Ajuste de la tasa de aprendizaje a 0.01
loss_fn = nn.CrossEntropyLoss() # mirar que funcion de perdida es mejor
optimizer = torch.optim.Adam(simple_model.parameters(), lr=learning_rate)

"""Se entrena el modelo"""
num_epochs = 1000
log_epochs = num_epochs / 10 # cada cuantas epocas voy a imprimir un resultado
# loss_hist = [0] * num_epochs # lista almacena el valor de la perdida en cada epoca
# accuracy_hist = [0] * num_epochs #lista almacena el valor de la precision en cada epoca

for epoch in range (num_epochs):
    hatY = simple_model(X_train_tensor) # sacamos las predicciones
    loss = loss_fn(hatY, Y_train_tensor) # calculamos la perdida entre la prediccion y la Y real
    optimizer.zero_grad()
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if log_epochs % 100 == 0:
        print(f"{log_epochs} {loss.item():0.2f} {torch.nn.Softmax(dim=1)(hatY)[:, 1].detach()}")
    
    
# Paso 1: Realiza predicciones en el conjunto de prueba
with torch.no_grad():  # No necesitamos calcular gradientes para la evaluación
    Y_test_pred = simple_model(X_test_tensor)  # Obtén las predicciones

# Paso 2: Calcula la pérdida
test_loss = loss_fn(Y_test_pred, Y_test_tensor)  # Pérdida en el conjunto de prueba
print(f"Test Loss: {test_loss.item():.4f}")

# Paso 3: Calcular precisión
# La precisión se puede calcular comparando las predicciones con las etiquetas reales
_, predicted_classes = torch.max(Y_test_pred, 1)  # Obtén las clases predichas
correct_predictions = (predicted_classes == Y_test_tensor.long()).sum().item()  # Compara con las verdaderas etiquetas
accuracy = correct_predictions / len(Y_test_tensor)  # Calcula la precisión
print(f"Test Accuracy: {accuracy:.4f}")


#print(df_final_encoded.head())

# Guardar el DataFrame combinado en un archivo CSV
# output_path = 'D:/Documentos/diego/universidad/4 Curso/TFG/Redes Neuronales/gameData_preprocesado.csv'
# df_final_encoded.to_csv(output_path, index=False)

# print(f"\nEl DataFrame preprocesado se ha guardado en: {output_path}")