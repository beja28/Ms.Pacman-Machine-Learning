import os
import socket
import joblib
import torch
from preprocessing import preprocess_game_state
from model_pytorch import MyModelPyTorch
from main import n_features, n_classes 
from config import Config


# --- CREAR PATHS USANDO RUTAS RELATIVAS ---

# Obtener la ruta de la carpeta 'codigo' donde está 'main.py'
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta que sube dos niveles desde 'codigo' y entra en 'DataSets'
dataset_path = os.path.join(directorio_actual, '..', '..', 'DataSets', '03_gameStatesData.csv')

# Normalizar la ruta para evitar problemas con distintos sistemas operativos
dataset_path = os.path.normpath(dataset_path)

# Construir la ruta hacia la carpeta 'Redes_Entrenadas'
path_trained = os.path.join(directorio_actual, 'Redes_Entrenadas')

# Normalizar la ruta para evitar problemas con distintos sistemas operativos
path_trained = os.path.normpath(path_trained)



# """
#     Cargar modelo sklearn
# """
# # Cargar el modelo
# model_filename = 'mlp_trained_model_2024-10-24.pkl'  # Cambia 'mi_modelo' por el nombre que desees
# full_model_path = os.path.join(path_trained, model_filename)

# # Cargar el modelo entrenado
# mlp_model = joblib.load(full_model_path)


"""
    Cargar modelo PyTorch
"""
model_filename = 'pytorch_model_2024-10-31.pth'
full_model_path = os.path.join(path_trained, model_filename)
modelPytorch = MyModelPyTorch(n_features, n_classes)

modelPytorch.load_state_dict(torch.load(full_model_path, weights_only=True))
modelPytorch.eval()

#Configuración
HOST = 'localhost'  # Podemos poner una IP o localhost
PORT = 12345        # Puerto que queramos usar

#Crear socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))

#Escuchar conexiones entrantes
server_socket.listen()

print(f"Servidor escuchando en {HOST}:{PORT}")

#Aceptar conexión
conn, addr = server_socket.accept()
print(f"Conectado con {addr}")

#Recibir datos
while True:
    data = conn.recv(1024)  # El 1024 es el número máximo de bytes que se intenta recibir
    if not data:
        break

    mensaje = data.decode('utf-8')
    print(f"Datos recibidos: {mensaje}")
    
    preprocessed_state = preprocess_game_state(mensaje, dataset_path)
    
    #print(preprocessed_state)
    
    # Convertir el estado preprocesado a tensor
    input_tensor = torch.tensor(preprocessed_state.values, dtype=torch.float32).unsqueeze(0)
        
    # Realizar la predicción
    with torch.no_grad():  # No calcular gradientes
        prediccion = modelPytorch(input_tensor)
    
    # Seleccionar el índice con el valor máximo para obtener el movimiento predicho
    predicted_index = torch.argmax(prediccion)
    
    moves = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'NEUTRAL']
    predicted_move = moves[predicted_index]

    # Responder con algo diferente según lo recibido
    if mensaje:
        respuesta = f"{predicted_move}\n"  # Agrega un salto de línea al final
        conn.sendall(respuesta.encode('utf-8'))  # Enviar respuesta codificada en UTF-8

conn.close()