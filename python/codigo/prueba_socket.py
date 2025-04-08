import os
import socket
import joblib
import torch
import signal
import select
import sys
from preprocessing import preprocess_game_state
from model_pytorch import MyModelPyTorch
import numpy as np

""" --- CREAR PATHS USANDO RUTAS RELATIVAS --- """

# Obtener la ruta de la carpeta 'codigo' donde está 'main.py'
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta que sube dos niveles desde 'codigo' y entra en 'DataSets'
dataset_path = os.path.join(directorio_actual, '..', '..', 'DataSets', '06_gameStatesData.csv')

# Normalizar la ruta para evitar problemas con distintos sistemas operativos
dataset_path = os.path.normpath(dataset_path)

# Construir la ruta hacia la carpeta 'Redes_Entrenadas'
path_trained = os.path.join(directorio_actual, 'Redes_Entrenadas')

# Normalizar la ruta para evitar problemas con distintos sistemas operativos
path_trained = os.path.normpath(path_trained)



def model_for_prediction(model_type,n_features, n_classes):

    """
    Cargar el modelo deseado
    """
    if model_type == 'sklearn':
        model_filename= 'mlp_trained_model_2025-03-18.pkl' 
        full_model_path = os.path.join(path_trained, model_filename)
        mlp_model = joblib.load(full_model_path)  
        return mlp_model, None
    elif model_type == 'pytorch':
        model_filename = 'pytorch_model_2025-03-18.pth'
        full_model_path = os.path.join(path_trained, model_filename)
        modelPytorch = MyModelPyTorch(n_features, n_classes)
        modelPytorch.load_state_dict(torch.load(full_model_path, weights_only=True))
        modelPytorch.eval()
        return None, modelPytorch
    else:
            raise ValueError("Modelo no reconocido. Elige 'pytorch' o 'sklearn'.")


def get_prediction(model_type, mensaje, mlp_model=None, modelPytorch=None):
    
    preprocessed_state = preprocess_game_state(mensaje, dataset_path)
    
    # Convertir el estado preprocesado a tensor o array, segun el modelo
    
    if model_type == 'pytorch':
        input_tensor = torch.tensor(preprocessed_state.values, dtype=torch.float32).unsqueeze(0)  # Agregar dimensión para batch (modelo espera recibir un batch de un solo ejemplo)
        # Realizar la predicción
        with torch.no_grad():  # No calcular gradientes
            prediccion = modelPytorch(input_tensor)               
        # Seleccionar el índice con el valor máximo para obtener el movimiento predicho
        predicted_index = torch.argmax(prediccion)      
    elif model_type == 'sklearn':
        prediccion = mlp_model.predict(preprocessed_state) 
        predicted_index = prediccion[0] # Prediccion me devuelve el indice del movimiento
    

    moves = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NEUTRAL']
    predicted_move = moves[predicted_index]
    
    return predicted_move
    
    
def start_socket(model_type, n_features, n_classes):
    #Configuración
    HOST = 'localhost'  # Podemos poner una IP o localhost
    PORT = 12345        # Puerto que queramos usar

    #Crear socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen() #Escuchar conexiones entrantes
    print(f"Servidor escuchando en {HOST}:{PORT}")

    # Manejar señales de cierre (CTRL+C o SIGTERM)
    def close_server(sig, frame):
        print("\nRecibida señal de cierre. Cerrando socket...")
        server_socket.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, close_server)  
    signal.signal(signal.SIGTERM, close_server)

    # Cargar el modelo para la predicción
    mlp_model, modelPytorch = model_for_prediction(model_type, n_features, n_classes)

    while True:
        try:
            readable, _, _ = select.select([server_socket], [], [], 1.0)
            if server_socket in readable:
                conn, addr = server_socket.accept()
                print(f"Conectado con {addr}")

                while True:
                    data = conn.recv(1024)
                    if not data:
                        print("El cliente cerró la conexión.")
                        break

                    mensaje = data.decode('utf-8')
                    predicted_move = get_prediction(model_type, mensaje, mlp_model, modelPytorch)
                    

                    print(f"Datos recibidos: {mensaje}", end="")
                    print(f"Enviando respuesta: {predicted_move}")
                    print()

                    respuesta = f"{predicted_move}\n"
                    conn.sendall(respuesta.encode('utf-8'))

                conn.close()
                print("Esperando nueva conexión...")

        except ConnectionResetError:
            print("Error: El cliente cerró la conexión de manera abrupta.")