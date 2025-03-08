import os
import socket
import joblib
import torch
import signal
import sys
import select
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



def model_for_prediction(model_type, n_features, n_classes, intersection_id=None):
    """
    Cargar el modelo deseado basándose en la intersección.
    """

    if model_type == 'sklearn':
        model_filename = f'mlp_trained_model_2025-03-05_({intersection_id},).pkl'
        full_model_path = os.path.join(path_trained, 'models_2025-03-06', model_filename)
                
        mlp_model = joblib.load(full_model_path)
        return mlp_model, None
    
    elif model_type == 'pytorch':
       
        model_filename = f'pytorch_model_2025-03-05_({intersection_id},).pth'
        full_model_path = os.path.join(path_trained, 'models_2025-03-05', model_filename)
        
        modelPytorch = MyModelPyTorch(n_features, n_classes)
        modelPytorch.load_state_dict(torch.load(full_model_path, weights_only=True))
        modelPytorch.eval()
        return None, modelPytorch
    
    else:
        raise ValueError("Modelo no reconocido. Elige 'pytorch' o 'sklearn'.")


def get_prediction(model_type, mensaje, n_features, n_classes):
    
    preprocessed_state = preprocess_game_state(mensaje, dataset_path)
    
    intersection_id = identify_intersection(preprocessed_state) # envio el estado del juego como un diccionario
    # Cargar el modelo para la predicción
    mlp_model, modelPytorch = model_for_prediction(model_type, n_features, n_classes, intersection_id)
    # Convertir el estado preprocesado a tensor o array, segun el modelo
    
    if model_type == 'pytorch':
        input_tensor = torch.tensor(preprocessed_state.values, dtype=torch.float32).unsqueeze(0)  # Agregar dimensión para batch (modelo espera recibir un batch de un solo ejemplo)
        # Realizar la predicción
        with torch.no_grad():  # No calcular gradientes
            prediccion = modelPytorch(input_tensor)               
        # Seleccionar el índice con el valor máximo para obtener el movimiento predicho
        predicted_index = torch.argmax(prediccion)      
    elif model_type == 'sklearn':
        probabilidades = mlp_model.predict_proba(preprocessed_state)  # Obtiene las probabilidades de cada clase
        predicted_index = np.argmax(probabilidades)  # Índice de la clase con mayor probabilidad
        prediccion = np.max(probabilidades)  # Obtiene el valor máximo de probabilidad
    

    moves = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NEUTRAL']
    predicted_move = moves[predicted_index]
    
    return predicted_move

def identify_intersection(preprocessed_state):
    
    intersection = preprocessed_state.get('pacmanCurrentNodeIndex', None)
    
    return intersection.iloc[0]
    
    

    
    
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
                    predicted_move = get_prediction(model_type, mensaje, n_features, n_classes)
                    

                    print(f"Datos recibidos: {mensaje}", end="")
                    print(f"Enviando respuesta: {predicted_move}")
                    print()

                    respuesta = f"{predicted_move}\n"
                    conn.sendall(respuesta.encode('utf-8'))

                conn.close()
                print("Esperando nueva conexión...")

        except ConnectionResetError:
            print("Error: El cliente cerró la conexión de manera abrupta.")


    """
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
        
        # Obtener la predicción usando el modelo seleccionado
        predicted_move = get_prediction(model_type, mensaje, n_features, n_classes)
        
        respuesta = f"{predicted_move}\n" 
        print(respuesta)
        conn.sendall(respuesta.encode('utf-8'))  # Enviar respuesta codificada en UTF-8

    conn.close()
    """