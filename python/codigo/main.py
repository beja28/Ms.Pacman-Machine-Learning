from preprocessing import preprocess_csv
from model_pytorch import train_pytorch_nn, save_model_pth
from model_sklearn import MLPModel
from prueba_socket import start_socket 
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import argparse

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



# --- PREPROCESAMIENTO ---

# Preprocesamiento del CSV
X, Y = preprocess_csv(dataset_path)

# Configuramos los parámetros de la red
n_features = X.shape[1]
n_classes = 5  # 5 posibles movimientos de Pac-Man


# Dividimos el conjunto de datos
X_train, X_, y_train, y_ = train_test_split(X, Y, test_size=0.50, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.20, random_state=1)

# Convertir a tensores
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_cv_tensor = torch.tensor(X_cv.values, dtype=torch.float32)
Y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
Y_cv_tensor = torch.tensor(y_cv.values, dtype=torch.float32)

# Crear DataLoader
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)


def main():
    parser = argparse.ArgumentParser(description="Selecciona si quieres entrenar un modelo o utilizar uno para enviar la predicción")
    
    # Definición de comandos
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles") # Lo que se seleccione se guarda en command

    # Comando para seleccionar el modelo para las predicciones
    parser_select_model = subparsers.add_parser("model", help="Selecciona el modelo a elegir para sacar la predicción (pytorch | sklearn)")
    parser_select_model.add_argument("model", choices=["pytorch", "sklearn"], help="Para usar el modelo Pytorch escriba -> model pytorch | Para usar el modelo MLP de scikit-learn escriba -> model sklearn")

    # Comando para entrenar el modelo
    parser_train_model = subparsers.add_parser("train_model", help="Elige el modelo a entrenar")
    parser_train_model.add_argument("train_model", choices=["pytorch", "sklearn"], help="Para entrenar el modelo Pytorch escriba -> train_model pytorch | Para entrenar el modelo MLP de scikit-learn escriba -> train_model sklearn")
    
    args = parser.parse_args()

    
    if args.command == "model":
        start_socket(args.model, n_features, n_classes)
        
    elif args.command == "train_model":
        if args.train_model == "pytorch":
            """ Entrenar el modelo con PyTorch """
            pytorch_model = train_pytorch_nn(X_cv_tensor, Y_cv_tensor, train_loader, n_features, n_classes)
            save_model_pth(pytorch_model, path_trained)
            
        elif args.train_model == "sklearn":
            """ Entrenar con MLP de Scikit-learn """
            mlp_model = MLPModel()
            mlp_model.train_and_cross_validate(X, Y)
            mlp_model.save_model(path_trained)


if __name__ == "__main__":
    main()
    