from preprocessing import preprocess_csv
from model_pytorch import train_pytorch_nn, save_model_pth
from model_sklearn import MLPModel
from prueba_socket import start_socket 
from explicabilidad import Explicabilidad
from Pytorch_Predictor import PyTorchPredictor
from model_pytorch import MyModelPyTorch
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import argparse

# Ruta de datos y de la carpeta para guardar modelos --> RUTA ABSOLUTA --> PROBLEMA
# path = 'D:/Documentos/diego/universidad/4 Curso/TFG/Ms.Pacman-Machine-Learning/DataSets/01_gameStatesData.csv'
# path_trained = 'D:/Documentos/diego/universidad/4 Curso/TFG/Ms.Pacman-Machine-Learning/python/codigo/Redes_Entrenadas'



# --- CREAR PATHS USANDO RUTAS RELATIVAS ---

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



# --- PREPROCESAMIENTO ---
grouped_df  = preprocess_csv(dataset_path)
# Configuramos los parámetros de la red
#acceder al primer grupo por ej
print(grouped_df)
n_features = grouped_df.first().shape[1]
print(n_features)
n_classes = 5  # 5 posibles movimientos de Pac-Man

def main():
    parser = argparse.ArgumentParser(description="Selecciona si quieres entrenar un modelo o utilizar uno para enviar la predicción o realizar la explicabilidad")

    # Definición de comandos
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # Comando para seleccionar el modelo para las predicciones
    parser_select_model = subparsers.add_parser("model", help="Selecciona el modelo a elegir para sacar la predicción (pytorch | sklearn)")
    parser_select_model.add_argument("model", choices=["pytorch", "sklearn"], help="Para usar el modelo Pytorch escriba -> model pytorch | Para usar el modelo MLP de scikit-learn escriba -> model sklearn")

    # Comando para entrenar el modelo
    parser_train_model = subparsers.add_parser("train_model", help="Elige el modelo a entrenar")
    parser_train_model.add_argument("train_model", choices=["pytorch", "sklearn"], help="Para entrenar el modelo Pytorch escriba -> train_model pytorch | Para entrenar el modelo MLP de scikit-learn escriba -> train_model sklearn")
    
    # Comando para realizar la explicabilidad
    parser_explain = subparsers.add_parser("explain", help="Explica el modelo seleccionado usando una técnica específica (SHAP, Feature Importance, LIME)")
    parser_explain.add_argument("model", choices=["pytorch", "sklearn"], help="Selecciona el modelo a explicar (pytorch o sklearn)")
    parser_explain.add_argument("technique", choices=["shap", "feature_importance", "lime"], help="Selecciona la técnica de explicabilidad (SHAP, Feature Importance, LIME)")

    args = parser.parse_args()


    if args.command == "model":
        start_socket(args.model, n_features, n_classes)
        
    elif args.command == "train_model":
        for key, group in grouped_df:
            print('Grupo de la interseccion : ', key)

            # Variables independientes (X) y dependientes (Y)
            X = group.drop(columns=['PacmanMove']) 
            Y = group['PacmanMove']
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
            if args.train_model == "pytorch":
                """ Entrenar el modelo con PyTorch """
                pytorch_model = train_pytorch_nn(X_cv_tensor, Y_cv_tensor, train_loader, n_features, n_classes)
                save_model_pth(pytorch_model, path_trained, key)

            elif args.train_model == "sklearn":
                """ Entrenar con MLP de Scikit-learn """
                mlp_model = MLPModel()
                mlp_model.train_and_cross_validate(X, Y)
                mlp_model.save_model_mlp(path_trained, key)   
                
    elif args.command == "explain":

        explicador = Explicabilidad()
        
        """ 
        --------------------------------------------------------------------------------------------------
        Importante seleccionar la carpeta de los modelos sklearn o pytorch para que funcione correctamente
        --------------------------------------------------------------------------------------------------
        """
        model_directory = os.path.join(path_trained, "models_2024-12-14")
        
        # Obtener los archivos de modelo del directorio
        model_files = [f for f in os.listdir(model_directory) if f.endswith(('.pkl', '.pth'))]
         
        if len(model_files) != len(grouped_df):
            print("Error: El número de modelos no coincide con el número de grupos.")
            return

        # Iterar sobre los archivos de modelo y los grupos del DataFrame al mismo tiempo
        for (key, group), model_filename in zip(grouped_df, model_files):
            print(f"Explicabilidad para el grupo: {key}, usando el modelo: {model_filename}")

            # Variables independientes (X) y dependientes (Y)
            X = group.drop(columns=['PacmanMove'])
            Y = group['PacmanMove']

            # Dividimos el conjunto de datos
            X_train, X_, y_train, y_ = train_test_split(X, Y, test_size=0.50, random_state=1)
            X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.20, random_state=1)

            # Cargar el modelo correspondiente
            full_model_path = os.path.join(model_directory, model_filename)
            
            if args.model == "pytorch":
                # Crea una instancia del modelo
                model = MyModelPyTorch(n_features, n_classes)
                # Carga los pesos
                model.load_state_dict(torch.load(full_model_path, weights_only=True))
                model.eval()
                predictor = PyTorchPredictor(model)
            elif args.model == "sklearn":
                model = MLPModel.load_model_mlp(full_model_path)
            
            if args.technique == "feature_importance":
                explicador.ejecutar_explicabilidad(model, model_filename, args.technique, X_cv, key, y_cv)
            elif args.technique == "lime":
                if args.model == "pytorch":
                    predictor = PyTorchPredictor(model)
                else:
                    predictor = model  # Los modelos de Scikit-Learn ya tienen predict_proba
                explicador.ejecutar_explicabilidad(predictor, model_filename, args.technique, X_cv, key)
            else:
                explicador.ejecutar_explicabilidad(model, model_filename, args.technique, X_cv, key)
        
        explicador.generar_grafico_explicabilidad_global(args.model)
        explicador.guardar_explicabilidad_txt(directorio_actual, args.model)


if __name__ == "__main__":
    main()