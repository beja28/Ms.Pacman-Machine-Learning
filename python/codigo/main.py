from preprocessing import preprocess_csv
from models import train_pytorch_nn, save_model_pth, cross_validate_sklearn_mlp, save_model_mlp
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Ruta de datos y de la carpeta para guardar modelos
path = 'D:/Documentos/diego/universidad/4 Curso/TFG/Ms.Pacman-Machine-Learning/DataSets/01_gameStatesData.csv'
path_trained = 'D:/Documentos/diego/universidad/4 Curso/TFG/Ms.Pacman-Machine-Learning/python/codigo/Redes_Entrenadas'

# Preprocesamiento del CSV
X, Y = preprocess_csv(path)

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

# Entrenar el modelo con PyTorch
pytorch_model = train_pytorch_nn(X_train_tensor, Y_train_tensor, X_cv_tensor, Y_cv_tensor, n_features, n_classes, train_loader)
save_model_pth(pytorch_model, path_trained)

# Entrenar con MLP de Scikit-learn
mlp_model = cross_validate_sklearn_mlp(X, Y)
save_model_mlp(mlp_model, path_trained)
