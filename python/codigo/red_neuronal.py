import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, make_scorer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

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


# Función de red neuronal con PyTorch
def train_pytorch_nn(X_train_tensor, Y_train_tensor, x_cv_tensor, Y_cv_tensor, n_features, n_classes, train_loader):
    simple_model = nn.Sequential(
        nn.Linear(n_features, 100),
        nn.ReLU(),
        nn.Linear(100, n_classes)
    )
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    
    num_epochs = 500
    log_epochs = num_epochs / 10
    loss_hist = [0] * num_epochs
    accuracy_hist = [0] * num_epochs

    print("Empiezo a entrenar")
    # Entrenamiento
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            hatY = simple_model(x_batch)
            loss = loss_fn(hatY, y_batch.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            loss_hist[epoch] += loss.item() * x_batch.size(0)
            is_correct = (torch.argmax(hatY, dim=1) == y_batch).float()
            accuracy_hist[epoch] += is_correct.sum().item()
        
        loss_hist[epoch] /= len(train_loader.dataset)
        accuracy_hist[epoch] /= len(train_loader.dataset)
        if epoch % log_epochs == 0:
            print(f"Epoch {epoch} Loss {loss_hist[epoch]:.4f} Accuracy {accuracy_hist[epoch]:.4f}")
        
        """Se evaluan los datos"""
    print("\nPrecision y error con los Datos de Entrenamiento")
    print(f'Precision: {accuracy_hist[epoch]:.2f}% --> Error(Misclassified points): {100 - accuracy_hist[epoch]*100:.2f}%')

    print("\nSe evalua el modelo complejo con los datos de Cross-Validation")
    pred_test = simple_model(X_cv_tensor)

    correct = (torch.argmax(pred_test, dim=1) == Y_cv_tensor).float()
    accuracy = correct.mean()
    print(f'Precision: {accuracy:.2f}% --> Error(Misclassified points): {100 - accuracy*100:.2f}%')
    
    return simple_model

# Función de red neuronal MLP con Scikit-learn usando cross_validate
def cross_validate_sklearn_mlp(X, Y, cv=5):
    mlp = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', batch_size=100,
             learning_rate_init=0.001, max_iter=500, random_state=333)
    
    # Definimos el scoring (evaluación de precisión)
    scoring = {'accuracy': make_scorer(accuracy_score)}
    
    # Validación cruzada con cross_validate
    print("Empiezo a entrenar")
    cv_results = cross_validate(mlp, X, Y, cv=cv, scoring=scoring, return_train_score=True)
    
    # Mostramos los resultados de la validación cruzada
    print(f"Train Accuracy (mean): {cv_results['train_accuracy'].mean():.4f}")
    print(f"Test Accuracy (mean): {cv_results['test_accuracy'].mean():.4f}")
    
    return cv_results

# Guardar el modelo entrenado en un archivo
def save_model(model, path_trained):
    torch.save(model.state_dict(), path_trained)
    print(f'Modelo guardado en {path_trained}')


# Llamada a las funciones
path = 'D:/Documentos/diego/universidad/4 Curso/TFG/Ms.Pacman-Machine-Learning/python/01_gameStatesData.csv'
path_trained = 'D:/Documentos/diego/universidad/4 Curso/TFG/Ms.Pacman-Machine-Learning/python/pytorch_model.pth'
X, Y = preprocess_csv(path)

# Configuramos los parámetros de la red
n_features = X.shape[1]
n_classes = 5  # 5 posibles movimientos de Pac-Man

#Se divide el conjunto de datos en entrenamiento 50% y temporal (para CrossValidation y Test)
X_train, X_, y_train, y_ = train_test_split(X, Y, test_size=0.50, random_state=1)

#Se divide del conjunto temporal en validación cruzada 80% y prueba 20%
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.20, random_state=1)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
X_cv_tensor = torch.tensor(X_cv.values, dtype=torch.float32)
Y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
Y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
Y_cv_tensor = torch.tensor(y_cv.values, dtype=torch.float32)

#Se crean el dataSet
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)

#Se crean el dataLoader
torch.manual_seed(1)
batch_size = 100  #Numero de ejemplos de datos que pasan en una iteracion por la red
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


pytorch_model = train_pytorch_nn(X_train_tensor, Y_train_tensor, X_cv_tensor, Y_cv_tensor, n_features, n_classes,train_loader)

# Entrenamos la red con MLP de Scikit-learn usando cross_validate
#cv_results = cross_validate_sklearn_mlp(X, Y)

# Guardar el modelo entrenado
save_model(pytorch_model, path_trained)
