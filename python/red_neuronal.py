import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, make_scorer

# Función de preprocesamiento del CSV
def preprocess_csv(path):
    df = pd.read_csv(path)
    
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
def train_pytorch_nn(X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, n_features, n_classes, n_hidden=34, learning_rate=0.01, num_epochs=1000):
    simple_model = nn.Sequential(
        nn.Linear(n_features, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_classes)
    )
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=learning_rate)

    # Entrenamiento
    for epoch in range(num_epochs):
        hatY = simple_model(X_train_tensor)
        loss = loss_fn(hatY, Y_train_tensor.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluación
    with torch.no_grad():
        Y_test_pred = simple_model(X_test_tensor)
        test_loss = loss_fn(Y_test_pred, Y_test_tensor.long())
        _, predicted_classes = torch.max(Y_test_pred, 1)
        correct_predictions = (predicted_classes == Y_test_tensor.long()).sum().item()
        accuracy = correct_predictions / len(Y_test_tensor)
        
    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
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

# Llamada a las funciones
path = 'D:/Documentos/diego/universidad/4 Curso/TFG/Ms.Pacman-Machine-Learning/python/01_gameStatesData.csv'
X, Y = preprocess_csv(path)

# Configuramos los parámetros de la red
n_features = X.shape[1]
n_classes = 5  # 5 posibles movimientos de Pac-Man

# Entrenamos la red con PyTorch
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
# Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.long)
# Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.long)

#pytorch_model = train_pytorch_nn(X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, n_features, n_classes)

# Entrenamos la red con MLP de Scikit-learn usando cross_validate
cv_results = cross_validate_sklearn_mlp(X, Y)
