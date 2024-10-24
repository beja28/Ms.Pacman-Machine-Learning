import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score
from datetime import datetime
import joblib
import os

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
    pred_test = simple_model(x_cv_tensor)

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
def save_model_pth(model, path_trained):
    
    # Obtener la fecha actual
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Ruta para guardar el modelo
    final_path = os.path.join(path_trained, f'pytorch_model_{date_str}.pth')
    
    torch.save(model.state_dict(), final_path)
    print(f'Modelo guardado en {final_path}')
    
def save_model_mlp(mlp_model, path_trained):
    date_str = datetime.now().strftime('%Y-%m-%d')
    final_path = os.path.join(path_trained, f'mlp_trained_model_{date_str}.pkl')
    
    # Guardar el modelo entrenado en un archivo
    joblib.dump(mlp_model, final_path)
    print(f'Modelo MLP guardado en {final_path}')