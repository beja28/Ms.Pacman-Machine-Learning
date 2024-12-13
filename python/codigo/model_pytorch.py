import torch
import torch.nn as nn
from datetime import datetime
import os

class MyModelPyTorch(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MyModelPyTorch, self).__init__()
        self.network = nn.Sequential(
        nn.Linear(n_features, 100),
        nn.ReLU(),
        nn.Linear(100, n_classes)
    )

        
    def forward(self, x):
        return self.network(x)

# Función de red neuronal con PyTorch
def train_pytorch_nn(x_cv_tensor, Y_cv_tensor, train_loader, n_features, n_classes):
    
    model = MyModelPyTorch(n_features, n_classes)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 500
    log_epochs = num_epochs / 10
    loss_hist = [0] * num_epochs
    accuracy_hist = [0] * num_epochs

    print("Empiezo a entrenar")
    # Entrenamiento
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            hatY = model(x_batch)
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
    pred_test = model(x_cv_tensor)

    correct = (torch.argmax(pred_test, dim=1) == Y_cv_tensor).float()
    accuracy = correct.mean()
    print(f'Precision: {accuracy:.2f}% --> Error(Misclassified points): {100 - accuracy*100:.2f}%')
    
    return model

def save_model_pth(model, path_trained, key):
    """
    Guarda el modelo entrenado en un archivo .pth con la fecha actual en el nombre.
    Crea una carpeta para cada fecha y guarda los modelos con un identificador único (key).
    """
    # Crear una carpeta con la fecha actual
    date_str = datetime.now().strftime('%Y-%m-%d')
    folder_path = os.path.join(path_trained, f'models_{date_str}')
    os.makedirs(folder_path, exist_ok=True)
    
    # Crear el nombre del archivo incluyendo el identificador 'key'
    model_filename = f'pytorch_model_{date_str}_{key}.pth'
    final_path = os.path.join(folder_path, model_filename)
    
    # Guardar el modelo
    torch.save(model.state_dict(), final_path)
    print(f'Modelo guardado en {final_path}')
    