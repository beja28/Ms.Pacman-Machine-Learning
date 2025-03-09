import torch
import torch.nn as nn
from datetime import datetime
import os

class MyModelPyTorch(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MyModelPyTorch, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.Dropout(0.2), #apaga el 20% de las neuronas
            nn.Linear(128, 64), #segunda capa oculta
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            nn.Linear(64, n_classes)
        )

        
    def forward(self, x):
        return self.network(x)

# Función de red neuronal con PyTorch
def train_pytorch_nn(x_cv_tensor, Y_cv_tensor, train_loader, n_features, n_classes, num_epochs=300, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyModelPyTorch(n_features, n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01) # mas estable para datasets grandes

    best_val_loss = float('inf')
    patience_counter = 0
    print("Empiezo a entrenar")
    # Entrenamiento
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_corrects = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            hatY = model(x_batch)
            loss = loss_fn(hatY, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            running_corrects += (hatY.argmax(1) == y_batch).sum().item() # suma los aciertos para la precision de cada batch
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects / len(train_loader.dataset)

        model.eval() # Activa modo evaluación (desactiva Dropout y usa medias en BatchNorm)
        with torch.no_grad():
            x_cv_tensor, Y_cv_tensor = x_cv_tensor.to(device), Y_cv_tensor.to(device)
            y_pred = model(x_cv_tensor)
            val_loss = loss_fn(y_pred, Y_cv_tensor.long()).item() # calcula la perdida en validacion
            val_acc = (y_pred.argmax(1) == Y_cv_tensor).float().mean().item() # calcula la precision en validacion

        # detener si no mejora
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping en epoch {epoch}")
                break

        if epoch % (num_epochs // 10) == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

    print(f"\nFinal Train Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    
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