import torch

class PyTorchPredictor:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        # Asegúrate de que el modelo devuelva probabilidades en el rango [0, 1]
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor)  # Salida del modelo
            probabilities = torch.softmax(logits, dim=1).numpy()
        return probabilities