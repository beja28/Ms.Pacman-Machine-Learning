import torch

class PyTorchPredictor:
    def __init__(self, model):
        self.model = model
        

    def predict_proba(self, X):
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor)  # Salida del modelo
            probabilities = torch.softmax(logits, dim=1).numpy()
        return probabilities
    
    def predict(self, X, class_names=None):
        if class_names is None:
            class_names = ["NEUTRAL", "LEFT", "RIGHT", "UP", "DOWN"]
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor)
            pred_indices = torch.argmax(logits, dim=1).numpy()
            return [class_names[i] for i in pred_indices]