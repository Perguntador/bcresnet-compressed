import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

__all__ = [
    "set_seed",
    "initialize_weights",
    "generate_test_report"
]


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Ensure determinism in CuDNN (may verify slightly slower training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def initialize_weights(m):
    """
    Applies Kaiming (He) initialization to Conv/Linear layers
    and constant initialization to BatchNorm layers.
    """
    if isinstance(m, nn.Conv2d):
        # Kaiming Normal is ideal for networks with ReLU/SiLU activations.
        # mode='fan_out' preserves variance in the backward pass (beneficial for ResNets).
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.BatchNorm2d):
        # BatchNorm should start with weight 1 (no scaling) and bias 0.
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


def generate_test_report(model, test_loader, device):
    """
    Evaluates the model on the test set and prints a classification report.
    
    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader containing test data.
                                  Must expose .dataset attribute with .idx_to_label dict.
        device (str): Device to run evaluation on ('cpu' or 'cuda').

    Returns:
        tuple: (y_true, y_pred) - Lists containing true labels and predicted labels.
    """
    
    # Access the underlying dataset to retrieve class names
    # Note: Ensure your DataLoader wrapper (PreprocessingDataLoader) exposes .dataset
    test_ds = test_loader.dataset

    y_pred = []
    y_true = []
    
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            _, preds = torch.max(output, 1)
            
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(target.cpu().numpy())
    
    # Ensure target names are sorted by their index (0, 1, 2...)
    # This matches the output order of the model's logits
    sorted_labels = [test_ds.idx_to_label[i] for i in sorted(test_ds.idx_to_label.keys())]
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=sorted_labels,
        digits=4  # Increased precision for better analysis
    )
    
    print("Test Set Classification Report:")
    print(report)

    return y_true, y_pred