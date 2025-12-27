import torch
import numpy as np
from tqdm import tqdm

__all__ = [
    'EarlyStopping',
    'BCResNetScheduler',
    'train_model'
]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): Function to print trace messages.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class BCResNetScheduler:
    def __init__(self, optimizer, n_batches, total_epochs=200, warmup_epochs=5, init_lr=1e-1, lr_lower_limit=0):
        """
        Custom scheduler designed to reproduce the exact learning rate schedule 
        of the official BC-ResNet implementation (Linear Warmup + Cosine Annealing).
        
        Args:
            optimizer (torch.optim.Optimizer): The optimizer instance.
            n_batches (int): Number of batches per epoch (len(train_loader)).
            total_epochs (int): Total number of training epochs.
            warmup_epochs (int): Number of epochs for the linear warmup phase.
            init_lr (float): Maximum initial learning rate after warmup.
            lr_lower_limit (float): Minimum learning rate (lower bound).
        """
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.lr_lower_limit = lr_lower_limit
        
        # Step definitions based on the original code logic
        self.n_step_warmup = n_batches * warmup_epochs
        self.total_iter = n_batches * total_epochs
        self.iterations = 0

    def step_batch(self):
        """
        Updates the learning rate. Should be called at every batch iteration.
        """
        self.iterations += 1
        
        # Exact logic extracted from the original main.py
        if self.iterations < self.n_step_warmup:
            # Linear Warmup Phase
            lr = self.init_lr * self.iterations / self.n_step_warmup
        else:
            # Cosine Annealing Phase
            lr = self.lr_lower_limit + 0.5 * (self.init_lr - self.lr_lower_limit) * (
                1 + np.cos(np.pi * (self.iterations - self.n_step_warmup) / (self.total_iter - self.n_step_warmup))
            )
        
        # Apply the calculated LR to the optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step_epoch(self):
        """
        Placeholder for epoch-level stepping. 
        Since this scheduler operates on a per-iteration (batch) basis, 
        this method does nothing but ensures compatibility with generic training loops.
        """
        pass


def train_model(model, early_stopping, n_epochs, optimizer, train_loader, valid_loader, criterion, scheduler=None, device='cpu'):
    """
    Main training loop.

    Args:
        model (nn.Module): The PyTorch model to train.
        early_stopping (EarlyStopping): Instance of the EarlyStopping utility.
        n_epochs (int): Maximum number of epochs.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        train_loader (DataLoader): Training data loader.
        valid_loader (DataLoader): Validation data loader.
        criterion (callable): Loss function (e.g., CrossEntropyLoss).
        scheduler (object, optional): Learning rate scheduler.
        device (str): Device to run training on ('cpu' or 'cuda').

    Returns:
        tuple: (model, train_losses, valid_losses, train_acc_history, valid_acc_history)
    """
    
    # Metric tracking
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    train_acc_history = []
    valid_acc_history = []
    
    for epoch in range(1, n_epochs + 1):

        # --- TRAIN PHASE ---
        model.train()
        running_corrects = 0
        total_samples = 0
        
        # Epoch progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
        
        for data, target in pbar:
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            # Batch-level Scheduler Step (if applicable, e.g., BCResNetScheduler)
            if scheduler and hasattr(scheduler, 'step_batch'):
                 scheduler.step_batch()
            
            train_losses.append(loss.item())
            
            # Calculate Batch Accuracy
            _, preds = torch.max(output, 1)
            running_corrects += torch.sum(preds == target.data)
            total_samples += target.size(0)

        # Epoch-level Scheduler Step (or Standard PyTorch schedulers fallback)
        if scheduler and hasattr(scheduler, 'step_epoch'):
            scheduler.step_epoch()
        elif scheduler: 
            try:
                scheduler.step()
            except:
                pass

        # Calculate Epoch Metrics (Train)
        epoch_train_acc = running_corrects.double() / total_samples
        train_loss = np.average(train_losses)
        
        avg_train_losses.append(train_loss)
        train_acc_history.append(epoch_train_acc.item()) 

        # --- VALIDATION PHASE ---
        model.eval()
        val_running_corrects = 0
        val_total_samples = 0
        
        with torch.no_grad(): 
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                valid_losses.append(loss.item())
                
                _, preds = torch.max(output, 1)
                val_running_corrects += torch.sum(preds == target.data)
                val_total_samples += target.size(0)

        # Calculate Epoch Metrics (Validation)
        valid_loss = np.average(valid_losses)
        epoch_valid_acc = val_running_corrects.double() / val_total_samples
        
        avg_valid_losses.append(valid_loss)
        valid_acc_history.append(epoch_valid_acc.item())

        epoch_len = len(str(n_epochs))
        
        # Logging
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'train_acc: {epoch_train_acc:.4f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'valid_acc: {epoch_valid_acc:.4f}')
        
        print(print_msg)
        
        # Reset batch lists for next epoch
        train_losses = []
        valid_losses = []
        
        # Early Stopping Check
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load best model weights found during training
    model.load_state_dict(torch.load(early_stopping.path, map_location=device))

    return model, avg_train_losses, avg_valid_losses, train_acc_history, valid_acc_history
