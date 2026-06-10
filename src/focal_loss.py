import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        """
        Focal Loss for addressing class imbalance.
        
        Args:
            alpha: tensor (num_classes,) - Class weight factor (typically 1/f_t normalized).
            gamma: float - Focusing parameter. gamma=0 reduces to weighted CrossEntropy.
        """
        super().__init__()
        # Register buffer to automatically move it to the device of input/labels
        self.register_buffer('alpha', alpha)
        self.gamma = gamma

    def forward(self, logits, targets):
        # ce_loss matches log(pt)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Gather alpha_t for each target
        alpha_t = self.alpha[targets]
        
        # Focal loss formula: L = -alpha_t * (1 - pt)^gamma * log(pt)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def compute_alpha(y_train, num_classes=8):
    """
    Computes class weighting alpha based on training fold label frequencies.
    
    Args:
        y_train: array-like - Class labels in the training set.
        num_classes: int - Total number of classes.
    Returns:
        alpha: torch.Tensor - Normalized class weights of shape (num_classes,).
    """
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1e-6, counts)  # avoid division by zero
    alpha = 1.0 / counts
    alpha = alpha / alpha.sum()
    return torch.tensor(alpha, dtype=torch.float32)
