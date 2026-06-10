import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plots and saves a normalized confusion matrix as a Seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums = np.where(row_sums == 0, 1e-12, row_sums)
    cm_norm = cm.astype('float') / row_sums
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, y_probs, class_names, save_path):
    """
    Plots and saves One-vs-Rest ROC curves for multi-class classification.
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_calibration_curve(y_true, y_probs, save_path):
    """
    Plots and saves a reliability calibration curve for multi-class confidence.
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_accuracies = []
    bin_confidences = []
    
    for i in range(10):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(accuracies[mask]))
            bin_confidences.append(np.mean(confidences[mask]))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(bin_centers[i])
            
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_confidences, bin_accuracies, 's-', color='darkblue', label='Model')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Fraction of Correct Predictions (Accuracy)')
    plt.title('Multi-class Reliability Calibration')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
