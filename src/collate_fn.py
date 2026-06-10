import torch

def collate_fn(batch):
    """
    Dynamically zero-pads spectra of different lengths within a batch.
    Detects if the batch contains (spectrum, label) or (spectrum, label_topo, label_chem).
    
    Args:
        batch: list of tuples of length 2 or 3
    Returns:
        padded: torch.Tensor of shape (B, 1, max_len)
        labels (or labels_topo, labels_chem): torch.Tensor of labels
        mask: torch.Tensor of shape (B, max_len)
    """
    first_item = batch[0]
    
    if len(first_item) == 2:
        spectra, labels = zip(*batch)
        max_len = max(s.shape[-1] if hasattr(s, 'shape') else len(s) for s in spectra)
        padded = torch.zeros(len(spectra), 1, max_len)
        mask = torch.zeros(len(spectra), max_len, dtype=torch.bool)
        for i, s in enumerate(spectra):
            s_tensor = torch.as_tensor(s, dtype=torch.float32)
            L = s_tensor.shape[-1]
            if s_tensor.ndim == 1:
                padded[i, 0, :L] = s_tensor
            else:
                padded[i, 0, :L] = s_tensor[0]
            mask[i, :L] = True
        return padded, torch.tensor(labels, dtype=torch.long), mask
        
    elif len(first_item) == 3:
        spectra, labels_topo, labels_chem = zip(*batch)
        max_len = max(s.shape[-1] if hasattr(s, 'shape') else len(s) for s in spectra)
        padded = torch.zeros(len(spectra), 1, max_len)
        mask = torch.zeros(len(spectra), max_len, dtype=torch.bool)
        for i, s in enumerate(spectra):
            s_tensor = torch.as_tensor(s, dtype=torch.float32)
            L = s_tensor.shape[-1]
            if s_tensor.ndim == 1:
                padded[i, 0, :L] = s_tensor
            else:
                padded[i, 0, :L] = s_tensor[0]
            mask[i, :L] = True
        return (padded, 
                torch.tensor(labels_topo, dtype=torch.long), 
                torch.tensor(labels_chem, dtype=torch.long), 
                mask)
    else:
        raise ValueError(f"Unexpected batch element structure with length {len(first_item)}")
