import torch
import torch.nn.functional as F
from train_utils import get_device

device = get_device()

# Define functions for contrastive learning
def get_similarity_matrix(ts_embeddings, text_embeddings, similarity_metric='cosine', temperature=0.1):
    """
    Calculate similarity matrix between time series and text embeddings
    """
    # Normalize embeddings (for cosine similarity)
    if similarity_metric == 'cosine':
        ts_embeddings = F.normalize(ts_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        # Calculate cosine similarity
        similarities = torch.matmul(ts_embeddings, text_embeddings.T)
    elif similarity_metric == 'l2':
        # Negative L2 distance (closer = higher similarity)
        similarities = -torch.cdist(ts_embeddings, text_embeddings, p=2)
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
    return similarities / temperature

def clip_contrastive_loss(ts_embeddings, text_embeddings, temperature=0.1):
    """
    Calculate CLIP-style contrastive loss for natural TS+DS pairs.
    """
    # Normalize embeddings for cosine similarity
    ts_embeddings = F.normalize(ts_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)
    
    # Calculate similarity matrix with temperature scaling
    # Each row contains similarities between one TS and all DS
    similarity_matrix = torch.matmul(ts_embeddings, text_embeddings.T) / temperature
    
    # Labels are the diagonal indices (positive pairs)
    # For each TS (row), the correct DS is at the same index (column)
    labels = torch.arange(similarity_matrix.size(0), device=device)
    
    # Calculate loss in both directions:
    # 1. Time Series → Discharge Summary matching (i2t)
    i2t_loss = F.cross_entropy(similarity_matrix, labels)
    
    # 2. Discharge Summary → Time Series matching (t2i)
    t2i_loss = F.cross_entropy(similarity_matrix.T, labels)
    
    # Total loss is the average of both directions
    total_loss = (i2t_loss + t2i_loss) / 2
    
    return total_loss

def infonce_loss(ts_embeddings, text_embeddings, temperature=0.1):
    """
    Calculate InfoNCE contrastive loss for natural TS+DS pairs.
    """
    # Normalize embeddings for cosine similarity
    ts_embeddings = F.normalize(ts_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)
    
    # Calculate similarity matrix with temperature scaling
    logits = torch.matmul(ts_embeddings, text_embeddings.T) / temperature
    
    # For InfoNCE, each positive pair is matched against all negative pairs
    # The positive pair is on the diagonal of the similarity matrix
    batch_size = ts_embeddings.shape[0]
    labels = torch.arange(batch_size, device=device)
    
    # InfoNCE loss in TS→DS direction (each row of logits)
    ts_to_text_loss = F.cross_entropy(logits, labels)
    
    # InfoNCE loss in DS→TS direction (each column of logits, or each row of logits.T)
    text_to_ts_loss = F.cross_entropy(logits.T, labels)
    
    # Bidirectional InfoNCE loss
    total_loss = (ts_to_text_loss + text_to_ts_loss) / 2
    
    return total_loss

def contrastive_loss(ts_embeddings, text_embeddings, method='clip', temperature=0.1):
    """
    Calculate contrastive loss using the specified method (CLIP or InfoNCE).
    """
    if method.lower() == 'clip':
        return clip_contrastive_loss(ts_embeddings, text_embeddings, temperature)
    elif method.lower() == 'infonce':
        return infonce_loss(ts_embeddings, text_embeddings, temperature)
    else:
        raise ValueError(f"Unknown contrastive method: {method}")

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def detailed_count_parameters(model):
    """Count trainable parameters for each module in the model"""
    result = {}
    total = 0
    
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if params > 0 and name != "":  # Skip the parent module itself
            result[name] = params
            total += params
    
    # Add total count (might be slightly different due to parameter sharing)
    result['total'] = count_parameters(model)
    
    return result



