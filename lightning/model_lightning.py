# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from einops import *
from einops import repeat
import logging
import os
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
import wandb
import torch
from models import *

# Configure logging for debugging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model.log')
        # Console output disabled
    ]
)

class KEDGNLightning(pl.LightningModule):
    def __init__(
        self, 
        hidden_dim, 
        num_of_variables, 
        num_of_timestamps, 
        d_static,
        n_class=2, 
        node_enc_layer=2, 
        rarity_alpha=0.5, 
        query_vector_dim=5, 
        node_emb_dim=8, 
        plm_rep_dim=768, 
        use_gat=False, 
        num_heads=2, 
        use_adj_mask=False,
        use_transformer=False, 
        history_len=10, 
        nhead_transformer=2,
        learning_rate=1e-3
    ):
        super(KEDGNLightning, self).__init__()
        
        # Save hyperparameters for easy access and logging
        self.save_hyperparameters()
        
        # Initialize model components
        self.num_of_variables = num_of_variables
        self.num_of_timestamps = num_of_timestamps
        self.hidden_dim = hidden_dim
        
        # Initialize a learnable adjacency matrix
        self.adj = nn.Parameter(torch.ones(size=[num_of_variables, num_of_variables]))
        
        # Encoders for raw values and absolute time information
        self.value_enc = Value_Encoder(output_dim=hidden_dim)
        self.abs_time_enc = Time_Encoder(embed_time=hidden_dim, var_num=num_of_variables)
        
        # GRU to process observation time patterns with multiple layers
        self.obs_tp_enc = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim,
            num_layers=node_enc_layer, 
            batch_first=True, 
            bidirectional=False
        )
        
        # Observation encoder
        self.obs_enc = nn.Sequential(
            nn.Linear(in_features=6 * hidden_dim, out_features=hidden_dim),
            nn.ReLU()
        )
        
        # Embedding for variable type information
        self.type_emb = nn.Embedding(num_of_variables, hidden_dim)
        
        # Choose the appropriate neural network architecture based on input flags
        if use_transformer:
            self.GCRNN = VSDTransformerGATRNN(
                d_in=self.hidden_dim,
                d_model=self.hidden_dim, 
                num_of_nodes=num_of_variables, 
                history_len=history_len,
                query_vector_dim=query_vector_dim, 
                node_emb_dim=node_emb_dim,
                plm_rep_dim=plm_rep_dim, 
                num_heads=num_heads, 
                use_adj_mask=use_adj_mask,
                nhead_transformer=nhead_transformer
            )
            logging.info("Using Transformer-GAT model with history_len: {}, transformer heads: {}, GAT heads: {}"
                         .format(history_len, nhead_transformer, num_heads))
        elif use_gat:
            self.GCRNN = VSDGATRNN(
                d_in=self.hidden_dim, 
                d_model=self.hidden_dim,
                num_of_nodes=self.num_of_variables, 
                rarity_alpha=rarity_alpha,
                query_vector_dim=query_vector_dim, 
                node_emb_dim=node_emb_dim,
                plm_rep_dim=plm_rep_dim, 
                num_heads=num_heads, 
                use_adj_mask=use_adj_mask
            )
            logging.info("Using GRU-GAT model with {} heads, use_adj_mask: {}".format(num_heads, use_adj_mask))
        else:
            self.GCRNN = VSDGCRNN(
                d_in=self.hidden_dim, 
                d_model=self.hidden_dim,
                num_of_nodes=self.num_of_variables, 
                rarity_alpha=rarity_alpha,
                query_vector_dim=query_vector_dim, 
                node_emb_dim=node_emb_dim,
                plm_rep_dim=plm_rep_dim
            )
            logging.info("Using original GRU-GCN model")
        
        # Final layers
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        
        # Process static features if available
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_variables)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_variables * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_variables, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)
            )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Save learning rate
        self.learning_rate = learning_rate

    def forward(self, P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor):
        """
        Forward pass for KEDGN.
        Inputs:
            P: Tensor of shape [B, T, 2*V] where first V columns are observed values and last V are masks.
            P_static: Tensor of static features [B, d_static] (or None).
            P_avg_interval: Tensor, average intervals [B, T, V].
            P_length: Tensor, lengths of the time series for each batch element [B, 1].
            P_time: Tensor of timestamps [B, T, V] (or [B, T]).
            P_var_plm_rep_tensor: Tensor of pre-trained language model embeddings for each variable [B, V, plm_rep_dim]
        """
        # Logging input shapes
        logging.debug("Input P shape: {}".format(P.shape))
        b, t, v = P.shape
        # Since P contains both observed data and masks, divide v by 2.
        v = v // 2

        # Split P into observed data and corresponding observation masks.
        observed_data = P[:, :, :v]        # Shape: [B, T, V]
        observed_mask = P[:, :, v:]          # Shape: [B, T, V]
        logging.debug("Observed data shape: {} | Observed mask shape: {}".format(observed_data.shape, observed_mask.shape))

        # Encode the observed values and times.
        # The encoders expect additional dimensions, so we multiply elementwise by mask to zero-out missing data.
        value_emb = self.value_enc(observed_data) * observed_mask.unsqueeze(-1)  # [B, T, V, hidden_dim]
        abs_time_emb = self.abs_time_enc(P_time) * observed_mask.unsqueeze(-1)   # [B, T, V, hidden_dim]
        logging.debug("Value embedding shape: {} | Time embedding shape: {}".format(value_emb.shape, abs_time_emb.shape))

        # Get type embedding for variables.
        # Repeat the learnable embedding weight vector from self.type_emb (of shape [V, hidden_dim])
        # so that we have one per batch sample.
        type_emb = repeat(self.type_emb.weight, 'v d -> b v d', b=b)  # Shape: [B, V, hidden_dim]
        logging.debug("Type embedding shape: {}".format(type_emb.shape))

        # Prepare the structured input encoding.
        # This combines the value, time, and type embeddings.
        # We need to match dimensions so we repeat 'type_emb' along the time dimension.
        structure_input_encoding = (value_emb + abs_time_emb + repeat(type_emb, 'b v d -> b t v d', t=t)) * observed_mask.unsqueeze(-1)
        logging.debug("Structured input encoding shape: {}".format(structure_input_encoding.shape))
        
        # Pass the structured encoding along with mask, lengths, average intervals, and PLM embeddings
        # into the dynamic graph convolutional recurrent network.
        last_hidden_state = self.GCRNN(structure_input_encoding, observed_mask, P_length, P_avg_interval, P_var_plm_rep_tensor)
        logging.debug("Last hidden state shape: {}".format(last_hidden_state.shape))
        
        # Sum the hidden state across the feature channels (if that is the desired aggregation).
        aggregated_hidden = torch.sum(last_hidden_state, dim=-1)  # Shape: [B, V]
        logging.debug("Aggregated hidden state shape: {}".format(aggregated_hidden.shape))
        
        # Optionally integrate static features.
        if P_static is not None:
            static_emb = self.emb(P_static)  # Map static features to an embedding of shape [B, V]
            logging.debug("Static embedding shape: {}".format(static_emb.shape))
            # Concatenate aggregated hidden states and static embeddings along last dimension.
            fused_features = torch.cat([aggregated_hidden, static_emb], dim=-1)  # [B, 2*V]
            logging.debug("Fused feature shape (hidden + static): {}".format(fused_features.shape))
            output = self.classifier(fused_features)
        else:
            output = self.classifier(aggregated_hidden)
        
        logging.info("Output shape: {}".format(output.shape))
        return output

    def training_step(self, batch, batch_idx):
        """
        Lightning training step
        Args:
            batch: The output of your DataLoader
            batch_idx: Integer displaying index of this batch
        Returns:
            Dictionary with loss and any desired metrics for logging
        """
        # Unpack batch
        P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor, y = batch
        
        # Forward pass
        logits = self.forward(P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
        
        # Calculate loss
        loss = self.criterion(logits, y.long().squeeze(1))
        
        # Calculate additional metrics
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Convert tensors to CPU numpy arrays for sklearn metrics
        y_np = y.squeeze(1).cpu().numpy()
        probs_np = probs[:, 1].detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        
        # Calculate metrics
        acc = (preds == y.squeeze(1)).float().mean()
        try:
            auroc = roc_auc_score(y_np, probs_np)
            auprc = average_precision_score(y_np, probs_np)
        except:
            # Handle edge case where batch contains only one class
            auroc = 0.0
            auprc = 0.0
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_auroc', auroc, on_step=False, on_epoch=True)
        self.log('train_auprc', auprc, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Lightning validation step
        """
        # Unpack batch
        P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor, y = batch
        
        # Forward pass
        logits = self.forward(P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
        
        # Calculate loss
        loss = self.criterion(logits, y.long().squeeze(1))
        
        # Calculate additional metrics
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Convert tensors to CPU numpy arrays for sklearn metrics
        y_np = y.squeeze(1).cpu().numpy()
        probs_np = probs[:, 1].detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        
        # Calculate metrics
        acc = (preds == y.squeeze(1)).float().mean()
        try:
            auroc = roc_auc_score(y_np, probs_np)
            auprc = average_precision_score(y_np, probs_np)
        except:
            # Handle edge case where batch contains only one class
            auroc = 0.0
            auprc = 0.0
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_auroc', auroc, on_epoch=True)
        self.log('val_auprc', auprc, on_epoch=True)
        
        return {'val_loss': loss, 'val_acc': acc, 'val_auroc': auroc, 'val_auprc': auprc}

    def test_step(self, batch, batch_idx):
        """
        Lightning test step
        """
        # Unpack batch
        P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor, y = batch
        
        # Forward pass
        logits = self.forward(P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
        
        # Calculate loss
        loss = self.criterion(logits, y.long().squeeze(1))
        
        # Calculate additional metrics
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Convert tensors to CPU numpy arrays for sklearn metrics
        y_np = y.squeeze(1).cpu().numpy()
        probs_np = probs[:, 1].detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        
        # Calculate metrics
        acc = (preds == y.squeeze(1)).float().mean()
        
        # Detailed metrics for test set
        if y_np.size > 1:  # Only compute if there are multiple samples
            auroc = roc_auc_score(y_np, probs_np)
            auprc = average_precision_score(y_np, probs_np)
            conf_mat = confusion_matrix(y_np, preds_np, labels=list(range(self.hparams.n_class)))
            class_report = classification_report(y_np, preds_np, labels=list(range(self.hparams.n_class)))
            
            # Log metrics
            self.log('test_loss', loss)
            self.log('test_acc', acc)
            self.log('test_auroc', auroc)
            self.log('test_auprc', auprc)
            
            # Log confusion matrix as a plot
            if wandb.run is not None:
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    import numpy as np
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title('Confusion Matrix')
                    wandb.log({"confusion_matrix": wandb.Image(plt)})
                    plt.close()
                except:
                    pass
            
            return {'test_loss': loss, 'test_acc': acc, 'test_auroc': auroc, 'test_auprc': auprc, 
                    'confusion_matrix': conf_mat, 'classification_report': class_report}
        else:
            return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        """
        Configure the optimizer for training
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer 