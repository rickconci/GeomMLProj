import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import *
import os
#os.add_dll_directory('c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin')
#os.add_dll_directory(os.path.dirname(__file__))

from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import uniform, glorot, zeros, ones, reset

from models.transformer_conv import TransformerConv
from models.Ob_propagation import Observation_progation
import warnings
import numbers

device = get_device()

# Debug flag to control debug printing

class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        pe = pe.to(device)
        debug_print(f"PositionalEncodingTF output shape: {pe.shape}")
        return pe



class Raindrop_v2(nn.Module):
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
    """

    def __init__(self, d_inp=36, d_model=64, nhead=4, nhid=128, nlayers=2, dropout=0.3, max_len=215, d_static=9,
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2, global_structure=None, sensor_wise_mask=False, static=True):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.global_structure = global_structure
        self.sensor_wise_mask = sensor_wise_mask

        d_pe = 16
        d_enc = d_inp

        self.d_inp = d_inp
        self.d_model = d_model
        self.static = static
        if self.static:
            self.emb = nn.Linear(d_static, d_inp)

        self.d_ob = max(int(d_model/d_inp), 2)

        self.encoder = nn.Linear(d_inp*self.d_ob, self.d_inp*self.d_ob)

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        if self.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(self.d_inp*(self.d_ob+16), nhead, nhid, dropout)
        else:
            encoder_layers = TransformerEncoderLayer(d_model+16, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.adj = torch.ones([self.d_inp, self.d_inp]).to(device)

        self.R_u = Parameter(torch.Tensor(1, self.d_inp*self.d_ob)).to(device)

        self.ob_propagation = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                    n_nodes=d_inp, ob_dim=self.d_ob)

        self.ob_propagation_layer2 = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                           n_nodes=d_inp, ob_dim=self.d_ob)

        if static == False:
            d_final = d_model + d_pe
        else:
            d_final = d_model + d_pe + d_inp

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        self.aggreg = aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
        
        debug_print(f"Raindrop_v2 initialized with: d_inp={d_inp}, d_model={d_model}, d_ob={self.d_ob}, d_final={d_final}")
        debug_print(f"sensor_wise_mask={sensor_wise_mask}, static={static}")

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)
        glorot(self.R_u)

    def forward(self, src, static, times, lengths):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
        lengths = lengths: [128]: the number of nonzero recordings.
        """
        maxlen, batch_size = src.shape[0], src.shape[1]
        debug_print(f"[Raindrop_v2] Input src shape: {src.shape}, static shape: {None if static is None else static.shape}")
        debug_print(f"[Raindrop_v2] times shape: {times.shape}, lengths shape: {lengths.shape}")
        
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)]
        src = src[:, :, :int(src.shape[2]/2)]
        n_sensor = self.d_inp
        debug_print(f"[Raindrop_v2] After splitting, src shape: {src.shape}, missing_mask shape: {missing_mask.shape}")

        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        debug_print(f"[Raindrop_v2] After repeat_interleave, src shape: {src.shape}")
        
        h = F.relu(src*self.R_u)
        debug_print(f"[Raindrop_v2] After R_u multiplication, h shape: {h.shape}")
        
        pe = self.pos_encoder(times)
        debug_print(f"[Raindrop_v2] Positional encoding shape: {pe.shape}")
        
        if static is not None:
            emb = self.emb(static)
            debug_print(f"[Raindrop_v2] Static embedding shape: {emb.shape}")

        h = self.dropout(h)

        #Create a boolean mask indicating padding positions in the time dimension  - used to mask out padding tokens in the transformer encoder
        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).to(device)
        debug_print(f"[Raindrop_v2] Mask shape: {mask.shape}")

        step1 = True
        x = h
        if step1 == False:
            output = x
            distance = 0
        elif step1 == True:
            adj = self.global_structure.to(device)
            adj[torch.eye(self.d_inp).bool()] = 1

            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]
            debug_print(f"[Raindrop_v2] edge_index shape: {edge_index.shape}, edge_weights shape: {edge_weights.shape}")

            batch_size = src.shape[1]
            n_step = src.shape[0]
            output = torch.zeros([n_step, batch_size, self.d_inp*self.d_ob]).to(device)

            use_beta = False #False
            if use_beta == True:
                alpha_all = torch.zeros([int(edge_index.shape[1]/2), batch_size]).to(device)
            else:
                alpha_all = torch.zeros([edge_index.shape[1],  batch_size]).to(device)
            debug_print(f"[Raindrop_v2] alpha_all initial shape: {alpha_all.shape}")
            
            for unit in range(0, batch_size):
                stepdata = x[:, unit, :]
                p_t = pe[:, unit, :]
                debug_print(f"[Raindrop_v2] unit {unit} stepdata shape: {stepdata.shape}, p_t shape: {p_t.shape}")

                #reshape data from Time x Nodes x Features to Nodes x Time x Features
                stepdata = stepdata.reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2)
                #reshape data from Nodes x Time x Features to Nodes x Time*Features
                stepdata = stepdata.reshape(self.d_inp, n_step*self.d_ob)
                debug_print(f"[Raindrop_v2] After reshape, stepdata shape: {stepdata.shape}")

                stepdata, attentionweights = self.ob_propagation(stepdata, p_t=p_t, edge_index=edge_index, edge_weights=edge_weights,
                                 use_beta=use_beta,  edge_attr=None, return_attention_weights=True, residual=True)
                debug_print(f"[Raindrop_v2] After ob_propagation, stepdata shape: {stepdata.shape}")

                edge_index_layer2 = attentionweights[0]
                edge_weights_layer2 = attentionweights[1].squeeze(-1)
                debug_print(f"[Raindrop_v2] edge_index_layer2 shape: {edge_index_layer2.shape}, edge_weights_layer2 shape: {edge_weights_layer2.shape}")

                stepdata, attentionweights = self.ob_propagation_layer2(stepdata, p_t=p_t, edge_index=edge_index_layer2, edge_weights=edge_weights_layer2,
                                 use_beta=False,  edge_attr=None, return_attention_weights=True, residual=True)
                debug_print(f"[Raindrop_v2] After ob_propagation_layer2, stepdata shape: {stepdata.shape}")

                #reshape data from Nodes x Time x Features to Time x Nodes x Features
                stepdata = stepdata.view([self.d_inp, n_step, self.d_ob])
                #reshape data from Time x Nodes x Features to Nodes x Time x Features
                stepdata = stepdata.permute([1, 0, 2])
                #reshape data from Nodes x Time x Features to Nodes x Time*Features
                stepdata = stepdata.reshape([-1, self.d_inp*self.d_ob])
                debug_print(f"[Raindrop_v2] After final reshape, stepdata shape: {stepdata.shape}")

                output[:, unit, :] = stepdata
                alpha_all[:, unit] = attentionweights[1].squeeze(-1)

            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
            distance = torch.mean(distance)
            debug_print(f"[Raindrop_v2] Final output shape after loop: {output.shape}, distance: {distance}")

        if self.sensor_wise_mask == True:
            extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob)
            #give each sensor its own copy of the positional encoding
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.d_inp, 1])
            debug_print(f"[Raindrop_v2] extend_output shape: {extend_output.shape}, extended_pe shape: {extended_pe.shape}")
            
            output = torch.cat([extend_output, extended_pe], dim=-1) #Concatenates along the last dimension, so the output is now Time x Nodes x (Features + Positional Encoding)
            output = output.view(-1, batch_size, self.d_inp*(self.d_ob+16)) #Flattens the sensor dimension with feature+PE dimensions
            debug_print(f"[Raindrop_v2] After sensor_wise_mask, output shape: {output.shape}")
        else:
            output = torch.cat([output, pe], axis=2)
            debug_print(f"[Raindrop_v2] After concat with pe, output shape: {output.shape}")

        step2 = True
        if step2 == True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
            debug_print(f"[Raindrop_v2] After transformer_encoder, r_out shape: {r_out.shape}")
        elif step2 == False:
            r_out = output
        #The transformer encoder produces an output tensor of shape [maxlen, batch_size, d_inp(d_ob+16)], containing representations for each timestep

        #When using sensor_wise_mask=True, the output is reshaped to [maxlen, batch_size, d_inp, (d_ob+16)] to separate each sensor's features
        sensor_wise_mask = self.sensor_wise_mask

        #then for each individual sensor, we identify the non-missing timesteps and take the mean of the non-missing values. 
        masked_agg = True
        if masked_agg == True:
            lengths2 = lengths.unsqueeze(1)
            mask2 = mask.permute(1, 0).unsqueeze(2).long()
            debug_print(f"[Raindrop_v2] lengths2 shape: {lengths2.shape}, mask2 shape: {mask2.shape}")
            
            if sensor_wise_mask:
                output = torch.zeros([batch_size,self.d_inp, self.d_ob+16]).to(device)
                extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
                debug_print(f"[Raindrop_v2] extended_missing_mask shape: {extended_missing_mask.shape}")
                
                for se in range(self.d_inp): #for each sensor, 
                    r_out = r_out.view(-1, batch_size, self.d_inp, (self.d_ob+16)) #reshape transformer output to include sensor dim 
                    out = r_out[:, :, se, :] #select the output for the current sensor
                    len = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1) #count the missing values in the current sensor
                    out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (len + 1) #get the mean of the non-missing values
                    output[:, se, :] = out_sensor #store the sensor's output
                output = output.view([-1, self.d_inp*(self.d_ob+16)])
                debug_print(f"[Raindrop_v2] After sensor_wise_mask aggregation, output shape: {output.shape}")
            elif self.aggreg == 'mean':
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
                debug_print(f"[Raindrop_v2] After mean aggregation, output shape: {output.shape}")
        elif masked_agg == False:
            output = r_out[-1, :, :].squeeze(0)
        
        #final output shape [batch_size, d_inp(d_ob+16)] concatenates all these sensor-specific representations

        if static is not None:
            output = torch.cat([output, emb], dim=1)
            debug_print(f"[Raindrop_v2] After concat with static, final output shape: {output.shape}")
        #output = self.mlp_static(output)
        #then concatenate the static features with the output
        debug_print(f"[Raindrop_v2] Final output shape: {output.shape}, distance: {distance}")
        return output, distance, None
