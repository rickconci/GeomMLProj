# KEDGN: Knowledge-Enhanced Dynamic Graph Networks

This repository contains an implementation of KEDGN (Knowledge-Enhanced Dynamic Graph Networks) for multivariate time series classification. The model integrates pre-trained language model representations with graph neural networks to capture relationships between variables in irregularly-sampled time series.

## Features

- GRU-GCN (original model)
- GRU-GAT (Graph Attention variant)
- Transformer-GAT (Temporal transformer with graph attention)
- Time-aware positional encodings for handling irregular time intervals
- PyTorch Lightning integration for structured training
- Weights & Biases logging

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GeomMLProj.git
cd GeomMLProj

# Install dependencies
pip install -r requirements.txt
```

## Data

The code supports several medical time series datasets:

- PhysioNet (physionet)
- MIMIC-III (mimic3)
- P12
- P19

Place your datasets in the `data/` directory:

```
data/
  physionet/
  mimic3/
  P12/
  P19/
```

## Usage

### Original Training Script

```bash
python train.py --dataset physionet --epochs 10 --batch_size 256 --hidden_dim 4
```

### PyTorch Lightning Training Script

```bash
python train_lightning.py --dataset physionet --epochs 10 --batch_size 256 --hidden_dim 4 --use_wandb
```

### Model Variants

#### GRU-GCN (Original)

```bash
python train_lightning.py --dataset physionet
```

#### GRU-GAT

```bash
python train_lightning.py --dataset physionet --use_gat --num_heads 2
```

#### Transformer-GAT

```bash
python train_lightning.py --dataset physionet --use_transformer --history_len 15 --nhead_transformer 2 --use_gat --num_heads 2
```

### Multi-Task Models

The repository also includes implementations of multi-task models for predicting multiple healthcare outcomes simultaneously.

#### MultiTaskKEDGN

Extends the KEDGN model to predict:

1. Mortality prediction within 6 months of discharge
2. Readmission prediction within 15 days of discharge
3. PHE codes in the next admission

```bash
python train_multi_task.py --model_type full --hidden_dim 256 --lr 0.001
```

#### DS-Only MultiTask Model

A simpler model that uses only the discharge summary embeddings for predictions:

```bash
python train_multi_task.py --model_type ds_only --hidden_dim 256 --projection_dim 512
```

#### MultiTaskRaindropV2

An implementation that adapts the Raindrop_v2 architecture for multi-task prediction. Raindrop_v2 is a time series model that handles complex relationships between sensors through a specialized observation propagation mechanism and transformer architecture.

**Input data requirements:**

- Time series data with shape `[max_len, batch_size, 2*d_inp]` (first half contains sensor readings, second half contains missing value masks)
- Static/demographic features with shape `[batch_size, d_static]`
- Timestamps with shape `[max_len, batch_size]`
- Valid sequence lengths for each sample with shape `[batch_size]`
- Global structure (adjacency matrix) defining sensor relationships with shape `[d_inp, d_inp]`

**Key features:**

- Observation propagation mechanism that models relationships between variables
- Integration of transformer architecture for temporal patterns
- Attention mechanisms that account for missing data
- Multi-task prediction heads for mortality, readmission, and PHE codes

**Usage:**

```bash
python train_multi_task.py --model_type raindrop_v2 --hidden_dim 128 --d_model 64 --num_heads 4 --nlayers 2
```

**Additional RaindropV2 parameters:**

- `--d_model`: Number of expected model input features (default: 64)
- `--nlayers`: Number of transformer encoder layers (default: 2)
- `--global_structure`: Path to adjacency matrix file defining sensor relationships
- `--sensor_wise_mask`: Enable sensor-wise masking

## Command Line Arguments

- `--dataset`: Dataset name ('physionet', 'P12', 'P19', 'mimic3')
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--hidden_dim`: Hidden dimension size
- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: WandB project name
- `--wandb_entity`: WandB entity name
- `--rarity_alpha`: Hyperparameter for node rarity
- `--query_vector_dim`: Dimension of query vectors
- `--node_emb_dim`: Dimension of node embeddings
- `--use_gat`: Use Graph Attention Networks instead of GCN
- `--num_heads`: Number of attention heads for GAT
- `--use_adj_mask`: Use adjacency matrix as a mask for GAT attention
- `--use_transformer`: Use transformer per variable instead of GRU
- `--history_len`: History length for transformer model
- `--nhead_transformer`: Number of attention heads in transformer
- `--runs`: Number of runs with different seeds
- `--seed`: Random seed

## Model Architecture

The KEDGN model combines several components:

1. **Value and Time Encoders**: Process raw input values and timestamps
2. **Variable-specific Parameter Generation**: Dynamically generates parameters for each variable
3. **Graph Neural Network**: Either GCN or GAT for message passing between variables
4. **Temporal Modeling**: Either GRU cells or Transformer blocks for processing the time dimension

## Differences between GRU-GCN, GRU-GAT, and Transformer-GAT

1. **GRU-GCN (Original)**: Uses GRU cells for temporal processing and Graph Convolutional Networks for variable interactions

2. **GRU-GAT**: Replaces GCN with Graph Attention Networks to learn attention weights between variables

3. **Transformer-GAT**:
   - Replaces GRU with Transformer blocks for each variable
   - Includes time-aware positional encodings for handling irregular sampling
   - Maintains a history buffer for each variable
   - Uses self-attention to focus on the most relevant past observations

git clone <git@github.com>:rickconci/GeomMLProj.git
cd GeomMLProj
chmod +x setup_geomml.sh
./setup_geomml.sh

# Knowledge-Empowered Dynamic Graph Network for Irregularly Sampled Medical Time Series

## Overview

This repository contains implementation code for our NeurIPS 2024 paper: "*Knowledge-Empowered Dynamic Graph Network for Irregularly Sampled Medical Time Series*".

We propose Knowledge-Empowered Dynamic Graph Network (KEDGN), a graph neural network empowered by variables' textual medical information, aiming to model variable-specific temporal dependencies and inter-variable dependencies in ISMTS.
We evaluate KEDGN on four healthcare datasets.

## The model framework of KEDGN

![model](model.png)

## Experimental settings

 We conduct experiments on four widely used irregular medical time series datasets, namely P19 , Physionet, MIMIC-III  and P12  where Physionet is a reduced version of P12 considered by prior work. We compare our method with the state-of-the-art methods for modeling irregular time series,  including GRU-D , SeFT, mTAND, IP-Net , Raindrop and Warpformer. In addition, we also compared our method with two approaches initially designed for forecasting tasks, namely DGM^2  and MTGNN . The implementation and hyperparameter settings of these baselines were kept consistent with those used in Raindrop.

## Datasets

We prepared to run our code for KEDGN as well as the baseline methods with four healthcare
datasets.

### Raw data

**(1)** The PhysioNet Sepsis Early Prediction Challenge 2019  dataset consists of medical records from 38,803 patients. Each patient's record includes 34 variables. For every patient, there is a static vector indicating attributes such as age, gender, the time interval between hospital admission and ICU admission, type of ICU, and length of stay in the ICU measured in days. Additionally, each patient is assigned a binary label indicating whether sepsis occurs within the subsequent 6 hours. We follow the procedures of \cite{zhang2021graph} to ensure certain samples with excessively short or long time series are excluded. The raw data is available at <https://physionet.org/content/challenge-2019/1.0.0/>

**(2)** The P12  dataset comprises data from 11,988 patients after 12 inappropriate samples identified by \cite{horn2020set} were removed from the dataset. Each patient's record in the P12 dataset includes multivariate time series data collected during their initial 48-hour stay in the ICU. The time series data consists of measurements from 36 sensors (excluding weight). Additionally, each sample is associated with a static vector containing 9 elements, including age, gender, and other relevant attributes. Furthermore, each patient in the P12 dataset is assigned a binary label indicating the length of their stay in the ICU. A negative label signifies a hospitalization period of three days or shorter, while a positive label indicates a hospitalization period exceeding three days.  Raw data of **P12** can be found at <https://physionet.org/content/challenge-2012/1.0.0/>.

**(3)** MIMIC-III The MIMIC-III dataset is a widely used database that comprises de-identified Electronic Health Records of patients who were admitted to the ICU at Beth Israel Deaconess Medical Center from 2001 to 2012. Originally, it encompassed around 57,000 records of ICU patients, containing diverse variables such as medications, in-hospital mortality, and vital signs. Harutyunyan established a variety of benchmark tasks using a subset of this database. In our study, we focus on the binary in-hospital mortality prediction task to assess classification performance. Following preprocessing, our dataset consists of 16 features and 21,107 data points. It is available at <https://physionet.org/content/mimiciii/1.4/>

**(4)** Physionet contains the data from the first 48 hours of patients in ICU which is a reduced version of P12 considered by prior work. Therefore, we follow the same preprocessing methods as those used for the P12 dataset. The processed data set includes 3997 labeled instances. We focus on predicting in-hospital. It is available at <https://physionet.org/content/challenge-2012/>

### Processed data

For dataset P19 and P12. We use the data processed by [Raindrop](https://github.com/mims-harvard/Raindrop).

The raw data can be found at:

**(1)** P19: <https://physionet.org/content/challenge-2019/1.0.0/>

**(2)** P12: <https://physionet.org/content/challenge-2012/1.0.0/>

The datasets processed by [Raindrop](https://github.com/mims-harvard/Raindrop) can be obtained at:

**(1)** P19 (PhysioNet Sepsis Early Prediction Challenge 2019) <https://doi.org/10.6084/m9.figshare.19514338.v1>

**(2)** P12 (PhysioNet Mortality Prediction Challenge 2012) <https://doi.org/10.6084/m9.figshare.19514341.v1>

For the MIMIC-III dataset:

1. Obtain the raw data from <https://mimic.physionet.org/>.
2. Execute the mortality prediction data preprocessing program from <https://github.com/YerevaNN/mimic3-benchmarks> to obtain the .csv files.
3. Run the data preprocessing code from <https://github.com/ExpectationMax/medical_ts_datasets> to obtain the .npy files.

For the PhysioNet dataset:

1. Obtain the raw data from <https://physionet.org/content/challenge-2012/1.0.0/>. Use only the set-a portion.

2. Execute the preprocessing file in data/physionet/process_scripts/.

## Requirements

KEDGN has tested using Python 3.9.

To have consistent libraries and their versions, you can install needed dependencies
for this project running the following command:

```
pip install -r requirements.txt
```

## Running the code

After obtaining the dataset and corresponding variable representations, starting from root directory *KEDGN*, you can run models on four datasets as follows:

- Physionet

```
python train.py --dataset physionet --batch_size 256 --lr 0.001 --plm bert --plm_rep_dim 768 --query_vector_dim 5 --node_emb_dim 9 --rarity_alpha 1  --hidden_dim 8 -- source gpt 
```

- P19

```
python train.py --dataset P19 --batch_size 512 --lr 0.005 --plm bert --plm_rep_dim 768 --query_vector_dim 5 --node_emb_dim 16 --rarity_alpha 1 --hidden_dim 16 -- source gpt  
```

- P12

```
python train.py --dataset P12 --batch_size 512 --lr 0.001 --plm bert --plm_rep_dim 768 --query_vector_dim 5 --node_emb_dim 7 --rarity_alpha 3 --hidden_dim 12 -- source gpt  
```

- MIMIC-III

```
python train.py --dataset mimic3 --batch_size 256 --lr 0.005 --plm bert --plm_rep_dim 768 --query_vector_dim 7 --node_emb_dim 7 --rarity_alpha 2  --hidden_dim 12 -- source gpt 
```

Algorithms can be run with named arguments, which allow the use of different settings from the paper:

- *dataset*: Choose which dataset to use. Options: [P12, P19, physionet, mimic3].
- *batch-size*: Training batch size.
- *lr*: Training learning rate.
- *plm*: Choose the pre-trained model used for extracting variable semantic representations. Options: [bert, bart, led, gpt2, pegasus, t5].
- *plm_rep_dim*: Dimension of the output representation of the pre-trained model, corresponding to *d* in the paper. Except for Pegasus, which is 1024, others are all 768.
- *query_vector_dim*: Dimension of the query vector, corresponding to *q* in the paper.
- *node_emb_dim*: Dimension of the variable node embedding, corresponding to *n* in the paper.
- *rarity_alpha*: Proportion of the density score, corresponding to *α* in the paper.
- *hidden_dim*: Dimension of the node state/observation encoding, corresponding to *h / k* in the paper.
- *source*: Choose the textual source. Options: [gpt, name, wiki].

### Variable Semantic Representations Extraction

- Download the corresponding pre-trained language model (PLM) offline files from Hugging Face (<https://huggingface.co/>).

- Move the downloaded files to the respective directory under /data/plm/.

- Run the following command:

- cd ./data

  ```
   python get_var_rep.py --plm [plm_name]
  ```

  Available PLM options: [bert, bart, led, gpt2, pegasus, t5].
