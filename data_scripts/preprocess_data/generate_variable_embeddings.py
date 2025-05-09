import os
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pickle
# Import our custom modules
from data_scripts.LLM_utils import run_LLM

class VariableEmbeddingGenerator:
    def __init__(self, data_path, temp_dfs_path, model_name="medicalai/ClinicalBERT"):
        """
        Initialize the variable embedding generator.
        
        Args:
            base_path: Path to MIMIC-IV data
            temp_dfs_path: Path to directory with existing processed files
            model_name: Name of the pretrained ClinicalBERT model
        """
        self.data_path = data_path
        self.temp_dfs_path = temp_dfs_path
        self.model_name = model_name
        
        # Initialize ClinicalBERT model and tokenizer
        print(f"Loading {model_name} model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        # Set device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dfs_path, exist_ok=True)
    
    def generate_descriptions(self, var_names, output_file="variable_descriptions.json"):
        """
        Generate descriptions for clinical variables using LLM.
        
        Args:
            var_names: List of variable names to generate descriptions for
            output_file: Path to save the variable descriptions
        
        Returns:
            Dictionary mapping variable names to descriptions
        """
        self.var_names = var_names
        
        # Check if descriptions file already exists
        output_path = os.path.join(self.temp_dfs_path, output_file)
        if os.path.exists(output_path):
            print(f"Loading existing descriptions from {output_path}")
            with open(output_path, 'r') as f:
                return json.load(f)
        
        print("Generating descriptions for variables using LLM...")
        
        # Set up system prompt
        system_prompt = """
        You are a medical expert assistant who can provide clear, accurate descriptions of clinical measurements, lab tests, and vital signs.
        """
        
        # Process variables in batches to avoid too many API calls
        batch_size = 10
        descriptions = {}
        
        for i in tqdm(range(0, len(self.var_names), batch_size)):
            batch = self.var_names[i:i+batch_size]
            
            # Create prompt for this batch
            prompt = f"""
            For each of the following clinical measurements, lab tests, or vital signs from the MIMIC-IV database, provide a concise description (1-2 sentences) explaining what it measures and its clinical significance.

            {', '.join(batch)}

            Format your response as a valid JSON object where keys are the exact measurement names and values are the descriptions:
            {{
              "measurement_name": "Description of what this measures and its significance.",
              ...
            }}
            """
            
            # Get response from LLM
            response = run_LLM(system_prompt, prompt, iterations=1, model="gpt-4o")
            
            # Parse the response
            try:
                # Extract JSON from the response
                response = response.strip()
                if response.startswith('```json'):
                    response = response[7:]
                if response.endswith('```'):
                    response = response[:-3]
                
                batch_descriptions = json.loads(response)
                descriptions.update(batch_descriptions)
                print(f"Processed batch {i//batch_size + 1}/{(len(self.var_names) + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"Error parsing LLM response for batch {i//batch_size + 1}: {e}")
                print(f"Raw response: {response}")
        
        # Save descriptions to file
        with open(output_path, 'w') as f:
            json.dump(descriptions, f, indent=2)
        
        print(f"Saved {len(descriptions)} variable descriptions to {output_path}")
        return descriptions
    
    def generate_embeddings(self,
            var_names,
            output_file="mimic4_bert_var_rep_gpt_source.pt",
            batch_size=32,
            pooling="cls",              # "cls" or "mean"
        ):
        """
        Generate ClinicalBERT embeddings for the provided variable names.

        Returns
        -------
        embeddings : torch.Tensor
            Shape (len(var_names), hidden_size)
        descriptions : dict
            Variable → description mapping (same as returned by self.generate_descriptions)
        """
        self.var_names = var_names                    # remember for later calls
        descriptions = self.generate_descriptions(var_names)

        output_path = os.path.join(self.temp_dfs_path, output_file)
        if os.path.exists(output_path):
            print(f"Loading existing embeddings from {output_path}")
            return torch.load(output_path), descriptions

        print(f"Generating embeddings in batches of {batch_size}...")

        # Pre-assemble description strings in input order
        text_list = [
            descriptions.get(v, f"Clinical measurement: {v}")
            for v in self.var_names
        ]

        all_embs = []
        self.model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(text_list), batch_size)):
                batch_text = text_list[start:start + batch_size]

                tok = self.tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                out = self.model(**tok, output_hidden_states=True)
                last_layer = out.hidden_states[-1]        # (B, L, H)

                if pooling == "cls":
                    emb = last_layer[:, 0, :]             # [CLS]
                elif pooling == "mean":
                    # mask-aware mean: ignore padding tokens
                    mask = tok["attention_mask"].unsqueeze(-1)  # (B, L, 1)
                    emb = (last_layer * mask).sum(1) / mask.sum(1)
                else:
                    raise ValueError("pooling must be 'cls' or 'mean'")

                all_embs.append(emb.cpu())

        embeddings_tensor = torch.cat(all_embs, dim=0)    # (N, H)
        torch.save(embeddings_tensor, output_path)
        print(f"Saved embeddings of shape {embeddings_tensor.shape} → {output_path}")

        return embeddings_tensor, descriptions