import os
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Import our custom modules
from data_scripts.LLM_utils import run_LLM

class VariableEmbeddingGenerator:
    def __init__(self, base_path, temp_dfs_path, model_name="medicalai/ClinicalBERT"):
        """
        Initialize the variable embedding generator.
        
        Args:
            base_path: Path to MIMIC-IV data
            temp_dfs_path: Path to directory with existing processed files
            model_name: Name of the pretrained ClinicalBERT model
        """
        self.base_path = base_path
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
    
    def generate_embeddings(self, descriptions, output_file="mimic4_bert_var_rep_gpt_source.pt"):
        """
        Generate embeddings for clinical variables using descriptions and ClinicalBERT.
        
        Args:
            descriptions: Dictionary mapping variable names to descriptions
            output_file: Path to save the variable embeddings
        
        Returns:
            Tensor of variable embeddings
        """
        # Check if embeddings file already exists
        output_path = os.path.join(self.temp_dfs_path, output_file)
        if os.path.exists(output_path):
            print(f"Loading existing embeddings from {output_path}")
            return torch.load(output_path)
        
        print("Generating embeddings for variables using ClinicalBERT...")
        
        # Create embeddings for all variables
        embeddings = []
        for var_name in tqdm(self.var_names):
            # Get description for this variable
            description = descriptions.get(var_name, f"Clinical measurement: {var_name}")
            
            # Tokenize the description
            inputs = self.tokenizer(description, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embedding from ClinicalBERT
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding as the variable representation
                embedding = outputs.last_hidden_state[:, 0, :].cpu()
            
            embeddings.append(embedding)
        
        # Stack embeddings into a single tensor
        embeddings_tensor = torch.cat(embeddings, dim=0)
        
        # Save embeddings to file
        torch.save(embeddings_tensor, output_path)
        
        print(f"Saved embeddings with shape {embeddings_tensor.shape} to {output_path}")
        return embeddings_tensor
