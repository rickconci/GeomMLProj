#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the variable embedding generation process.
This script simplifies running the generator with default parameters.
"""

import os
import argparse
import torch
from generate_variable_embeddings import VariableEmbeddingGenerator
from data import MIMICDemographicsLoader, MIMICClinicalEventsProcessor

def main():
    parser = argparse.ArgumentParser(description='Run the variable embedding generation process')
    parser.add_argument('--base_path', type=str, 
                        default='/Users/riccardoconci/Local_documents/!!MIMIC',
                        help='Path to MIMIC-IV data')
    parser.add_argument('--temp_dfs_path', type=str, 
                        default='temp_dfs',
                        help='Path to directory for processed files')
    args = parser.parse_args()
    
    print(f"Starting variable embedding generation with data from {args.base_path}")
    
    # Create cache directory if needed
    os.makedirs(args.temp_dfs_path, exist_ok=True)
    
    # First get all variable names by loading minimal data
    print("Loading minimal data to get variable names...")
    
    # Initialize demographics loader
    demo_loader = MIMICDemographicsLoader(
        args.base_path, 
        args.temp_dfs_path
    )
    
    # Load demographics
    demo_loader.load_demographics()
    
    # Initialize events processor
    event_processor = MIMICClinicalEventsProcessor(
        args.base_path,
        demo_loader.get_hadm_ids(),
        args.temp_dfs_path
    )
    
    # Process events minimally to get variable names
    event_processor.load_all_events(load_only_essential=True)
    event_processor.discharge_df = demo_loader.discharge_df
    event_processor.merged_with_disch_df = demo_loader.merged_with_disch_df
    event_processor.process_events()
    
    # Get variable names
    var_names = event_processor.provide_physio_var_names()
    print(f"Found {len(var_names)} variables")
    
    # Create embedding generator
    generator = VariableEmbeddingGenerator(
        base_path=args.base_path,
        temp_dfs_path=args.temp_dfs_path
    )
    
    # Generate descriptions
    descriptions = generator.generate_descriptions(var_names)
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(descriptions)
    
    print(f"Variable embedding generation complete!")
    print(f"Generated embeddings with shape: {embeddings.shape}")
    print(f"Embeddings saved to: {os.path.join(args.temp_dfs_path, 'mimic4_bert_var_rep_gpt_source.pt')}")
    print(f"Descriptions saved to: {os.path.join(args.temp_dfs_path, 'variable_descriptions.json')}")
    print("\nNow you can run train_with_wrapper.py to train the model!")

if __name__ == "__main__":
    main() 