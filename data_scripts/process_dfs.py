import os
import pandas as pd
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#############################################
# 1. Discharge Notes Processor
#############################################

class MIMICDischargeNotesProcessor:
    def __init__(self, disch_path='/Users/riccardoconci/Local_documents/!!MIMIC/note/discharge.csv'):
        self.disch_path = disch_path
        # Load only the necessary columns and index by hadm_id for speed.
        self.discharge_df = pd.read_csv(self.disch_path)[['hadm_id', 'charttime', 'text']]
        self.discharge_df = self.discharge_df.set_index('hadm_id')
        self.memory_cache = {}
        
        self.sections_to_include = [
            "History of Present Illness",
            "Past Medical History",
            "Social History",
            "Family History",
            "Physical Exam",
            "Brief Hospital Course",
            "IMPRESSION",
            "DISCHARGE PHYSICAL EXAM",
            "ACUTE/ACTIVE ISSUES",
            "Discharge Diagnosis",
        ]
        self.sections_to_ignore = [
            "pertinent results", 
            "DISCHARGE LABS", 
            "Medications on Admission", 
            "Medications on Discharge", 
            "Discharge Medications", 
            "Discharge Disposition", 
            "Discharge Instructions"
        ]
    
    def get_discharge_chunks(self, hadm_id):
        """
        Retrieve or compute text chunks for a given hadm_id.
        Uses caching to avoid re-processing.
        """
        if hadm_id in self.memory_cache:
            return self.memory_cache[hadm_id]
        
        discharge_note_df = self.load_discharge_note(hadm_id)
        chunks = self.process_discharge_note(discharge_note_df)
        self.memory_cache[hadm_id] = chunks
        return chunks
    
    def load_discharge_note(self, hadm_id):
        """Quickly load the discharge note corresponding to hadm_id."""
        try:
            note_df = self.discharge_df.loc[[hadm_id]]
        except KeyError:
            note_df = pd.DataFrame()
        return note_df
    
    def process_discharge_note(self, discharge_note_df):
        """Extract selected sections and split the note into text chunks."""
        if discharge_note_df.empty:
            return []
        # Use the 'text' column.
        note = discharge_note_df['text'].iloc[0]
        selected_blocks = self.parse_included_sections(note, self.sections_to_include, self.sections_to_ignore)
        chunks = self.chunkify_blocks_nltk(selected_blocks)
        return chunks
    
    def parse_included_sections(self, text, included_headings, excluded_headings, case_insensitive=True):
        """Extract and return text blocks for the included sections."""
        all_headings = list(set(included_headings + excluded_headings))
        pattern_str = "(" + "|".join(re.escape(h) for h in all_headings) + "):"
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern_str, flags=flags)
        
        matches = list(regex.finditer(text))
        sentinel = type("SentinelMatch", (), {})()
        setattr(sentinel, "start", lambda: len(text))
        matches.append(sentinel)
        
        included_set = set(h.lower() for h in included_headings)
        excluded_set = set(h.lower() for h in excluded_headings)
        
        selected_blocks = []
        selecting = False
        current_heading_text = None
        for i in range(len(matches) - 1):
            this_match = matches[i]
            next_match = matches[i+1]
            this_heading_start = this_match.start()
            next_heading_start = next_match.start()
            heading_matched = this_match.group(1) if hasattr(this_match, "group") else None
            
            if heading_matched:
                heading_lower = heading_matched.lower()
                if heading_lower in included_set:
                    selecting = True
                    current_heading_text = heading_matched.strip()
                elif heading_lower in excluded_set:
                    selecting = False
                    current_heading_text = None
            
            if selecting and heading_matched:
                content = text[this_heading_start:next_heading_start]
                selected_blocks.append({
                    "start_offset": this_heading_start,
                    "end_offset": next_heading_start,
                    "heading": current_heading_text,
                    "content": content
                })
        return selected_blocks
    
    def chunkify_blocks_nltk(self, blocks, words_per_chunk=150):
        """Tokenize each block using NLTK and reassemble them into uniform chunks."""
        all_words = []
        for block in blocks:
            words = word_tokenize(block['content'])
            all_words.extend(words)
        total_words = len(all_words)
        chunks = []
        for i in range(0, total_words, words_per_chunk):
            chunk = " ".join(all_words[i: i + words_per_chunk])
            chunks.append(chunk)
        return chunks

#############################################
# 2. Dataset & DataLoader for Embedding Extraction
#############################################

class DSDataset(Dataset):
    def __init__(self, processor, merged_pickle_path='../temp_dfs/merged_with_disch_df_final_filtered.pkl'):
        """
        The dataset returns (hadm_id, chunks) for each discharge note.
        """
        self.processor = processor
        merged_df = pickle.load(open(merged_pickle_path, 'rb'))
        discharge_df = pd.read_csv(processor.disch_path)[['hadm_id', 'charttime', 'text']]
        filtered = discharge_df['hadm_id'].isin(merged_df['hadm_id'])
        self.discharge_df = discharge_df.loc[filtered].set_index('hadm_id')
        self.hadm_ids = list(self.discharge_df.index.unique())
    
    def __len__(self):
        return len(self.hadm_ids)
    
    def __getitem__(self, idx):
        hadm_id = self.hadm_ids[idx]
        chunks = self.processor.get_discharge_chunks(hadm_id)
        return hadm_id, chunks

def get_ds_dataloader(processor, batch_size=8, num_workers=0):
    dataset = DSDataset(processor)
    # Each batch is a list of tuples (hadm_id, chunks)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=lambda x: x)

#############################################
# 3. Embedding Extractor (Transformer Only)
#############################################

class DSEmbeddingExtractor(nn.Module):
    def __init__(self, model_name="medicalai/ClinicalBERT"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.hidden_dim = self.transformer.config.hidden_size  # e.g., 768
    
    def forward(self, chunks_batch):
        """
        Args:
            chunks_batch: A list (batch_size) of lists of text chunks.
            
        Returns:
            batch_embeddings: A list (batch_size) where each element is a tensor of shape [num_chunks, hidden_dim].
        """
        batch_size = len(chunks_batch)
        all_chunks = []
        sample_indices = []
        
        # Flatten all chunks (and remember which sample each belongs to)
        for i, chunks in enumerate(chunks_batch):
            for chunk in chunks:
                all_chunks.append(chunk)
                sample_indices.append(i)
        
        # If there are no chunks in any sample, return empty tensors for each
        if len(all_chunks) == 0:
            return [torch.empty(0, self.hidden_dim, device=device) for _ in range(batch_size)]
        
        inputs = self.tokenizer(all_chunks, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            self.transformer.eval()
            outputs = self.transformer(**inputs, output_hidden_states=True)
            # Use the CLS token embedding from the last hidden state
            last_hidden_states = outputs.hidden_states[-1]
            cls_embeddings = last_hidden_states[:, 0, :]  # [num_chunks, hidden_dim]
        
        # Reassemble embeddings per sample.
        batch_embeddings = [[] for _ in range(batch_size)]
        for idx, emb in zip(sample_indices, cls_embeddings):
            batch_embeddings[idx].append(emb)
        
        # For each sample, stack embeddings (if any) into a tensor.
        final_embeddings = []
        for emb_list in batch_embeddings:
            if emb_list:
                final_embeddings.append(torch.stack(emb_list, dim=0))
            else:
                final_embeddings.append(torch.empty(0, self.hidden_dim, device=device))
        return final_embeddings

#############################################
# 4. Batch Processing and Saving Embeddings
#############################################

if __name__ == '__main__':
    output_dir = "../temp_dfs/DS_embeddings"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing processor...")
    processor = MIMICDischargeNotesProcessor()
    print("Initializing dataloader...")
    dataloader = get_ds_dataloader(processor, batch_size=8)
    
    extractor = DSEmbeddingExtractor().to(device)
    extractor.eval()
    
    # Process each batch with tqdm for progress tracking.
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        hadm_ids = []
        chunks_batch = []
        for hadm_id, chunks in batch:
            hadm_ids.append(hadm_id)
            chunks_batch.append(chunks)
        
        with torch.no_grad():
            # Get a list (length batch_size) of tensors [num_chunks, hidden_dim]
            batch_embeddings = extractor(chunks_batch)
        
        # Save each embedding tensor as "embedding_{hadm_id}.pt"
        for h_id, emb_tensor in zip(hadm_ids, batch_embeddings):
            save_path = os.path.join(output_dir, f"embedding_{h_id}.pt")
            torch.save(emb_tensor.cpu(), save_path)
            # Optionally, print progress per saved file.