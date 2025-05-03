import os
import pandas as pd
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Get the current working directory where the script is being run from
WORKING_DIR = os.getcwd()

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#############################################
# 1. Discharge Notes Processor
#############################################

class MIMICDischargeNotesProcessor:
    def __init__(self, cache_dir='temp_dfs_lite'):
        # Make cache_dir relative to working directory
        self.cache_dir = os.path.join(WORKING_DIR, cache_dir)

        discharge_pickle_path = os.path.join(self.cache_dir, 'discharge_df_filtered.pkl')
        if os.path.exists(discharge_pickle_path):
            self.discharge_df = pickle.load(open(discharge_pickle_path, 'rb'))
            print(f"Loaded discharge_df from {discharge_pickle_path}")
        else:
            raise FileNotFoundError(f"provide discharge_df_filtered.pkl file or CSV file")
          
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
    
    def chunkify_blocks_nltk(self, blocks, words_per_chunk=100):
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
    def __init__(self, processor):
        """
        The dataset returns (hadm_id, chunks) for each discharge note.
        
        Args:
            processor: The MIMICDischargeNotesProcessor instance
            cache_dir: Directory containing cached files (default: 'temp_dfs')
            merged_pickle_path: Optional path to the merged_with_disch_df pickle file
        """
        self.processor = processor
        self.cache_dir = processor.cache_dir
        
        # Use provided merged_pickle_path or construct from cache_dir
        merged_pickle_path = os.path.join(self.cache_dir, "merged_with_disch_df_final_filtered.pkl")
        merged_df = pickle.load(open(merged_pickle_path, 'rb'))
        discharge_df = processor.discharge_df

        if 'hadm_id' in discharge_df.columns:
            filtered = discharge_df['hadm_id'].isin(merged_df['hadm_id'])
            self.discharge_df = discharge_df.loc[filtered].set_index('hadm_id')

        self.hadm_ids = list(discharge_df.index.unique())
    
    def __len__(self):
        return len(self.hadm_ids)
    
    def __getitem__(self, idx):
        hadm_id = self.hadm_ids[idx]
        chunks = self.processor.get_discharge_chunks(hadm_id)
        return hadm_id, chunks

#############################################
# 3. Embedding Extractor (Transformer Only)
#############################################
class DSEmbeddingExtractor(nn.Module):
    def __init__(self, model_name="medicalai/ClinicalBERT"):
        super().__init__()
        self.device = DEVICE
        # Use the base AutoModel (no MLM head) for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.transformer.eval()  # always inference mode
        
        self.hidden_dim = self.transformer.config.hidden_size
        self.max_length = 512
        self.tokenizer.model_max_length = self.max_length
        
        print(f"Using model '{model_name}' with max sequence length {self.max_length} on {DEVICE}")

    def forward(self, chunks_batch):
        """
        Args:
            chunks_batch: List of tuples (hadm_id, chunks), 
                          where chunks is a list of strings.
        Returns:
            final_embeddings: List (batch_size) of Tensors [num_chunks, hidden_dim].
        """
        batch_size = len(chunks_batch)

        # Flatten all non-empty chunks, track which sample they came from
        flattened, sample_map = [], []
        for i, (_, chunks) in enumerate(chunks_batch):
            for c in chunks or []:
                if c and c.strip():
                    flattened.append(c)
                    sample_map.append(i)
        
        # Early exit if nothing to encode
        if not flattened:
            empty = torch.zeros(0, self.hidden_dim, device=self.device)
            return [empty.clone() for _ in range(batch_size)]

        # Tokenize once, with statistics
        enc = self.tokenizer(
            flattened,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_lengths = enc["input_ids"].ne(self.tokenizer.pad_token_id).sum(-1)
        print(
            f"Chunk lengths — min: {input_lengths.min().item()}, "
            f"max: {input_lengths.max().item()}, "
            f"avg: {input_lengths.float().mean().item():.1f}"
        )

        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.transformer(**enc, output_hidden_states=False)
            # out.last_hidden_state: [total_chunks, seq_len, hidden_dim]
            cls_embeds = out.last_hidden_state[:, 0, :]

        # Group back into per-sample lists
        grouped = [[] for _ in range(batch_size)]
        for idx, emb in zip(sample_map, cls_embeds):
            grouped[idx].append(emb)

        # Stack and handle samples that had no valid chunks
        result = []
        for i, lst in enumerate(grouped):
            if lst:
                result.append(torch.stack(lst, dim=0))
            else:
                # if no text, return a single zero-vector
                print(f"Warning: no valid chunks for sample {chunks_batch[i][0]}")
                result.append(torch.zeros(1, self.hidden_dim, device=self.device))

        return result
    


#############################################
# 4. Batch Processing and Saving Embeddings
#############################################

if __name__ == '__main__':
    # Make output_dir relative to working directory
    output_dir = os.path.join(WORKING_DIR, "temp_dfs_lite/DS_embeddings")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing processor...")
    processor = MIMICDischargeNotesProcessor(cache_dir='temp_dfs_lite')
    print("Initializing dataloader...")
    dataset = DSDataset(processor)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=lambda x: x)
    
    extractor = DSEmbeddingExtractor()
    extractor.eval()

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        hadm_ids, chunks_lists = zip(*batch)
        for h_id, ch in zip(hadm_ids, chunks_lists):
            print(h_id, len(ch))

        with torch.no_grad():
            # Pass the tuples directly—this matches DSEmbeddingExtractor.forward’s signature
            batch_embeddings = extractor(batch)
        
        for h_id, emb_tensor in zip(hadm_ids, batch_embeddings):
             print('saving embedding for', h_id, 'with shape', emb_tensor.shape)
             save_path = os.path.join(output_dir, f"embedding_{h_id}.pt")
             torch.save(emb_tensor.cpu(), save_path)