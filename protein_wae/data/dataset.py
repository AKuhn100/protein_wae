"""
Dataset class for protein sequences.
"""

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm


class SimpleSeqRecord:
    """Simple sequence record to replace BioPython dependency."""
    def __init__(self, seq: str, id: str):
        self.seq = seq
        self.id = id


class SimpleFastaParser:
    """Simple FASTA parser to replace BioPython SeqIO."""
    
    @staticmethod
    def parse(fasta_path: str) -> List[SimpleSeqRecord]:
        """Parse FASTA file and return list of sequence records."""
        records = []
        
        with open(fasta_path, 'r') as f:
            current_seq = ""
            current_id = ""
            
            for line in f:
                line = line.strip()
                
                if line.startswith('>'):
                    # Save previous record if exists
                    if current_seq:
                        records.append(SimpleSeqRecord(current_seq, current_id))
                    
                    # Start new record
                    current_id = line[1:]
                    current_seq = ""
                else:
                    # Append to current sequence
                    current_seq += line
            
            # Save last record
            if current_seq:
                records.append(SimpleSeqRecord(current_seq, current_id))
                
        return records


class ProteinDataset(Dataset):
    """Dataset for protein sequences with support for different decoder types."""
    
    def __init__(self, 
                 fasta_path: str,
                 tokenizer,
                 min_len: int,
                 max_len: int,
                 decoder_type: str = "causal",
                 seed: int = 42):
        """
        Initialize protein dataset.
        
        Args:
            fasta_path: Path to FASTA file
            tokenizer: Tokenizer instance
            min_len: Minimum sequence length
            max_len: Maximum sequence length
            decoder_type: Type of decoder ("causal" or "plm")
            seed: Random seed for shuffling
        """
        self.tokenizer = tokenizer
        self.min_len = min_len
        self.max_len = max_len
        self.decoder_type = decoder_type
        
        # For causal decoder, we need space for BOS/EOS tokens
        self.max_len_aa = max_len
        self.max_tensor_len = max_len + 2 if decoder_type == "causal" else max_len
        
        # Load and filter sequences
        self.records = []
        self.lengths = []
        self._load_sequences(fasta_path, seed)
        
    def _load_sequences(self, fasta_path: str, seed: int):
        """Load and filter sequences from FASTA file."""
        dropped = Counter()
        
        print(f"\n📂 Loading sequences from {fasta_path}...")
        all_records = SimpleFastaParser.parse(fasta_path)
        
        # Shuffle for reproducibility
        random.Random(seed).shuffle(all_records)
        
        for rec in tqdm(all_records, desc="Filtering sequences"):
            seq = str(rec.seq).upper().replace("*", "")
            
            # Length filter
            if not (self.min_len <= len(seq) <= self.max_len):
                if len(seq) < self.min_len:
                    dropped["too_short"] += 1
                else:
                    dropped["too_long"] += 1
                continue
            
            # Character filter
            if not self.tokenizer.is_valid_sequence(seq):
                dropped["invalid_chars"] += 1
                continue
            
            self.records.append(seq)
            self.lengths.append(len(seq))
        
        print(f"✓ Loaded {len(self.records)} sequences")
        if dropped:
            print(f"  Dropped {sum(dropped.values())}: {dict(dropped)}")
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        seq = self.records[idx]
        length = len(seq)
        
        if self.decoder_type == "causal":
            return self._get_causal_item(seq, length)
        else:
            return self._get_plm_item(seq, length)
    
    def _get_causal_item(self, seq: str, length: int) -> Dict[str, torch.Tensor]:
        """Prepare item for causal autoregressive decoder."""
        # Encoder input (without special tokens)
        enc_indices = self.tokenizer.encode(seq, add_special_tokens=False)
        enc_padded = enc_indices + [self.tokenizer.pad_idx] * (self.max_len_aa - len(enc_indices))
        enc_one_hot = F.one_hot(torch.tensor(enc_padded), num_classes=self.tokenizer.vocab_size).float()
        
        # Decoder input and target (with special tokens)
        dec_indices = self.tokenizer.encode(seq, add_special_tokens=True)
        dec_padded = dec_indices + [self.tokenizer.pad_idx] * (self.max_tensor_len - len(dec_indices))
        
        # For autoregressive: decoder input is shifted right
        decoder_input = dec_padded[:-1]  # Everything except last token
        target_sequence = dec_padded[1:]  # Everything except first token (BOS)
        
        return {
            "encoder_input": enc_one_hot.permute(1, 0),  # (vocab_size, seq_len)
            "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
            "target_sequence": torch.tensor(target_sequence, dtype=torch.long),
            "length": torch.tensor(length + 1, dtype=torch.long)  # +1 for EOS token
        }
    
    def _get_plm_item(self, seq: str, length: int) -> Dict[str, torch.Tensor]:
        """Prepare item for permutation language model decoder."""
        # Encoder input (without special tokens)
        enc_indices = self.tokenizer.encode(seq, add_special_tokens=False)
        enc_padded = enc_indices + [self.tokenizer.pad_idx] * (self.max_len_aa - len(enc_indices))
        enc_one_hot = F.one_hot(torch.tensor(enc_padded), num_classes=self.tokenizer.vocab_size).float()
        
        # Decoder input (no special tokens for PLM)
        dec_indices = self.tokenizer.encode(seq, add_special_tokens=False)
        dec_padded = dec_indices + [self.tokenizer.pad_idx] * (self.max_len_aa - len(dec_indices))
        
        return {
            "encoder_input": enc_one_hot.permute(1, 0),  # (vocab_size, seq_len)
            "decoder_input": torch.tensor(dec_padded, dtype=torch.long),
            "target_sequence": torch.tensor(dec_padded, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long)
        }


def create_data_loaders(config, tokenizer) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer instance
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = ProteinDataset(
        fasta_path=config.data.fasta_path,
        tokenizer=tokenizer,
        min_len=config.data.min_seq_len,
        max_len=config.data.max_seq_len,
        decoder_type=config.model.decoder_type,
        seed=config.data.seed
    )
    
    # Split into train and validation
    val_size = int(len(full_dataset) * config.data.val_fraction)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.data.seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    print(f"✓ Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader