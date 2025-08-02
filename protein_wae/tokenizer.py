"""
Tokenizer for protein sequences with special tokens support.
"""

from typing import List, Union
import torch


class Tokenizer:
    """Tokenizer for protein sequences with special tokens."""
    
    def __init__(self):
        # Standard amino acids
        self.AA20 = "ACDEFGHIKLMNPQRSTVWY"
        
        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.GAP_TOKEN = "-"
        self.BOS_TOKEN = "<BOS>"
        self.EOS_TOKEN = "<EOS>"
        self.MASK_TOKEN = "<MASK>"
        
        # Build vocabulary
        self.token_list = [
            self.PAD_TOKEN, 
            self.BOS_TOKEN, 
            self.EOS_TOKEN, 
            self.GAP_TOKEN,
            self.MASK_TOKEN
        ] + list(self.AA20)
        
        self.token_to_idx = {token: i for i, token in enumerate(self.token_list)}
        self.idx_to_token = {i: token for i, token in enumerate(self.token_list)}
        
        # Store useful properties
        self.vocab_size = len(self.token_list)
        self.pad_idx = self.token_to_idx[self.PAD_TOKEN]
        self.bos_idx = self.token_to_idx[self.BOS_TOKEN]
        self.eos_idx = self.token_to_idx[self.EOS_TOKEN]
        self.gap_idx = self.token_to_idx[self.GAP_TOKEN]
        self.mask_idx = self.token_to_idx[self.MASK_TOKEN]
        
        # Valid characters for sequences
        self.valid_chars = self.AA20 + self.GAP_TOKEN
    
    def encode(self, sequence: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode a protein sequence to token indices.
        
        Args:
            sequence: Protein sequence string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token indices
        """
        indices = [self.token_to_idx[aa] for aa in sequence]
        
        if add_special_tokens:
            indices = [self.bos_idx] + indices + [self.eos_idx]
            
        return indices
    
    def encode_batch(self, sequences: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        """Encode a batch of sequences."""
        return [self.encode(seq, add_special_tokens) for seq in sequences]
    
    def decode(self, indices: Union[List[int], torch.Tensor], remove_special_tokens: bool = True) -> str:
        """
        Decode token indices back to protein sequence.
        
        Args:
            indices: Token indices (list or tensor)
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded protein sequence string
        """
        tokens = []
        
        for idx in indices:
            # Handle both tensor and int
            idx_item = idx.item() if isinstance(idx, torch.Tensor) else idx
            
            # Skip padding
            if idx_item == self.pad_idx:
                continue
                
            # Skip special tokens if requested
            if remove_special_tokens and idx_item in [self.bos_idx, self.eos_idx, self.mask_idx]:
                continue
                
            tokens.append(self.idx_to_token[idx_item])
            
        return "".join(tokens)
    
    def decode_batch(self, indices_batch: Union[List[List[int]], torch.Tensor], 
                     remove_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token indices."""
        if isinstance(indices_batch, torch.Tensor):
            indices_batch = indices_batch.tolist()
        return [self.decode(indices, remove_special_tokens) for indices in indices_batch]
    
    def is_valid_sequence(self, sequence: str) -> bool:
        """Check if a sequence contains only valid characters."""
        return all(char in self.valid_chars for char in sequence)
    
    def get_special_token_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get mask for special tokens.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Boolean mask where True indicates a special token
        """
        special_tokens = {self.pad_idx, self.bos_idx, self.eos_idx, self.mask_idx}
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        
        for token_idx in special_tokens:
            mask |= (token_ids == token_idx)
            
        return mask
    
    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size}, valid_chars='{self.valid_chars}')"