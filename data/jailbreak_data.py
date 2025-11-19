import numpy as np
from typing import Tuple, List
from datasets import load_dataset

from .dataset import Dataset


class JailbreakDataset(Dataset):
    """
    Dataset for jailbreak adversarial attacks using AdvBench.
    
    Uses the "walledai/AdvBench" HuggingFace dataset, which contains
    harmful prompts and their corresponding jailbreak targets.
    """
    
    def __init__(self, train_split: float, val_split: float, test_split: float):
        """
        Initialize the jailbreak dataset.
        
        Args:
            train_split: Proportion for training
            val_split: Proportion for validation
            test_split: Proportion for testing
        """
        super().__init__(train_split, val_split, test_split)
        self._load_dataset()
    
    def _load_dataset(self):
        """Load AdvBench dataset from HuggingFace."""
        print("Loading AdvBench dataset from HuggingFace...", flush=True)
        
        # Load the dataset (all data is in 'train' split)
        dataset = load_dataset("walledai/AdvBench", split='train')
        
        # Extract prompts and targets
        self.prompts = dataset['prompt']
        self.targets = dataset['target']
        
        # Create splits with fixed random seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(len(self.prompts))
        
        train_end = int(self.train_split * len(indices))
        val_end = int((self.train_split + self.val_split) * len(indices))
        
        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]
        
        print(f"Loaded {len(self.prompts)} samples from AdvBench", flush=True)
        print(f"  Train: {len(self.train_indices)}, Val: {len(self.val_indices)}, Test: {len(self.test_indices)}", flush=True)
    
    def get_splits(self) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        """
        Return dataset splits.
        
        Returns:
            (train_x, train_y, val_x, val_y, test_x, test_y)
            where x are prompts and y are target jailbreak responses
        """
        train_x = [self.prompts[i] for i in self.train_indices]
        train_y = [self.targets[i] for i in self.train_indices]
        
        val_x = [self.prompts[i] for i in self.val_indices]
        val_y = [self.targets[i] for i in self.val_indices]
        
        test_x = [self.prompts[i] for i in self.test_indices]
        test_y = [self.targets[i] for i in self.test_indices]
        
        return train_x, train_y, val_x, val_y, test_x, test_y

