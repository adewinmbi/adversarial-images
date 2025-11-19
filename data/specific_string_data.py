import json
import numpy as np
from pathlib import Path
from typing import Tuple, List

from .dataset import Dataset
from .fetch_alpaca import download_alpaca_dataset


class SpecificStringDataset(Dataset):
    """
    Dataset for training adversarial images to produce a specific target string.
    
    Uses alpaca.json for instructions, with a fixed target string as the label
    for all samples.
    """
    
    def __init__(self, train_split: float, val_split: float, test_split: float, 
                 target_string: str, dataset_path: str = 'data/alpaca.json'):
        """
        Initialize the specific string dataset.
        
        Args:
            train_split: Proportion for training
            val_split: Proportion for validation
            test_split: Proportion for testing
            target_string: The target string to use as label for all samples
            dataset_path: Path to alpaca.json file
        """
        super().__init__(train_split, val_split, test_split)
        self.target_string = target_string
        self.dataset_path = Path(dataset_path)
        self._load_dataset()
    
    def _load_dataset(self):
        """Load alpaca.json dataset."""
        # Download if doesn't exist
        if not self.dataset_path.exists():
            print("Dataset not found, downloading...", flush=True)
            if not download_alpaca_dataset():
                raise RuntimeError("Failed to download alpaca dataset")
        
        # Load the dataset
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Extract instructions
        self.instructions = [item['instruction'] for item in dataset]
        
        # Create splits with fixed random seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(len(self.instructions))
        
        train_end = int(self.train_split * len(indices))
        val_end = int((self.train_split + self.val_split) * len(indices))
        
        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]
    
    def get_splits(self) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        """
        Return dataset splits.
        
        Returns:
            (train_x, train_y, val_x, val_y, test_x, test_y)
            where x are instructions and y are the target string (repeated)
        """
        train_x = [self.instructions[i] for i in self.train_indices]
        val_x = [self.instructions[i] for i in self.val_indices]
        test_x = [self.instructions[i] for i in self.test_indices]
        
        # All labels are the same target string
        train_y = [self.target_string] * len(train_x)
        val_y = [self.target_string] * len(val_x)
        test_y = [self.target_string] * len(test_x)
        
        return train_x, train_y, val_x, val_y, test_x, test_y

