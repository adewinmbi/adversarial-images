from abc import ABC, abstractmethod


class Dataset(ABC):
    """Abstract base class for datasets used in adversarial image training."""
    
    def __init__(self, train_split: float, val_split: float, test_split: float):
        """
        Initialize dataset with split ratios.
        
        Args:
            train_split: Proportion of data for training (e.g., 0.7)
            val_split: Proportion of data for validation (e.g., 0.2)
            test_split: Proportion of data for testing (e.g., 0.1)
        """
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Validate splits sum to 1.0
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")
    
    @abstractmethod
    def _load_dataset(self):
        """Load the raw dataset. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_splits(self):
        """
        Return the dataset splits.
        
        Returns:
            tuple: (train_x, train_y, val_x, val_y, test_x, test_y)
                   where x are instructions/prompts and y are target strings/labels.
                   All should be lists of strings.
        """
        pass
