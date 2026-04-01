import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ConfigLoader:
    
    def __init__(self, subject, date, hyperparams_file=None):
        """
        Initialize ConfigLoader to load configuration settings.
        
        Parameters:
        -----------
        subject : str
            Subject identifier
        date : str
            Date identifier
        hyperparams_file : str, optional
            Path to hyperparameters file. If None, uses default path.
        """
        self.subject = subject
        self.date = date
        
        # Set default hyperparameters file path if not provided
        if hyperparams_file is None:
            hyperparams_file = f"D:\\Neural-Pipeline\\results\\analysis_population\\hyperparameters\\{self.subject}_{self.date}_all_areas_all_targets_combined.npy"
        
        self.hyperparams_file = Path(hyperparams_file)
        
        # Load hyperparameters
        if not self.hyperparams_file.exists():
            raise FileNotFoundError(f"File not found: {self.hyperparams_file}")
        
        self.all_hyperparams = np.load(self.hyperparams_file, allow_pickle=True).item()
    
    def get_config(self, brain_area=None, decode_target=None):
        """
        Load configuration for a specific brain area and decode target.
        
        Returns:
        --------
        dict : Configuration dictionary containing brain_area, decode_target, and config params
        """
        if brain_area is None:
            brain_area = list(self.all_hyperparams.keys())[0]
        if decode_target is None:
            decode_target = list(self.all_hyperparams[brain_area].keys())[0]
        
        if brain_area not in self.all_hyperparams:
            raise ValueError(f"Brain area '{brain_area}' not found in hyperparameters")
        if decode_target not in self.all_hyperparams[brain_area]:
            raise ValueError(f"Decode target '{decode_target}' not found for area '{brain_area}'")
        
        config = self.all_hyperparams[brain_area][decode_target]
        
        return {
            'brain_area': brain_area,
            'decode_target': decode_target,
            'config': config
        }
    
    def get_all_areas(self):
        """Get list of all available brain areas."""
        return list(self.all_hyperparams.keys())
    
    def get_all_targets(self, brain_area):
        """Get list of all available decode targets for a brain area."""
        if brain_area not in self.all_hyperparams:
            raise ValueError(f"Brain area '{brain_area}' not found")
        return list(self.all_hyperparams[brain_area].keys())
