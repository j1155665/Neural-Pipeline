"""
Sliding Window Partial Correlation Analysis

This module computes partial correlations between heading and choice signals
in neural activity across different modalities and coherences. It uses partial
regression to isolate each signal by removing confounding variables.

Author: [Your Name]
Date: [Date]
"""

import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class PartialCorrelationAnalyzer:
    """
    Analyzer for computing partial correlations between heading and choice signals.
    
    The analysis isolates heading and choice signals by regressing out confounding
    variables (PDW for heading, heading for choice), then computes the correlation
    between these isolated signals.
    """
    
    def __init__(self, subject, date):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        subject : str
            Subject identifier
        date : str
            Session date
        """
        self.subject = subject
        self.date = date
        self.save_dir = Path(rf'D:\Neural-Pipeline\results\analysis_single_neurons\partial_correlation')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self, spikes, behavior, valid_units, trial_mod, trial_coh, trial_del=0):
        """
        Prepare data for a specific condition.
        
        Parameters:
        -----------
        spikes : ndarray
            Neural activity (neurons x trials x time) or (neurons x trials)
        behavior : dict
            Behavioral data dictionary
        valid_units : ndarray or None
            Boolean array or indices of valid units
        trial_mod : int
            Modality to filter (1=vestibular, 2=visual, 3=combined)
        trial_coh : int
            Coherence level to filter
        trial_del : int or array-like
            Delta value(s) to filter
            
        Returns:
        --------
        filtered_spikes : ndarray
            Filtered spikes (trials x neurons)
        filtered_behavior : dict
            Filtered behavior dictionary
        trial_indices : ndarray
            Indices of selected trials
        """
        if valid_units is not None:
            spikes = spikes[valid_units, :]
        
        # Handle delta filtering
        del_mask = np.isin(behavior['delta'], trial_del) if isinstance(trial_del, (list, np.ndarray)) else (behavior['delta'] == trial_del)
        
        # Create condition mask
        mask = (behavior['modality'] == trial_mod) & (behavior['coherenceInd'] == trial_coh) & del_mask
        
        if np.sum(mask) == 0:
            return np.array([]), {}, np.array([])
        
        # Filter spikes (transpose to trials x neurons)
        filtered_spikes = spikes[:, mask].T
        
        # Filter behavior
        behavior_keys = ['choice', 'PDW', 'modality', 'headingInd', 'coherenceInd', 'goodtrial', 
                        'deltaInd', 'correct', 'oneTargChoice', 'oneTargConf', 'heading', 
                        'coherence', 'delta', 'RT']
        filtered_behavior = {k: behavior[k][mask] for k in behavior_keys if k in behavior}
        
        return filtered_spikes, filtered_behavior, np.where(mask)[0]

    def compute_heading_signal(self, spikes, behavior):
        """
        Compute heading signal by regressing out choice and PDW.
        
        This isolates the component of neural activity that relates to heading
        independent of choice and confidence.
        
        Parameters:
        -----------
        spikes : ndarray
            Neural activity (trials x neurons)
        behavior : dict
            Behavioral data
            
        Returns:
        --------
        heading_signals : ndarray
            Isolated heading signals (trials x neurons)
        """
        heading = behavior['heading']
        choice = (behavior['choice'] - 1).astype(int)
        pdw = behavior['PDW'].astype(int)
        n_trials = len(heading)
        
        # Initialize output
        heading_signals = np.zeros_like(spikes)
        
        # For each neuron, regress out choice and PDW, keep heading
        for neuron_idx in range(spikes.shape[1]):
            neural_activity = spikes[:, neuron_idx]
            
            # Create design matrix with nuisance variables (intercept, choice, PDW)
            X_nuisance = np.column_stack([np.ones(n_trials), choice, pdw])
            
            # Fit model and compute residuals
            try:
                coeffs_nuisance = np.linalg.lstsq(X_nuisance, neural_activity, rcond=None)[0]
                # Residual after removing choice and PDW = heading signal
                heading_signals[:, neuron_idx] = neural_activity - X_nuisance @ coeffs_nuisance
            except np.linalg.LinAlgError:
                # If singular matrix, use zeros
                heading_signals[:, neuron_idx] = 0
        
        return heading_signals

    def compute_choice_signal(self, spikes, behavior):
        """
        Compute choice signal by regressing out heading and PDW.
        
        This isolates the component of neural activity that relates to choice
        independent of stimulus and confidence.
        
        Parameters:
        -----------
        spikes : ndarray
            Neural activity (trials x neurons)
        behavior : dict
            Behavioral data
            
        Returns:
        --------
        choice_signals : ndarray
            Isolated choice signals (trials x neurons)
        """
        heading = behavior['heading']
        choice = (behavior['choice'] - 1).astype(int)
        pdw = behavior['PDW'].astype(int)
        n_trials = len(heading)
        
        # Initialize output
        choice_signals = np.zeros_like(spikes)
        
        # For each neuron, regress out heading and PDW, keep choice
        for neuron_idx in range(spikes.shape[1]):
            neural_activity = spikes[:, neuron_idx]
            
            # Create design matrix with nuisance variables (intercept, heading, PDW)
            X_nuisance = np.column_stack([np.ones(n_trials), heading, pdw])
            
            # Fit model and compute residuals
            try:
                coeffs_nuisance = np.linalg.lstsq(X_nuisance, neural_activity, rcond=None)[0]
                # Residual after removing heading and PDW = choice signal
                choice_signals[:, neuron_idx] = neural_activity - X_nuisance @ coeffs_nuisance
            except np.linalg.LinAlgError:
                # If singular matrix, use zeros
                choice_signals[:, neuron_idx] = 0
        
        return choice_signals

    def compute_partial_correlation_single_timepoint(self, spikes, behavior):
        """
        Compute partial correlation between heading and choice signals at a single timepoint.
        
        This is the core analysis: after isolating heading and choice signals,
        compute how correlated they are across trials for each neuron.
        
        Parameters:
        -----------
        spikes : ndarray
            Neural activity at one timepoint (trials x neurons)
        behavior : dict
            Behavioral data
            
        Returns:
        --------
        correlations : ndarray
            Partial correlations for each neuron
        p_values : ndarray
            P-values for each correlation
        heading_signals : ndarray
            Heading signals for each neuron (trials x neurons)
        choice_signals : ndarray
            Choice signals for each neuron (trials x neurons)
        """
        n_neurons = spikes.shape[1]
        
        # Get isolated signals
        heading_signals = self.compute_heading_signal(spikes, behavior)
        choice_signals = self.compute_choice_signal(spikes, behavior)
        
        # Initialize outputs
        correlations = np.zeros(n_neurons)
        p_values = np.zeros(n_neurons)
        
        # Compute correlation between heading and choice signals for each neuron
        for neuron_idx in range(n_neurons):
            h_sig = heading_signals[:, neuron_idx]
            c_sig = choice_signals[:, neuron_idx]
            
            # Check for valid data (non-zero variance)
            if np.std(h_sig) > 0 and np.std(c_sig) > 0:
                try:
                    corr, pval = pearsonr(h_sig, c_sig)
                    correlations[neuron_idx] = corr
                    p_values[neuron_idx] = pval
                except:
                    correlations[neuron_idx] = np.nan
                    p_values[neuron_idx] = np.nan
            else:
                correlations[neuron_idx] = np.nan
                p_values[neuron_idx] = np.nan
        
        return correlations, p_values, heading_signals, choice_signals

    def run_partial_correlation_analysis(self, spikes_data, behavior_data, time_axes, area, 
                                        valid_units=None, conditions=None, save_results=True,
                                        verbose=True):
        """
        Run partial correlation analysis across all conditions and timepoints.
        
        This is the main analysis function that processes all alignments, conditions,
        and timepoints.
        
        Parameters:
        -----------
        spikes_data : dict
            Dictionary with keys ['stimOn', 'saccOnset', 'postTargHold']
            Each value is ndarray (neurons x trials x time)
        behavior_data : dict
            Behavioral data dictionary
        time_axes : dict
            Time axes for each alignment
        area : str
            Brain area name
        valid_units : ndarray or None
            Boolean array or indices of valid units
        conditions : list of tuples or None
            List of (modality, coherence) tuples to analyze
            If None, analyzes default 5 conditions
        save_results : bool
            Whether to save results to disk
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        all_results : dict
            Dictionary with results for each alignment
        """
        
        if conditions is None:
            # Default 5 conditions
            conditions = [
                (1, 1),  # mod1 coh1 (vestibular only)
                (2, 1),  # mod2 coh1 (visual low coherence)
                (2, 2),  # mod2 coh2 (visual high coherence)
                (3, 1),  # mod3 coh1 (combined low coherence)
                (3, 2),  # mod3 coh2 (combined high coherence)
            ]
        
        if verbose:
            print(f"{'='*60}")
            print(f"Partial Correlation Analysis: {area}")
            print(f"Conditions: {conditions}")
            print(f"{'='*60}")
        
        # Get valid units
        spikes_example = spikes_data['stimOn']
        if valid_units is not None:
            sel_idx = np.where(valid_units)[0] if valid_units.dtype == bool else valid_units
            final_units = np.zeros(spikes_example.shape[0], dtype=bool)
            final_units[sel_idx] = True
        else:
            sel_idx = np.arange(spikes_example.shape[0])
            final_units = None
        
        n_units = len(sel_idx)
        if verbose:
            print(f"Units: {n_units}")
        
        all_results = {}
        
        # Loop over alignments
        for alignment in ['stimOn', 'saccOnset', 'postTargHold']:
            if alignment not in spikes_data:
                continue
                
            if verbose:
                print(f"\n{'='*40}\n{alignment}\n{'='*40}")
            
            spikes = spikes_data[alignment]
            n_times = spikes.shape[2]
            
            alignment_results = {}
            
            # Loop over conditions
            for mod, coh in conditions:
                if verbose:
                    print(f"\n  Condition: mod={mod}, coh={coh}")
                
                condition_results = {
                    'correlations': [],      # time x neurons
                    'p_values': [],          # time x neurons
                    'mean_corr': [],         # time
                    'median_corr': [],       # time
                    'n_significant': [],     # time (p < 0.05)
                    'n_trials': [],          # time
                    'heading_signals': [],   # time x trials x neurons (optional, can be large)
                    'choice_signals': [],    # time x trials x neurons (optional, can be large)
                }
                
                # Loop over time
                for t in range(n_times):
                    if verbose and t % 10 == 0:
                        print(f"    Time {t}/{n_times}")
                    
                    spikes_t