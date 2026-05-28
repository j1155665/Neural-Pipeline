"""
Partial Correlation Analysis Following Gu et al. (2008) Method
Modified for Dataset with PDW - MULTIPLE TRIAL FILTERING OPTIONS

This module computes partial correlations following the method from:
Gu, Angelaki & DeAngelis (2008) - Neural correlates of multisensory cue integration
As described in: Zaidel, DeAngelis & Angelaki (2017)

MODIFICATIONS FOR THIS DATASET:
1. Include PDW (post-decision wager) as nuisance variable
2. Center (demean) predictors and FR across trials before residualization
3. stimOn alignment is stimulus-locked (no RT centering)
4. Data already binned at 20ms
5. IFR windows: 200ms (10 bins), stepped by 20ms (1 bin)
6. Time index from window center
7. Choice coded as -1 (leftward), +1 (rightward)

TRIAL FILTERING OPTIONS:
1. small_headings: -10° < heading < 10° (all trials)
2. errors_zeros: error trials + all zero heading trials
3. high_pdw_small: high PDW trials with -10° < heading < 10°
4. low_pdw_small: low PDW trials with -10° < heading < 10°

Author: [Your Name]
Date: [Date]
"""

import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class PartialCorrelationAnalyzer:
    """
    Analyzer for computing partial correlations following Gu et al. (2008) method,
    modified to include PDW and use trial-wise centering.
    
    Data is pre-binned at 20ms, so 200ms window = 10 bins.
    """
    
    def __init__(self, subject, date, session_data=None, 
                 window_size_sec=0.2, step_size_sec=0.02, bin_size_sec=0.020):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        subject : str
            Subject identifier
        date : str
            Session date
        session_data : dict, optional
            Session data dictionary containing unit_info and units_data for ID mapping
        window_size_sec : float
            IFR window size in seconds (default: 0.2s = 200ms = 10 bins)
        step_size_sec : float
            IFR step size in seconds (default: 0.02s = 20ms = 1 bin for max flexibility)
        bin_size_sec : float
            Original bin size of spike data in seconds (default: 0.020s = 20ms)
        """
        self.subject = subject
        self.date = date
        self.session_data = session_data
        self.save_dir = Path(r'D:\Neural-Pipeline\results\analysis_single_neurons\dot3DMP_partialcorr')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # IFR parameters (from paper, adapted to 20ms bins)
        self.window_size_sec = window_size_sec
        self.step_size_sec = step_size_sec
        self.bin_size_sec = bin_size_sec
        
        # Calculate bin counts
        self.window_bins = int(np.round(window_size_sec / bin_size_sec))  # 10 bins
        self.step_bins = int(np.round(step_size_sec / bin_size_sec))      # 1 bin
        
        # Extract unit information if available
        self.unit_info = session_data.get('unit_info') if session_data else None
        self.units_data = session_data.get('units_data') if session_data else None
        
        print(f"Initialized with bin_size={bin_size_sec*1000:.0f}ms")
        print(f"  Window: {self.window_bins} bins ({self.window_size_sec*1000:.0f}ms)")
        print(f"  Step: {self.step_bins} bins ({self.step_size_sec*1000:.0f}ms)")
        
    def get_unit_id(self, unit_idx):
        """Get the real cluster ID for a unit index."""
        try:
            if self.unit_info is not None:
                if hasattr(self.unit_info, 'iloc'):  # DataFrame
                    return int(self.unit_info['cluster_id'].iloc[unit_idx])
                elif isinstance(self.unit_info, dict):
                    return int(self.unit_info['cluster_id'][unit_idx])
            return unit_idx
        except Exception as e:
            print(f"Warning: Could not get unit_id for unit_idx {unit_idx}: {e}")
            return unit_idx
    
    def get_unit_area(self, unit_idx):
        """Get the brain area for a unit."""
        try:
            if self.units_data is not None:
                if unit_idx in self.units_data.get('MST', []):
                    return 'MST'
                elif unit_idx in self.units_data.get('VPS', []):
                    return 'VPS'
                elif unit_idx in self.units_data.get('dual', []):
                    return 'MT'
            return 'unknown'
        except Exception as e:
            print(f"Warning: Could not get area for unit_idx {unit_idx}: {e}")
            return 'unknown'
    
    def get_unit_depth(self, unit_idx):
        """Get the recording depth for a unit."""
        try:
            if self.unit_info is not None:
                if hasattr(self.unit_info, 'iloc'):  # DataFrame
                    depth = self.unit_info['depth'].iloc[unit_idx]
                elif isinstance(self.unit_info, dict):
                    depth = self.unit_info['depth'][unit_idx]
                else:
                    return 0.0
                
                if isinstance(depth, str):
                    depth = float(depth.strip('[]'))
                
                return float(depth) / 1000.0
            return 0.0
        except Exception as e:
            print(f"Warning: Could not get depth for unit_idx {unit_idx}: {e}")
            return 0.0
    
    def calculate_mean_RT_for_condition(self, behavior_data):
        """
        Calculate mean RT from behavior data for a condition.
        
        Parameters:
        -----------
        behavior_data : dict
            Filtered behavioral data for a condition
        
        Returns:
        --------
        mean_RT : float
            Mean RT in seconds, or None if RT data not available
        """
        if 'RT' not in behavior_data:
            return None
        
        RTs = behavior_data['RT']
        
        # Handle different data types
        if isinstance(RTs, (list, tuple)):
            RTs = np.array(RTs)
        
        # Filter out invalid RTs (NaN, inf, negative, too large)
        valid_mask = np.isfinite(RTs) & (RTs > 0) & (RTs < 10)
        
        if np.sum(valid_mask) == 0:
            return None
        
        mean_RT = np.mean(RTs[valid_mask])
        return mean_RT
    
    def calculate_IFR(self, spikes, original_time_axis, alignment='stimOn'):
        """
        Calculate Instantaneous Firing Rate using sliding window as per paper.
        
        From Gu et al. (2008):
        "Instantaneous FRs... were calculated as the average FR within a 0.2 s 
        rectangular window that was stepped through the data in intervals of 0.1 s."
        
        Modified: Use standard time axis (no RT centering).
        
        Parameters:
        -----------
        spikes : ndarray
            Spike data (neurons x trials x time_bins)
        original_time_axis : ndarray
            Original time axis in seconds
        alignment : str
            Alignment type ('stimOn', 'saccOnset', 'postTargHold')
        
        Returns:
        --------
        IFR : ndarray
            Instantaneous firing rates (neurons x trials x time_windows)
        IFR_time_axis : ndarray
            Time axis for IFR (center of each window) in seconds
        """
        n_neurons, n_trials, n_bins = spikes.shape
        
        window_bins = self.window_bins
        step_bins = self.step_bins
        
        available_bins = n_bins
        n_windows = (available_bins - window_bins) // step_bins + 1
        
        if n_windows <= 0:
            print(f"    Warning: Not enough bins for windowing.")
            print(f"    Available: {available_bins} bins, Need: {window_bins} bins")
            return np.array([]), np.array([])
        
        IFR = np.zeros((n_neurons, n_trials, n_windows))
        IFR_time_axis = np.zeros(n_windows)
        
        valid_windows = 0
        for w in range(n_windows):
            window_start = w * step_bins
            window_end = window_start + window_bins
            
            if window_end > n_bins:
                break
            
            # Average firing rate in window
            IFR[:, :, valid_windows] = np.mean(spikes[:, :, window_start:window_end], axis=2)
            
            # Time index from CENTER of window
            center_bin = window_start + window_bins // 2
            if center_bin < len(original_time_axis):
                IFR_time_axis[valid_windows] = original_time_axis[center_bin]
            else:
                IFR_time_axis[valid_windows] = original_time_axis[0] + center_bin * self.bin_size_sec
            
            valid_windows += 1
        
        # Truncate to valid windows
        IFR = IFR[:, :, :valid_windows]
        IFR_time_axis = IFR_time_axis[:valid_windows]
        
        return IFR, IFR_time_axis
    
    def prepare_data(self, spikes, behavior, valid_units, trial_mod, trial_coh, trial_del=0, 
                    filter_type='all'):
        """
        Prepare data for a specific condition with different filtering options.
        
        Parameters:
        -----------
        spikes : ndarray
            Spike counts (neurons x trials) for filtering
        behavior : dict
            Behavioral data
        valid_units : ndarray or None
            Boolean array or indices of valid units
        trial_mod : int
            Modality (1=vestibular, 2=visual, 3=combined)
        trial_coh : int
            Coherence level
        trial_del : int or array
            Delta value(s)
        filter_type : str
            Type of trial filtering:
            - 'all': no additional filtering
            - 'small_headings': -10° < heading < 10° (all trials)
            - 'correct_small': correct trials + small headings
            - 'error_small': error trials + small headings
            - 'high_pdw_small': high PDW + small headings
            - 'low_pdw_small': low PDW + small headings
        
        Returns:
        --------
        filtered_spikes : ndarray
            Filtered spike data (trials x neurons)
        filtered_behavior : dict
            Filtered behavioral data (includes 'correct')
        trial_indices : ndarray
            Indices of selected trials
        """
        if valid_units is not None:
            spikes = spikes[valid_units, :]
        
        # Handle delta filtering
        if isinstance(trial_del, (list, np.ndarray)):
            del_mask = np.isin(behavior['delta'], trial_del)
        else:
            del_mask = (behavior['delta'] == trial_del)
        
        # Base condition mask
        mask = (behavior['modality'] == trial_mod) & \
            (behavior['coherenceInd'] == trial_coh) & \
            del_mask
        
        # Apply additional filters based on filter_type
        heading = behavior['heading']
        correct = behavior['correct']
        
        if filter_type == 'small_headings':
            # Filter: -10° < heading < 10° (all trials)
            heading_mask = (heading > -10) & (heading < 10)
            mask = mask & heading_mask
            filter_desc = "small headings (-10° < h < 10°), all trials"
            
        elif filter_type == 'correct_small':
            # Filter: correct trials AND small headings
            correct_mask = (correct == 1)
            heading_mask = (heading > -10) & (heading < 10)
            mask = mask & correct_mask & heading_mask
            filter_desc = "correct trials + small headings"
            
        elif filter_type == 'error_small':
            # Filter: error trials AND small headings
            error_mask = (correct == 0)
            heading_mask = (heading > -10) & (heading < 10)
            mask = mask & error_mask & heading_mask
            filter_desc = "error trials + small headings"
            
        elif filter_type == 'high_pdw_small':
            # Filter: high PDW (1) AND small headings
            pdw = behavior['PDW']
            pdw_mask = (pdw == 1)
            heading_mask = (heading > -10) & (heading < 10)
            mask = mask & pdw_mask & heading_mask
            filter_desc = "high PDW + small headings"
            
        elif filter_type == 'low_pdw_small':
            # Filter: low PDW (0) AND small headings
            pdw = behavior['PDW']
            pdw_mask = (pdw == 0)
            heading_mask = (heading > -10) & (heading < 10)
            mask = mask & pdw_mask & heading_mask
            filter_desc = "low PDW + small headings"
            
        else:  # 'all'
            filter_desc = "all trials"
        
        n_trials = np.sum(mask)
        
        if n_trials == 0:
            return np.array([]), {}, np.array([])
        
        # Filter spikes (transpose to trials x neurons)
        filtered_spikes = spikes[:, mask].T
        
        # Filter behavior
        behavior_keys = ['choice', 'PDW', 'modality', 'headingInd', 'coherenceInd', 
                        'goodtrial', 'deltaInd', 'correct', 'oneTargChoice', 
                        'oneTargConf', 'heading', 'coherence', 'delta', 'RT']
        filtered_behavior = {k: behavior[k][mask] for k in behavior_keys if k in behavior}
        
        # Add filter info to behavior dict
        filtered_behavior['filter_type'] = filter_type
        filtered_behavior['filter_description'] = filter_desc
        
        return filtered_spikes, filtered_behavior, np.where(mask)[0]

    def compute_partial_correlation_with_pdw(self, spikes_t, behavior, center=True):
        """
        Compute partial correlations with PDW as nuisance variable.
        
        KEY FIX: Center (demean) predictors and FR across trials before residualization.
        This removes systematic baseline offsets caused by no-intercept regression.
        
        Modified from paper to include PDW:
        - R_heading = R(FR, heading | choice, PDW)
        - R_choice = R(FR, choice | heading, PDW)
        
        Choice is coded as -1 for leftward and +1 for rightward.
        
        Parameters:
        -----------
        spikes_t : ndarray
            Firing rates at one timepoint (trials x neurons)
        behavior : dict
            Must contain 'heading', 'choice', and 'PDW'
        center : bool
            Whether to center variables across trials (recommended: True)
        
        Returns:
        --------
        heading_corrs : ndarray
            Heading partial correlations for each neuron
        heading_pvals : ndarray
            P-values for heading correlations
        choice_corrs : ndarray
            Choice partial correlations for each neuron
        choice_pvals : ndarray
            P-values for choice correlations
        """
        heading = np.asarray(behavior['heading'], dtype=float)
        choice = np.asarray(behavior['choice'], dtype=float)
        pdw = np.asarray(behavior['PDW'], dtype=float)
        
        # Code choice as -1 (leftward) and +1 (rightward)
        choice_coded = 2 * (choice - 1.5)  # Maps 1->-1, 2->+1
        
        # Handle NaNs jointly
        valid_trials = np.isfinite(heading) & np.isfinite(choice_coded) & np.isfinite(pdw)
        if np.sum(valid_trials) < 5:
            n_neurons = spikes_t.shape[1]
            nanv = np.full(n_neurons, np.nan)
            return nanv, nanv, nanv, nanv
        
        heading = heading[valid_trials]
        choice_coded = choice_coded[valid_trials]
        pdw = pdw[valid_trials]
        spikes_t = spikes_t[valid_trials, :]  # trials x neurons
        
        n_trials, n_neurons = spikes_t.shape
        
        # Initialize output arrays
        heading_corrs = np.full(n_neurons, np.nan)
        heading_pvals = np.full(n_neurons, np.nan)
        choice_corrs = np.full(n_neurons, np.nan)
        choice_pvals = np.full(n_neurons, np.nan)
        
        # KEY FIX: Center across trials (removes constant offsets)
        if center:
            heading = heading - np.mean(heading)
            choice_coded = choice_coded - np.mean(choice_coded)
            pdw = pdw - np.mean(pdw)
            spikes_t = spikes_t - np.mean(spikes_t, axis=0, keepdims=True)  # per neuron
        
        # Check for sufficient variance after centering
        if np.std(heading) == 0 or np.std(choice_coded) == 0:
            return heading_corrs, heading_pvals, choice_corrs, choice_pvals
        
        # === Heading Partial Correlation: R(FR, heading | choice, PDW) ===
        X_nuisance_heading = np.column_stack([choice_coded, pdw])
        
        # Residualize heading w.r.t. choice and PDW
        if np.linalg.matrix_rank(X_nuisance_heading) >= 1:
            try:
                beta_heading = np.linalg.lstsq(X_nuisance_heading, heading, rcond=None)[0]
                heading_resid = heading - X_nuisance_heading @ beta_heading
            except np.linalg.LinAlgError:
                heading_resid = heading
        else:
            heading_resid = heading
        
        # Fallback if no variance in residual
        if np.std(heading_resid) == 0:
            heading_resid = heading
        
        # === Choice Partial Correlation: R(FR, choice | heading, PDW) ===
        X_nuisance_choice = np.column_stack([heading, pdw])
        
        # Residualize choice w.r.t. heading and PDW
        if np.linalg.matrix_rank(X_nuisance_choice) >= 1:
            try:
                beta_choice = np.linalg.lstsq(X_nuisance_choice, choice_coded, rcond=None)[0]
                choice_resid = choice_coded - X_nuisance_choice @ beta_choice
            except np.linalg.LinAlgError:
                choice_resid = choice_coded
        else:
            choice_resid = choice_coded
        
        # Fallback if no variance in residual
        if np.std(choice_resid) == 0:
            choice_resid = choice_coded
        
        # For each neuron, compute partial correlations
        for n in range(n_neurons):
            FR = spikes_t[:, n]
            
            # Skip if no variance
            if np.std(FR) == 0:
                continue
            
            # === Heading partial ===
            if np.linalg.matrix_rank(X_nuisance_heading) >= 1:
                try:
                    beta_FR_h = np.linalg.lstsq(X_nuisance_heading, FR, rcond=None)[0]
                    FR_resid_h = FR - X_nuisance_heading @ beta_FR_h
                except np.linalg.LinAlgError:
                    FR_resid_h = FR
            else:
                FR_resid_h = FR
            
            # Correlate residuals
            if np.std(heading_resid) > 0 and np.std(FR_resid_h) > 0:
                try:
                    r, p = pearsonr(heading_resid, FR_resid_h)
                    heading_corrs[n] = r
                    heading_pvals[n] = p
                except:
                    pass
            
            # === Choice partial ===
            if np.linalg.matrix_rank(X_nuisance_choice) >= 1:
                try:
                    beta_FR_c = np.linalg.lstsq(X_nuisance_choice, FR, rcond=None)[0]
                    FR_resid_c = FR - X_nuisance_choice @ beta_FR_c
                except np.linalg.LinAlgError:
                    FR_resid_c = FR
            else:
                FR_resid_c = FR
            
            # Correlate residuals
            if np.std(choice_resid) > 0 and np.std(FR_resid_c) > 0:
                try:
                    r, p = pearsonr(choice_resid, FR_resid_c)
                    choice_corrs[n] = r
                    choice_pvals[n] = p
                except:
                    pass
        
        return heading_corrs, heading_pvals, choice_corrs, choice_pvals

    def run_partial_correlation_analysis(self, spikes_data, behavior_data, time_axes, area, 
                                        valid_units=None, conditions=None, 
                                        filter_types=['small_headings', 'errors_zeros', 
                                                     'high_pdw_small', 'low_pdw_small'],
                                        save_results=True, verbose=True):
        """
        Run partial correlation analysis with multiple trial filtering options.
        
        Parameters:
        -----------
        spikes_data : dict
            Dictionary with keys ['stimOn', 'saccOnset', 'postTargHold']
            Each contains spike counts (neurons x trials x time_bins)
        behavior_data : dict
            Behavioral data dictionary
        time_axes : dict
            Original time axes for each alignment (in seconds)
        area : str
            Brain area name
        valid_units : ndarray or None
            Boolean array or indices of valid units
        conditions : list of tuples or None
            List of (modality, coherence) tuples
        filter_types : list of str
            List of filter types to apply:
            - 'small_headings': -10° < heading < 10°
            - 'errors_zeros': error trials + zero heading trials
            - 'high_pdw_small': high PDW + small headings
            - 'low_pdw_small': low PDW + small headings
        save_results : bool
            Whether to save results
        verbose : bool
            Whether to print progress
        
        Returns:
        --------
        all_results_dict : dict
            Dictionary with results for each filter type
        """
        
        if conditions is None:
            conditions = [
                (1, 1),  # vestibular
                (2, 1),  # visual low coh
                (2, 2),  # visual high coh
                (3, 1),  # combined low coh
                (3, 2),  # combined high coh
            ]
        
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
        
        # Create unit metadata (shared across all filter types)
        unit_metadata = []
        for idx in sel_idx:
            unit_meta = {
                'unit_idx': int(idx),
                'unit_id': self.get_unit_id(idx),
                'area': self.get_unit_area(idx),
                'depth_mm': self.get_unit_depth(idx)
            }
            unit_metadata.append(unit_meta)
        
        unit_metadata_df = pd.DataFrame(unit_metadata)
        
        # Store results for each filter type
        all_results_dict = {}
        
        # Loop over filter types
        for filter_type in filter_types:
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"FILTER TYPE: {filter_type}")
                print(f"{'='*80}")
                print(f"Partial Correlation Analysis (Trial-wise Centered): {area}")
                print(f"Subject: {self.subject}, Date: {self.date}")
                print(f"Bin size: {self.bin_size_sec*1000:.0f}ms")
                print(f"Window: {self.window_size_sec*1000:.0f}ms ({self.window_bins} bins)")
                print(f"Step: {self.step_size_sec*1000:.0f}ms ({self.step_bins} bins)")
                print(f"Choice coding: -1 (leftward), +1 (rightward)")
                print(f"Conditions: {conditions}")
                print(f"{'='*80}")
            
            all_results = {}
            all_results['unit_metadata'] = unit_metadata_df
            all_results['filter_type'] = filter_type
            all_results['method_params'] = {
                'window_size_sec': self.window_size_sec,
                'step_size_sec': self.step_size_sec,
                'bin_size_sec': self.bin_size_sec,
                'window_bins': self.window_bins,
                'step_bins': self.step_bins,
                'reference': 'Gu, Angelaki & DeAngelis (2008); Zaidel, DeAngelis & Angelaki (2017)',
                'modifications': [
                    'Include PDW as nuisance variable',
                    'Center (demean) predictors and FR across trials before residualization',
                    'stimOn alignment is stimulus-locked (no RT centering)',
                    'Data pre-binned at 20ms',
                    f'Trial filtering: {filter_type}'
                ]
            }
            
            # Condition key mapping
            condition_to_key = {
                (1, 1): 'mod1_coh1',
                (2, 1): 'mod2_coh1',
                (2, 2): 'mod2_coh2',
                (3, 1): 'mod3_coh1',
                (3, 2): 'mod3_coh2'
            }
            
            # Loop over alignments
            for alignment in ['stimOn', 'saccOnset', 'postTargHold']:
                if alignment not in spikes_data:
                    continue
                    
                if verbose:
                    print(f"\n{'='*40}\n{alignment}\n{'='*40}")
                
                spikes = spikes_data[alignment]
                original_time_axis = time_axes.get(alignment, np.arange(spikes.shape[2]) * self.bin_size_sec)
                
                if verbose:
                    print(f"  Original data:")
                    print(f"    Bins: {spikes.shape[2]}, time range: [{original_time_axis[0]:.3f}, {original_time_axis[-1]:.3f}]s")
                
                # Apply unit selection
                if final_units is not None:
                    spikes_sel = spikes[final_units, :, :]
                else:
                    spikes_sel = spikes
                
                alignment_results = {
                    'original_time_axis': original_time_axis,
                    'n_units': n_units,
                    'unit_indices': sel_idx,
                    'unit_ids': [self.get_unit_id(idx) for idx in sel_idx],
                    'unit_areas': [self.get_unit_area(idx) for idx in sel_idx],
                    'unit_depths': [self.get_unit_depth(idx) for idx in sel_idx],
                    'conditions': {}
                }
                
                # Loop over conditions
                for mod, coh in conditions:
                    condition_key = condition_to_key[(mod, coh)]
                    
                    if verbose:
                        print(f"\n  Condition: {condition_key}")
                    
                    # Filter trials for this condition WITH FILTER TYPE
                    spikes_t0 = spikes[:, :, 0]
                    _, behavior_cond, trial_indices = self.prepare_data(
                        spikes_t0, behavior_data, final_units, mod, coh, trial_del=0,
                        filter_type=filter_type
                    )
                    
                    if len(trial_indices) == 0:
                        if verbose:
                            print(f"    No trials for this condition with filter: {filter_type}")
                        continue
                    
                    if verbose:
                        print(f"    Filter: {behavior_cond.get('filter_description', 'N/A')}")
                        print(f"    Trials: {len(trial_indices)}")
                        
                        # Print heading-choice correlation to check collinearity
                        heading_cond = behavior_cond['heading']
                        choice_cond = 2 * (behavior_cond['choice'] - 1.5)
                        valid_hc = np.isfinite(heading_cond) & np.isfinite(choice_cond)
                        if np.sum(valid_hc) > 5:
                            r_hc, _ = pearsonr(heading_cond[valid_hc], choice_cond[valid_hc])
                            print(f"    Heading-Choice correlation: r = {r_hc:.3f}")
                    
                    # Calculate mean RT for metadata
                    mean_RT_sec = self.calculate_mean_RT_for_condition(behavior_cond)
                    if verbose and mean_RT_sec is not None:
                        print(f"    Mean RT: {mean_RT_sec*1000:.1f} ms")
                    
                    # Calculate IFR for this condition
                    if verbose:
                        print(f"    Calculating IFR...")
                    
                    # Filter spikes for this condition
                    spikes_cond = spikes_sel[:, trial_indices, :]
                    
                    # Calculate IFR
                    IFR_cond, IFR_time_axis = self.calculate_IFR(
                        spikes_cond, original_time_axis, alignment
                    )
                    
                    if len(IFR_time_axis) == 0:
                        if verbose:
                            print(f"    Skipping: insufficient data for IFR calculation")
                        continue
                    
                    if verbose:
                        print(f"    IFR windows: {len(IFR_time_axis)}")
                        print(f"    Time range: [{IFR_time_axis[0]:.3f}, {IFR_time_axis[-1]:.3f}]s")
                    
                    # Initialize results storage
                    condition_results = {
                        'modality': mod,
                        'coherence': coh,
                        'filter_type': filter_type,
                        'filter_description': behavior_cond.get('filter_description', ''),
                        'mean_RT_sec': mean_RT_sec,
                        'time_axis': IFR_time_axis,
                        'time_axis_description': f'{alignment} alignment (stimulus-locked)' if alignment=='stimOn' else f'{alignment} alignment',
                        'heading_corrs': [],
                        'heading_pvals': [],
                        'heading_mean': [],
                        'heading_median': [],
                        'heading_std': [],
                        'heading_n_sig': [],
                        'choice_corrs': [],
                        'choice_pvals': [],
                        'choice_mean': [],
                        'choice_median': [],
                        'choice_std': [],
                        'choice_n_sig': [],
                        'n_trials': []
                    }
                    
                    # Compute partial correlations at each time window
                    n_windows = IFR_cond.shape[2]
                    
                    if verbose:
                        print(f"    Computing partial correlations (centered, with PDW)...")
                    
                    for t in range(n_windows):
                        if verbose and (t % 10 == 0 or t == n_windows - 1):
                            print(f"    Window {t+1}/{n_windows} (t={IFR_time_axis[t]:.3f}s)")
                        
                        # Get IFR at this time window: trials x neurons
                        IFR_t = IFR_cond[:, :, t].T
                        
                        # Compute partial correlations
                        h_corrs, h_pvals, c_corrs, c_pvals = \
                            self.compute_partial_correlation_with_pdw(IFR_t, behavior_cond, center=True)
                        
                        # Store results
                        condition_results['heading_corrs'].append(h_corrs)
                        condition_results['heading_pvals'].append(h_pvals)
                        condition_results['heading_mean'].append(np.nanmean(h_corrs))
                        condition_results['heading_median'].append(np.nanmedian(h_corrs))
                        condition_results['heading_std'].append(np.nanstd(h_corrs))
                        condition_results['heading_n_sig'].append(np.sum(h_pvals < 0.05))
                        
                        condition_results['choice_corrs'].append(c_corrs)
                        condition_results['choice_pvals'].append(c_pvals)
                        condition_results['choice_mean'].append(np.nanmean(c_corrs))
                        condition_results['choice_median'].append(np.nanmedian(c_corrs))
                        condition_results['choice_std'].append(np.nanstd(c_corrs))
                        condition_results['choice_n_sig'].append(np.sum(c_pvals < 0.05))
                        
                        condition_results['n_trials'].append(IFR_t.shape[0])
                    
                    # Convert lists to arrays
                    for key in ['heading_corrs', 'heading_pvals', 'heading_mean', 'heading_median',
                               'heading_std', 'heading_n_sig',
                               'choice_corrs', 'choice_pvals', 'choice_mean', 'choice_median',
                               'choice_std', 'choice_n_sig', 'n_trials']:
                        condition_results[key] = np.array(condition_results[key])
                    
                    if verbose:
                        valid_times = ~np.isnan(condition_results['heading_mean'])
                        if np.any(valid_times):
                            print(f"\n    Results Summary:")
                            print(f"    Heading partial correlation R(FR, heading | choice, PDW):")
                            print(f"      Mean: {np.nanmean(condition_results['heading_mean']):.4f}")
                            print(f"      Range: [{np.nanmin(condition_results['heading_mean']):.4f}, "
                                  f"{np.nanmax(condition_results['heading_mean']):.4f}]")
                            print(f"      Mean sig neurons: {np.mean(condition_results['heading_n_sig'][valid_times]):.1f}")
                            
                            print(f"    Choice partial correlation R(FR, choice | heading, PDW):")
                            print(f"      Mean: {np.nanmean(condition_results['choice_mean']):.4f}")
                            print(f"      Range: [{np.nanmin(condition_results['choice_mean']):.4f}, "
                                  f"{np.nanmax(condition_results['choice_mean']):.4f}]")
                            print(f"      Mean sig neurons: {np.mean(condition_results['choice_n_sig'][valid_times]):.1f}")
                    
                    alignment_results['conditions'][condition_key] = condition_results
                
                all_results[alignment] = alignment_results
            
            # Save results for this filter type
            if save_results:
                self.save_analysis_results(all_results, area, suffix=filter_type)
            
            # Store in dict
            all_results_dict[filter_type] = all_results
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"Filter type '{filter_type}' completed!")
                print(f"{'='*80}\n")
        
        if verbose:
            print(f"\n{'='*80}")
            print("ALL ANALYSES COMPLETED!")
            print(f"Saved {len(filter_types)} different filtered datasets")
            print(f"{'='*80}\n")
        
        return all_results_dict
    
    def save_analysis_results(self, results, area, suffix=''):
        """Save analysis results to disk with optional suffix."""
        # Add suffix to filename
        suffix_str = f"_{suffix}" if suffix else ""
        filename = f"{self.subject}_{self.date}_{area}_partialcorr_centered{suffix_str}.pkl"
        filepath = self.save_dir / filename
        
        filter_type = results.get('filter_type', 'unknown')
        
        results['metadata'] = {
            'subject': self.subject,
            'date': self.date,
            'area': area,
            'analysis': 'partial_correlation_centered',
            'filter_type': filter_type,
            'description': f'Partial correlation with PDW, trial-wise centering, filter: {filter_type}',
            'reference': 'Gu, Angelaki & DeAngelis (2008); Zaidel, DeAngelis & Angelaki (2017)',
            'modifications': [
                'Include PDW as nuisance variable',
                'Center (demean) predictors and FR across trials before residualization',
                'stimOn alignment is stimulus-locked (t=0 = stimulus onset)',
                'Heading partial: R(FR, heading | choice, PDW)',
                'Choice partial: R(FR, choice | heading, PDW)',
                'Data pre-binned at 20ms',
                f'Trial filtering: {filter_type}'
            ],
            'window_size_sec': self.window_size_sec,
            'step_size_sec': self.step_size_sec,
            'bin_size_sec': self.bin_size_sec,
            'window_bins': self.window_bins,
            'step_bins': self.step_bins,
            'choice_coding': '-1 (leftward), +1 (rightward)'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to: {filepath}")
        
        # Save unit metadata
        if 'unit_metadata' in results:
            metadata_filename = f"{self.subject}_{self.date}_{area}_partialcorr_centered{suffix_str}_units.csv"
            metadata_filepath = self.save_dir / metadata_filename
            results['unit_metadata'].to_csv(metadata_filepath, index=False)
            print(f"Unit metadata saved to: {metadata_filepath}")
        
        # Save summary
        summary_filename = f"{self.subject}_{self.date}_{area}_partialcorr_centered{suffix_str}_summary.txt"
        summary_filepath = self.save_dir / summary_filename
        
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Partial Correlation Analysis Summary\n")
            f.write(f"{'='*60}\n")
            f.write(f"Filter Type: {filter_type}\n")
            f.write(f"Based on: Gu, Angelaki & DeAngelis (2008)\n")
            f.write(f"Reference: Zaidel, DeAngelis & Angelaki (2017)\n")
            f.write(f"Subject: {self.subject}\n")
            f.write(f"Date: {self.date}\n")
            f.write(f"Area: {area}\n\n")
            
            f.write(f"Trial Filtering:\n")
            if filter_type == 'small_headings':
                f.write(f"  -10° < heading < 10° (all trials)\n")
            elif filter_type == 'errors_zeros':
                f.write(f"  Error trials + zero heading trials\n")
            elif filter_type == 'high_pdw_small':
                f.write(f"  High PDW (2) + small headings (-10° < h < 10°)\n")
            elif filter_type == 'low_pdw_small':
                f.write(f"  Low PDW (1) + small headings (-10° < h < 10°)\n")
            f.write(f"\n")
            
            f.write(f"Method Parameters:\n")
            f.write(f"  Bin size: {self.bin_size_sec*1000:.0f} ms\n")
            f.write(f"  Window size: {self.window_size_sec*1000:.0f} ms ({self.window_bins} bins)\n")
            f.write(f"  Step size: {self.step_size_sec*1000:.0f} ms ({self.step_bins} bins)\n")
            f.write(f"  Choice coding: -1 (leftward), +1 (rightward)\n\n")
            
            f.write(f"Modifications:\n")
            f.write(f"  1. Include PDW as nuisance variable\n")
            f.write(f"  2. Center (demean) predictors and FR across trials\n")
            f.write(f"  3. stimOn alignment is stimulus-locked\n")
            f.write(f"  4. Trial filtering: {filter_type}\n\n")
            
            if 'unit_metadata' in results:
                f.write(f"Units:\n")
                f.write(f"  Total: {len(results['unit_metadata'])}\n")
                for area_name, count in results['unit_metadata']['area'].value_counts().items():
                    f.write(f"  {area_name}: {count}\n")
                f.write(f"\n")
            
            for alignment, align_data in results.items():
                if alignment in ['metadata', 'unit_metadata', 'method_params', 'filter_type']:
                    continue
                    
                f.write(f"\n{'='*60}\n")
                f.write(f"{alignment}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Number of units: {align_data['n_units']}\n\n")
                
                for cond_key, cond_data in align_data['conditions'].items():
                    f.write(f"  {cond_key}:\n")
                    f.write(f"    Filter: {cond_data.get('filter_description', 'N/A')}\n")
                    f.write(f"    Trials: {cond_data['n_trials'][0]}\n")
                    
                    if cond_data['mean_RT_sec'] is not None:
                        f.write(f"    Mean RT: {cond_data['mean_RT_sec']*1000:.1f} ms\n")
                    
                    f.write(f"    Time windows: {len(cond_data['time_axis'])}\n")
                    f.write(f"    Time range: [{cond_data['time_axis'][0]:.3f}, {cond_data['time_axis'][-1]:.3f}]s\n")
                    
                    f.write(f"\n    Heading partial correlation:\n")
                    f.write(f"      Mean: {np.nanmean(cond_data['heading_mean']):.4f} ± "
                           f"{np.nanstd(cond_data['heading_mean']):.4f}\n")
                    f.write(f"      Range: [{np.nanmin(cond_data['heading_mean']):.4f}, "
                           f"{np.nanmax(cond_data['heading_mean']):.4f}]\n")
                    
                    f.write(f"\n    Choice partial correlation:\n")
                    f.write(f"      Mean: {np.nanmean(cond_data['choice_mean']):.4f} ± "
                           f"{np.nanstd(cond_data['choice_mean']):.4f}\n")
                    f.write(f"      Range: [{np.nanmin(cond_data['choice_mean']):.4f}, "
                           f"{np.nanmax(cond_data['choice_mean']):.4f}]\n")
                    
                    f.write("\n")
        
        print(f"Summary saved to: {summary_filepath}")
    
    @staticmethod
    def load_results(filepath):
        """Load saved partial correlation results."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    print("="*60)
    print("Partial Correlation Analyzer - Multiple Trial Filters")
    print("Based on: Gu, Angelaki & DeAngelis (2008)")
    print("="*60)
    print("\nTrial Filtering Options:")
    print("  1. small_headings: -10° < heading < 10°")
    print("  2. errors_zeros: error trials + zero heading trials")
    print("  3. high_pdw_small: high PDW + small headings")
    print("  4. low_pdw_small: low PDW + small headings")
    print("\nThis will generate 4 separate output files")
    print("="*60)