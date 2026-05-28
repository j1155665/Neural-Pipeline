import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class InterAreaCorrelation:
    
    def __init__(self, subject, date, save_dir=r'D:\Neural-Pipeline\results\analysis_population\area_correlation'):
        self.subject = subject
        self.date = date
        self.base_save_dir = Path(save_dir)
        
        # Create hierarchical structure: subject/date/
        self.save_dir = self.base_save_dir / subject / date
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Save directory: {self.save_dir}")
    
    def calculate_correlation(self, spikes_data, behavior_data, time_axes, units_data,
                            alignment='stimOn', 
                            task_type='regular',
                            trial_mod=None,
                            trial_coh=None,
                            trial_del=0,
                            trial_heading=None,
                            trial_pdw=None,
                            trial_correct=None,
                            time_lags_ms=range(-80, 81, 10),
                            interpolation_factor=1,
                            zscore_trials=True,
                            save_results=True):
        """
        Calculate time-resolved correlation between MST and VPS with time lags
        
        At each reference time t, correlates:
        - MST activity at time t across trials
        - VPS activity at time (t + lag) across trials
        
        X-axis represents the reference time t (when MST is measured)
        Y-axis represents the lag applied to VPS
        
        Parameters:
        -----------
        spikes_data : dict
            Dictionary with alignment keys, each containing spike data
        behavior_data : dict
            Dictionary containing behavioral variables
        time_axes : dict
            Dictionary with alignment keys, each containing time axis
        units_data : dict
            Dictionary with 'MST' and 'VPS' keys containing unit indices
        alignment : str, default='stimOn'
            Which alignment to use ('stimOn', 'saccOnset', 'postTargHold')
        task_type : str, default='regular'
            Task type identifier
        trial_mod : int or None
            Modality filter (1=vest, 2=vis, 3=comb, None=all)
        trial_coh : int or None
            Coherence filter (1=low, 2=high, None=all)
        trial_del : int or None
            Delta filter (0=no delay, None=all)
        trial_heading : int or list or None
            Heading index filter
        trial_pdw : int or None
            PDW filter (0=low bet, 1=high bet, None=all)
        trial_correct : int or None
            Correctness filter (0=error, 1=correct, None=all)
        time_lags_ms : range or list
            Time lags to test in milliseconds
        interpolation_factor : int, default=1
            Upsampling factor for temporal interpolation
        zscore_trials : bool, default=True
            If True, z-score each trial independently to remove trial-level baseline.
            Recommended: True for temporal dynamics, False for trial covariation.
        save_results : bool, default=True
            Whether to save results to disk
        
        Returns:
        --------
        results : dict
            Dictionary containing correlation matrices, metadata, and parameters
        """
        
        print(f"\n{'='*60}")
        print(f"CALCULATING INTER-AREA CORRELATION")
        print(f"Subject: {self.subject}, Date: {self.date}")
        print(f"Task: {task_type}, Alignment: {alignment}")
        print(f"Z-score trials: {zscore_trials}")
        print(f"Conditions: mod={trial_mod}, coh={trial_coh}, del={trial_del}, "
            f"heading={trial_heading}, pdw={trial_pdw}, correct={trial_correct}")
        if interpolation_factor > 1:
            print(f"Interpolation: {interpolation_factor}x upsampling")
        print(f"{'='*60}")
        
        # ========================================================================
        # 1. VALIDATE ALIGNMENT
        # ========================================================================
        if alignment not in spikes_data:
            available = list(spikes_data.keys())
            raise ValueError(f"Alignment '{alignment}' not found. Available: {available}")
        
        spikes = spikes_data[alignment]
        time_axis = time_axes[alignment]
        
        # ========================================================================
        # 2. GET UNIT INDICES
        # ========================================================================
        MST_units = units_data.get('MST', [])
        VPS_units = units_data.get('VPS', [])
        
        if len(MST_units) == 0 or len(VPS_units) == 0:
            print(f"Error: Insufficient units. MST: {len(MST_units)}, VPS: {len(VPS_units)}")
            return None
        
        print(f"Units: MST={len(MST_units)}, VPS={len(VPS_units)}")
        
        # ========================================================================
        # 3. BUILD TRIAL MASK
        # ========================================================================
        trial_mask = self._build_trial_mask(
            behavior_data, trial_mod, trial_coh, trial_del,
            trial_heading, trial_pdw, trial_correct
        )
        
        n_trials_used = np.sum(trial_mask)
        print(f"Trials after filtering: {n_trials_used}")
        
        if n_trials_used < 5:
            print("Warning: Too few trials for correlation analysis")
            return None
        
        # ========================================================================
        # 4. GET POPULATION FIRING RATES (WITHOUT Z-SCORING YET)
        # ========================================================================
        MST_pop = self._get_population_firing_rate(spikes, trial_mask, MST_units, 
                                                    zscore_trials=False)
        VPS_pop = self._get_population_firing_rate(spikes, trial_mask, VPS_units,
                                                    zscore_trials=False)
        
        if MST_pop is None or VPS_pop is None:
            print("Error: Failed to calculate population firing rates")
            return None
        
        print(f"Original data shape: MST={MST_pop.shape}, VPS={VPS_pop.shape}")
        
        # ========================================================================
        # 5. CHECK FOR LENGTH MISMATCH
        # ========================================================================
        if len(time_axis) != MST_pop.shape[1]:
            print(f"⚠ WARNING: Length mismatch detected!")
            print(f"  time_axis: {len(time_axis)}, MST time: {MST_pop.shape[1]}, VPS time: {VPS_pop.shape[1]}")
            min_len = min(len(time_axis), MST_pop.shape[1], VPS_pop.shape[1])
            print(f"  Trimming all to length: {min_len}")
            time_axis = time_axis[:min_len]
            MST_pop = MST_pop[:, :min_len]
            VPS_pop = VPS_pop[:, :min_len]
        
        # ========================================================================
        # 6. CALCULATE ORIGINAL BIN SIZE
        # ========================================================================
        original_bin_size_ms = np.mean(np.diff(time_axis)) * 1000
        print(f"Original bin size: {original_bin_size_ms:.2f} ms")
        
        # ========================================================================
        # 7. INTERPOLATE DATA IF REQUESTED
        # ========================================================================
        if interpolation_factor > 1:
            print(f"\n{'─'*60}")
            print(f"INTERPOLATING DATA: {interpolation_factor}x upsampling")
            print(f"{'─'*60}")
            
            MST_pop, VPS_pop, time_axis = self._interpolate_population_data(
                MST_pop, VPS_pop, time_axis, interpolation_factor
            )
            
            print(f"Interpolated data shape: MST={MST_pop.shape}, VPS={VPS_pop.shape}")
            
            effective_bin_size_ms = np.mean(np.diff(time_axis)) * 1000
            print(f"Effective bin size after interpolation: {effective_bin_size_ms:.2f} ms")
            print(f"{'─'*60}\n")
        else:
            effective_bin_size_ms = original_bin_size_ms
        
        # ========================================================================
        # 8. Z-SCORE TRIALS IF REQUESTED
        # ========================================================================
        if zscore_trials:
            print(f"\n{'─'*60}")
            print(f"Z-SCORING TRIALS")
            print(f"{'─'*60}")
            
            # Z-score MST
            mst_trial_means = np.mean(MST_pop, axis=1, keepdims=True)  # (n_trials, 1)
            mst_trial_stds = np.std(MST_pop, axis=1, keepdims=True)    # (n_trials, 1)
            mst_trial_stds[mst_trial_stds < 1e-10] = 1.0  # Avoid division by zero
            MST_pop = (MST_pop - mst_trial_means) / mst_trial_stds
            
            # Z-score VPS
            vps_trial_means = np.mean(VPS_pop, axis=1, keepdims=True)
            vps_trial_stds = np.std(VPS_pop, axis=1, keepdims=True)
            vps_trial_stds[vps_trial_stds < 1e-10] = 1.0
            VPS_pop = (VPS_pop - vps_trial_means) / vps_trial_stds
            
            print(f"✓ Z-scored {MST_pop.shape[0]} trials for MST")
            print(f"✓ Z-scored {VPS_pop.shape[0]} trials for VPS")
            print(f"  Each trial normalized: (rate - trial_mean) / trial_std")
            print(f"  This removes trial-level baseline differences")
            print(f"  Focus: Temporal dynamics within trials")
            print(f"{'─'*60}\n")
        else:
            print(f"\n{'─'*60}")
            print(f"NO Z-SCORING APPLIED")
            print(f"{'─'*60}")
            print(f"Using raw firing rates")
            print(f"Includes both trial-level baseline AND temporal dynamics")
            print(f"{'─'*60}\n")
        
        # ========================================================================
        # 9. CONVERT TIME LAGS TO BINS
        # ========================================================================
        time_lags_bins = [int(np.round(lag_ms / effective_bin_size_ms)) for lag_ms in time_lags_ms]
        
        print(f"Time lag settings:")
        print(f"  Range: {min(time_lags_ms)} to {max(time_lags_ms)} ms")
        print(f"  Step: {time_lags_ms[1] - time_lags_ms[0] if len(time_lags_ms) > 1 else 0} ms")
        print(f"  Number of lags: {len(time_lags_ms)}")
        print(f"  Lag bins: {min(time_lags_bins)} to {max(time_lags_bins)} bins")
        
        # ========================================================================
        # 10. INITIALIZE CORRELATION MATRICES
        # ========================================================================
        n_lags = len(time_lags_bins)
        n_time = MST_pop.shape[1]
        correlation_matrix = np.full((n_lags, n_time), np.nan)
        p_value_matrix = np.full((n_lags, n_time), np.nan)
        
        print(f"\nComputing correlations: {n_lags} lags × {n_time} timepoints...")
        print(f"Correlating across {n_trials_used} trials at each (lag, time) point")
        print(f"X-axis: Reference time (MST measurement time)")
        print(f"Y-axis: VPS lag relative to MST")
        
        # ========================================================================
        # 11. CALCULATE CORRELATIONS FOR EACH LAG
        # ========================================================================
        for lag_idx, lag_bins in enumerate(time_lags_bins):
            
            lag_ms = time_lags_ms[lag_idx]
            
            # For each REFERENCE time point (MST time)
            for t_ref in range(n_time):
                
                t_mst = t_ref
                t_vps = t_ref + lag_bins  # VPS time with lag applied
                
                # Check if VPS time point is within bounds
                if t_vps < 0 or t_vps >= n_time:
                    continue
                
                try:
                    # Get firing rates across trials at specific timepoints
                    mst_rates = MST_pop[:, t_mst]  # shape: (n_trials,)
                    vps_rates = VPS_pop[:, t_vps]  # shape: (n_trials,)
                    
                    # Skip if not enough variance
                    if np.std(mst_rates) < 1e-10 or np.std(vps_rates) < 1e-10:
                        continue
                    
                    # Calculate Pearson correlation across trials
                    r, p = pearsonr(mst_rates, vps_rates) 
    
                    
                    # Store at reference time (no shift!)
                    correlation_matrix[lag_idx, t_ref] = r
                    p_value_matrix[lag_idx, t_ref] = p
                    
                except Exception as e:
                    continue
            
            # Progress update
            if (lag_idx + 1) % 5 == 0 or lag_idx == 0:
                n_valid = np.sum(~np.isnan(correlation_matrix[lag_idx, :]))
                print(f"  ✓ Processed lag {lag_idx + 1}/{n_lags}: {lag_ms:+4d} ms ({lag_bins:+3d} bins) - {n_valid} valid timepoints")
        
        # ========================================================================
        # 12. CHECK VALIDITY OF RESULTS
        # ========================================================================
        n_valid = np.sum(~np.isnan(correlation_matrix))
        n_total = correlation_matrix.size
        print(f"\nValid correlations: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
        
        if n_valid == 0:
            print("⚠ Warning: No valid correlations computed!")
            return None
        
        # ========================================================================
        # 13. CREATE CONDITION LABEL
        # ========================================================================
        condition_str = self._create_condition_string(
            trial_mod, trial_coh, trial_del, trial_heading,
            trial_pdw, trial_correct
        )
        
        # ========================================================================
        # 14. PREPARE RESULTS DICTIONARY
        # ========================================================================
        results = {
            'correlation_matrix': correlation_matrix,
            'p_value_matrix': p_value_matrix,
            'time_axis': time_axis,
            'time_lags_ms': list(time_lags_ms),
            'time_lags_bins': time_lags_bins,
            'metadata': {
                'subject': self.subject,
                'date': self.date,
                'task_type': task_type,
                'alignment': alignment,
                'trial_mod': trial_mod,
                'trial_coh': trial_coh,
                'trial_del': trial_del,
                'trial_heading': trial_heading,
                'trial_pdw': trial_pdw,
                'trial_correct': trial_correct,
                'condition_string': condition_str,
                'n_trials_used': int(n_trials_used),
                'n_MST_units': len(MST_units),
                'n_VPS_units': len(VPS_units),
                'MST_units': MST_units.tolist() if hasattr(MST_units, 'tolist') else list(MST_units),
                'VPS_units': VPS_units.tolist() if hasattr(VPS_units, 'tolist') else list(VPS_units),
                'original_bin_size_ms': float(original_bin_size_ms),
                'effective_bin_size_ms': float(effective_bin_size_ms),
                'interpolation_factor': int(interpolation_factor),
                'interpolation_used': bool(interpolation_factor > 1),
                'zscore_trials': bool(zscore_trials)
            }
        }
        
        # ========================================================================
        # 15. SAVE RESULTS
        # ========================================================================
        if save_results:
            self._save_results(results, task_type, alignment, condition_str)
        
        print("✓ Correlation analysis complete!")
        return results


    def _interpolate_population_data(self, MST_pop, VPS_pop, time_axis, factor):
        """
        Interpolate population data (trial-by-trial) to simulate finer temporal resolution
        
        Parameters:
        -----------
        MST_pop : array (n_trials, n_time)
            MST population firing rates per trial
        VPS_pop : array (n_trials, n_time)
            VPS population firing rates per trial
        time_axis : array (n_time,)
            Original time points
        factor : int
            Upsampling factor
        
        Returns:
        --------
        MST_interp : array (n_trials, n_time * factor)
        VPS_interp : array (n_trials, n_time * factor)
        time_interp : array (n_time * factor,)
        """
        
        n_trials_mst, n_time = MST_pop.shape
        n_trials_vps = VPS_pop.shape[0]
        
        # Create new time axis with finer resolution
        n_time_new = (n_time - 1) * factor + 1
        time_interp = np.linspace(time_axis[0], time_axis[-1], n_time_new)
        
        print(f"  Original timepoints: {n_time} (spanning {time_axis[0]:.3f} to {time_axis[-1]:.3f} s)")
        print(f"  New timepoints: {n_time_new} (same span, {factor}x denser)")
        print(f"  Creating {n_time_new - n_time} synthetic timepoints via cubic interpolation")
        
        # Initialize interpolated arrays
        MST_interp = np.zeros((n_trials_mst, n_time_new))
        VPS_interp = np.zeros((n_trials_vps, n_time_new))
        
        # Interpolate each trial separately for MST
        print(f"  Interpolating MST data ({n_trials_mst} trials)...", end='')
        for trial_idx in range(n_trials_mst):
            f = interp1d(time_axis, MST_pop[trial_idx, :], 
                        kind='cubic',
                        fill_value='extrapolate',
                        bounds_error=False)
            MST_interp[trial_idx, :] = f(time_interp)
        print(" Done")
        
        # Interpolate each trial separately for VPS
        print(f"  Interpolating VPS data ({n_trials_vps} trials)...", end='')
        for trial_idx in range(n_trials_vps):
            f = interp1d(time_axis, VPS_pop[trial_idx, :], 
                        kind='cubic',
                        fill_value='extrapolate',
                        bounds_error=False)
            VPS_interp[trial_idx, :] = f(time_interp)
        print(" Done")
        
        return MST_interp, VPS_interp, time_interp
    
    def _build_trial_mask(self, behavior, trial_mod, trial_coh, trial_del,
                         trial_heading, trial_pdw, trial_correct):
        """Build boolean mask for trial selection - flexible filtering"""
        
        n_trials = len(behavior['modality'])
        mask = np.ones(n_trials, dtype=bool)
        
        # Modality filter
        if trial_mod is not None:
            mask = mask & (behavior['modality'] == trial_mod)
        
        # Coherence filter
        if trial_coh is not None:
            mask = mask & (behavior['coherenceInd'] == trial_coh)
        
        # Delta filter
        if trial_del is not None:
            if isinstance(trial_del, (list, np.ndarray)):
                mask = mask & np.isin(behavior['delta'], trial_del)
            else:
                mask = mask & (behavior['delta'] == trial_del)
        
        # Heading INDEX filter
        if trial_heading is not None:
            if isinstance(trial_heading, (list, np.ndarray)):
                mask = mask & np.isin(behavior['headingInd'], trial_heading)
            else:
                mask = mask & (behavior['headingInd'] == trial_heading)
        
        # PDW filter (high bet=1, low bet=0)
        if trial_pdw is not None:
            mask = mask & (behavior['PDW'] == trial_pdw)
        
        # Correctness filter
        if trial_correct is not None:
            mask = mask & (behavior['correct'] == trial_correct)
        
        return mask
    
    def _get_population_firing_rate(self, spikes, trial_mask, unit_indices, zscore_trials=False):
        """
        Calculate population firing rate across trials and units
        
        Parameters:
        -----------
        zscore_trials : bool
            If True, z-score each trial independently to remove trial-level baseline differences
        """
        if len(unit_indices) == 0:
            return None
        
        # Get relevant units and trials
        unit_spikes = spikes[unit_indices, :, :]  # (n_area_units, n_trials, n_time)
        trial_spikes = unit_spikes[:, trial_mask, :]  # (n_area_units, n_filtered_trials, n_time)
        
        if trial_spikes.shape[1] == 0:
            return None
    
        # Average across units to get population response per trial
        population_rate = np.mean(trial_spikes, axis=0)  # (n_filtered_trials, n_time)
        
        # Z-SCORE EACH TRIAL
        if zscore_trials:
            # For each trial, subtract its mean and divide by its std
            trial_means = np.mean(population_rate, axis=1, keepdims=True)  # (n_trials, 1)
            trial_stds = np.std(population_rate, axis=1, keepdims=True)    # (n_trials, 1)
            
            # Avoid division by zero
            trial_stds[trial_stds < 1e-10] = 1.0
            
            population_rate = (population_rate - trial_means) / trial_stds
            
            print(f"  ✓ Z-scored {population_rate.shape[0]} trials (removed trial-level baseline)")
        
        return population_rate          
        
    def _create_condition_string(self, trial_mod, trial_coh, trial_del,
                                trial_heading, trial_pdw, trial_correct):
        """Create human-readable condition string for filenames"""
        
        parts = []
        
        # Modality
        if trial_mod is not None:
            mod_names = {1: 'vest', 2: 'vis', 3: 'comb'}
            parts.append(f"mod{mod_names.get(trial_mod, trial_mod)}")
        else:
            parts.append("modAll")
        
        # Coherence
        if trial_coh is not None:
            coh_names = {1: 'low', 2: 'high'}
            parts.append(f"coh{coh_names.get(trial_coh, trial_coh)}")
        else:
            parts.append("cohAll")
        
        # Delta
        if trial_del is not None:
            if isinstance(trial_del, (list, np.ndarray)):
                parts.append(f"del{'-'.join(map(str, trial_del))}")
            else:
                parts.append(f"del{trial_del}")
        else:
            parts.append("delAll")
        
        # Heading
        if trial_heading is not None:
            if isinstance(trial_heading, (list, np.ndarray)):
                parts.append(f"hdg{'-'.join(map(str, trial_heading))}")
            else:
                parts.append(f"hdg{trial_heading}")
        else:
            parts.append("hdgAll")
        
        # PDW
        if trial_pdw is not None:
            pdw_names = {0: 'lowbet', 1: 'highbet'}
            parts.append(pdw_names.get(trial_pdw, f"pdw{trial_pdw}"))
        else:
            parts.append("pdwAll")
        
        # Correct
        if trial_correct is not None:
            correct_names = {0: 'error', 1: 'correct'}
            parts.append(correct_names.get(trial_correct, f"corr{trial_correct}"))
        else:
            parts.append("corrAll")
        
        return "_".join(parts)
    
    def _save_results(self, results, task_type, alignment, condition_str):
        """Save results to hierarchical folder structure"""
        
        # Create task-specific subfolder
        task_dir = self.save_dir / task_type
        task_dir.mkdir(exist_ok=True)
        
        # Create alignment-specific subfolder
        align_dir = task_dir / alignment
        align_dir.mkdir(exist_ok=True)
        
        # Add z-score suffix to filename
        zscore_suffix = "_zscored" if results['metadata'].get('zscore_trials', False) else "_raw"
        
        # Save main results
        filename = f"{condition_str}_correlation{zscore_suffix}.npz"
        filepath = align_dir / filename
        
        np.savez_compressed(
            filepath,
            correlation_matrix=results['correlation_matrix'],
            p_value_matrix=results['p_value_matrix'],
            time_axis=results['time_axis'],
            time_lags_ms=results['time_lags_ms'],
            metadata=results['metadata']
        )
        
        print(f"✓ Saved: {filepath.relative_to(self.base_save_dir)}")
        
        # Save data structure description
        description = self._generate_data_description(results)
        desc_filename = f"{condition_str}_correlation{zscore_suffix}_README.txt"
        desc_filepath = align_dir / desc_filename
        
        with open(desc_filepath, 'w', encoding='utf-8') as f:
            f.write(description)
    
    def _generate_data_description(self, results):
        """Generate human-readable data structure description"""
        
        meta = results['metadata']
        corr_shape = results['correlation_matrix'].shape
        
        interp_info = ""
        if meta.get('interpolation_used', False):
            interp_info = f"""
    INTERPOLATION:
    --------------
    Original bin size: {meta['original_bin_size_ms']:.2f} ms
    Interpolation factor: {meta['interpolation_factor']}x
    Effective bin size: {meta['effective_bin_size_ms']:.2f} ms

    ⚠ Note: Data was interpolated to achieve finer temporal resolution.
    This creates synthetic data points between actual measurements using
    cubic spline interpolation. Useful for exploring sub-bin dynamics,
    but should be validated with actual finer-sampled data when possible.
    """
        
        description = f"""
    ================================================================================
    INTER-AREA CORRELATION ANALYSIS - DATA STRUCTURE
    ================================================================================

    FILE INFORMATION:
    -----------------
    Subject: {meta['subject']}
    Date: {meta['date']}
    Task Type: {meta['task_type']}
    Alignment: {meta['alignment']}

    FILE LOCATION:
    --------------
    Hierarchical structure: subject/date/task_type/alignment/
    Current file: {meta['subject']}/{meta['date']}/{meta['task_type']}/{meta['alignment']}/

    EXPERIMENTAL CONDITIONS:
    ------------------------
    Modality (trial_mod): {meta['trial_mod']} (1=vest, 2=vis, 3=comb, None=all)
    Coherence (trial_coh): {meta['trial_coh']} (1=low, 2=high, None=all)
    Delta (trial_del): {meta['trial_del']} (0=no delay, None=all)
    Heading Index (trial_heading): {meta['trial_heading']} (4=zero, [1,7]=±10, None=all)
    PDW (trial_pdw): {meta['trial_pdw']} (1=high bet, 0=low bet, None=all)
    Correct (trial_correct): {meta['trial_correct']} (1=correct, 0=error, None=all)

    Condition String: {meta['condition_string']}
    {interp_info}
    DATA DIMENSIONS:
    ----------------
    Correlation Matrix Shape: {corr_shape}
    - Dimension 0 (rows): Time lags ({corr_shape[0]} lags)
    - Dimension 1 (cols): Time points ({corr_shape[1]} timepoints)

    Time Lags: {min(results['time_lags_ms'])} to {max(results['time_lags_ms'])} ms
    - Positive lag: VPS leads MST (VPS activity predicts future MST)
    - Negative lag: MST leads VPS (MST activity predicts future VPS)
    - Zero lag: Simultaneous activity

    Time Axis: {results['time_axis'][0]:.3f} to {results['time_axis'][-1]:.3f} seconds
    Effective Bin Size: {meta['effective_bin_size_ms']:.2f} ms

    NEURAL DATA:
    ------------
    MST Units: {meta['n_MST_units']} units
    VPS Units: {meta['n_VPS_units']} units
    Trials Used: {meta['n_trials_used']} trials

    ARRAYS IN .npz FILE:
    --------------------
    1. correlation_matrix: shape {corr_shape}
    - Pearson correlation coefficient at each (lag, time) point
    - Range: -1 to 1
    - NaN where insufficient data

    2. p_value_matrix: shape {corr_shape}
    - Statistical significance of correlations
    - p < 0.05 indicates significant correlation

    3. time_axis: shape ({len(results['time_axis'])},)
    - Time points in seconds relative to alignment event
    
    4. time_lags_ms: shape ({len(results['time_lags_ms'])},)
    - Time lag values in milliseconds

    5. metadata: dict
    - All experimental parameters and analysis details

    LOADING DATA IN PYTHON:
    -----------------------
    import numpy as np

    data = np.load('filename.npz', allow_pickle=True)
    corr_matrix = data['correlation_matrix']  # shape: (n_lags, n_timepoints)
    p_values = data['p_value_matrix']         # shape: (n_lags, n_timepoints)
    time_axis = data['time_axis']             # shape: (n_timepoints,)
    lags_ms = data['time_lags_ms']            # shape: (n_lags,)
    metadata = data['metadata'].item()        # dict

    # Check if data was interpolated
    if metadata.get('interpolation_used', False):
        print(f"Data interpolated {{metadata['interpolation_factor']}}x")
        print(f"Effective bin size: {{metadata['effective_bin_size_ms']:.2f}} ms")

    Generated: {np.datetime64('now')}
    ================================================================================
    """
    
        return description
    
    def load_results(self, task_type, alignment, condition_str):
        """Load previously saved correlation results"""
        
        filepath = self.save_dir / task_type / alignment / f"{condition_str}_correlation.npz"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        results = {
            'correlation_matrix': data['correlation_matrix'],
            'p_value_matrix': data['p_value_matrix'],
            'time_axis': data['time_axis'],
            'time_lags_ms': data['time_lags_ms'].tolist(),
            'metadata': data['metadata'].item()
        }
        
        return results
    
    def list_all_results(self):
        """List all saved results for this session"""
        
        all_files = {}
        
        for task_type in ['regular', 'tuning']:
            task_dir = self.save_dir / task_type
            if not task_dir.exists():
                continue
            
            all_files[task_type] = {}
            
            for align_dir in task_dir.iterdir():
                if not align_dir.is_dir():
                    continue
                
                alignment = align_dir.name
                npz_files = list(align_dir.glob('*_correlation.npz'))
                
                all_files[task_type][alignment] = [f.stem.replace('_correlation', '') 
                                                   for f in npz_files]
        
        return all_files
    
    def plot_correlation_heatmap(self, results, significance_threshold=0.05,
                                save_plot=True):
        """Plot correlation matrix as heatmap"""
        
        corr_matrix = results['correlation_matrix']
        p_matrix = results['p_value_matrix']
        time_axis = results['time_axis']
        lags_ms = results['time_lags_ms']
        meta = results['metadata']
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Correlation heatmap
        im1 = axes[0].imshow(corr_matrix, aspect='auto', cmap='RdBu_r',
                            vmin=-1, vmax=1,
                            extent=[time_axis[0], time_axis[-1], lags_ms[-1], lags_ms[0]])
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5, label='Zero lag')
        axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5, label='Alignment')
        axes[0].set_xlabel('Time (s)', fontsize=12)
        axes[0].set_ylabel('VPS lag relative to MST (ms)', fontsize=12)
        axes[0].set_title(f'MST-VPS Correlation\n{meta["condition_string"]}', fontsize=13)
        plt.colorbar(im1, ax=axes[0], label='Pearson r')
        axes[0].legend(loc='upper right', fontsize=9)
        
        # Plot 2: Significance mask
        sig_mask = p_matrix < significance_threshold
        im2 = axes[1].imshow(sig_mask.astype(float), aspect='auto', cmap='Greys',
                            extent=[time_axis[0], time_axis[-1], lags_ms[-1], lags_ms[0]])
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
        axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
        axes[1].set_xlabel('Time (s)', fontsize=12)
        axes[1].set_ylabel('VPS lag relative to MST (ms)', fontsize=12)
        axes[1].set_title(f'Significant Correlations (p < {significance_threshold})', fontsize=13)
        plt.colorbar(im2, ax=axes[1], label='Significant', ticks=[0, 1])
        
        # Title with interpolation info
        title = f'{meta["subject"]} {meta["date"]} - {meta["task_type"]} - {meta["alignment"]}\n'
        title += f'Trials: {meta["n_trials_used"]} | MST: {meta["n_MST_units"]} units | VPS: {meta["n_VPS_units"]} units'
        
        if meta.get('interpolation_used', False):
            title += f'\n[Interpolated {meta["interpolation_factor"]}x: {meta["effective_bin_size_ms"]:.1f}ms effective bins]'
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            task_dir = self.save_dir / meta['task_type']
            align_dir = task_dir / meta['alignment']
            plot_filename = f"{meta['condition_string']}_correlation_heatmap.png"
            plot_path = align_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {plot_path.relative_to(self.base_save_dir)}")
        
        plt.show()
        return fig
    
    def plot_correlation_across_alignments(self, results_dict, condition_name,
                                        significance_threshold=0.05,
                                        save_plot=True):
        """
        Plot correlation heatmaps across all three alignments side-by-side
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with keys 'stimOn', 'saccOnset', 'postTargHold'
            Each containing correlation results
        condition_name : str
            Name of the condition (e.g., 'mod3_coh2')
        significance_threshold : float
            p-value threshold for significance
        save_plot : bool
            Whether to save the plot
        """
        
        alignments = ['stimOn', 'saccOnset', 'postTargHold']
        alignment_labels = ['Stimulus Onset', 'Saccade Onset', 'Post-Decision Hold']
        
        # Check which alignments have data
        available_alignments = [a for a in alignments if a in results_dict and results_dict[a] is not None]
        n_alignments = len(available_alignments)
        
        if n_alignments == 0:
            print("No results available to plot")
            return None
        
        # Create figure with 2 rows × n_alignments columns
        fig = plt.figure(figsize=(6 * n_alignments, 10))
        gs = fig.add_gridspec(2, n_alignments, hspace=0.3, wspace=0.3)
        
        # Determine global color scale for consistency
        all_corr = []
        for align in available_alignments:
            if results_dict[align] is not None:
                all_corr.append(results_dict[align]['correlation_matrix'])
        
   
            
        vmin, vmax = -1, 1
        # Plot each alignment
        for idx, alignment in enumerate(available_alignments):
            results = results_dict[alignment]
            
            if results is None:
                continue
            
            corr_matrix = results['correlation_matrix']
            p_matrix = results['p_value_matrix']
            time_axis = results['time_axis']
            lags_ms = results['time_lags_ms']
            meta = results['metadata']
            
            # Row 1: Correlation heatmap
            ax1 = fig.add_subplot(gs[0, idx])
            
            im1 = ax1.imshow(corr_matrix, aspect='auto', cmap='RdBu_r',
                            vmin=vmin, vmax=vmax,
                            extent=[time_axis[0], time_axis[-1], 
                                lags_ms[-1], lags_ms[0]])
            
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
            ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
            ax1.set_xlabel('Time (s)', fontsize=11)
            ax1.set_ylabel('VPS lag relative to MST (ms)', fontsize=11)
            ax1.set_title(f'{alignment_labels[idx]}\n{condition_name}', fontsize=12, fontweight='bold')
            
            # Add colorbar only for the last subplot
            if idx == n_alignments - 1:
                cbar = plt.colorbar(im1, ax=ax1, label='Pearson r')
            
            # Row 2: Significance mask
            ax2 = fig.add_subplot(gs[1, idx])
            
            sig_mask = p_matrix < significance_threshold
            im2 = ax2.imshow(sig_mask.astype(float), aspect='auto', cmap='Greys',
                            extent=[time_axis[0], time_axis[-1], 
                                lags_ms[-1], lags_ms[0]])
            
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
            ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
            ax2.set_xlabel('Time (s)', fontsize=11)
            ax2.set_ylabel('VPS lag relative to MST (ms)', fontsize=11)
            ax2.set_title(f'Significant (p < {significance_threshold})', fontsize=11)
            
            # Add colorbar only for the last subplot
            if idx == n_alignments - 1:
                cbar2 = plt.colorbar(im2, ax=ax2, label='Significant', ticks=[0, 1])
        
        # Overall title
        title = f'{meta["subject"]} {meta["date"]} - {meta["task_type"]}\n'
        title += f'MST-VPS Correlation: {condition_name}\n'
        title += f'Trials: {meta["n_trials_used"]} | MST: {meta["n_MST_units"]} units | VPS: {meta["n_VPS_units"]} units'
        
        if meta.get('interpolation_used', False):
            title += f'\n[Interpolated {meta["interpolation_factor"]}x: {meta["effective_bin_size_ms"]:.1f}ms effective bins]'
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        if save_plot:
            # Save in the task directory (not alignment-specific)
            task_dir = self.save_dir / meta['task_type']
            plot_filename = f"{condition_name}_correlation_all_alignments.png"
            plot_path = task_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Multi-alignment plot saved: {plot_path.relative_to(self.base_save_dir)}")
        
        plt.show()
        return fig
    