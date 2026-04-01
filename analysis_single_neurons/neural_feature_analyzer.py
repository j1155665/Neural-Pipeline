import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class NeuralFeatureAnalyzer:
    def __init__(self, session_data, subject, date, alignment='stimOn', session_position=None, relative_position=None, center_position=(4, 4), mean_RT=None):
        """
        Initialize the analyzer with session data
        
        Parameters:
        -----------
        session_data : dict
            Dictionary containing loaded session data from load_session_data function
        subject : str
            Subject identifier
        date : str
            Session date
        alignment : str
            'stimOn' for tuning data or 'saccOnset' for regular task data
        session_position : tuple, optional
            Absolute position for this session as (x, y)
        relative_position : tuple, optional
            Relative position for this session as (rel_x, rel_y)
        center_position : tuple, optional
            Center position for reference, default (4, 4)
        """
        self.session_data = session_data
        self.subject = subject
        self.date = date
        self.alignment = alignment
        self.config = session_data['config']
        self.mean_RT = mean_RT
        
        # Choose data based on alignment
        if alignment == 'stimOn':
            self.behavior = session_data['behavior_tuning_converted']
            self.spikes_data = session_data['tuning_spikes_data'] 
            self.time_info = session_data['time_info_tuning']
            self.time_axes = session_data['time_axes_tuning']
            self.stim_on_t = session_data['time_info'] ['trial_start_t']
            self.is_tuning = True
            self.is_tuning_RT = True if mean_RT is not None else False
        else:  # saccOnset
            self.behavior = session_data['behavior_converted']
            self.spikes_data = session_data['spikes_data']
            self.time_info = session_data['time_info']
            self.time_axes = session_data['time_axes_dots3DMP']
            self.is_tuning = False
            self.is_tuning_RT = False # Reglar task has no RT tuning
  

        self.unit_info = session_data['unit_info']
        self.units_data = session_data['units_data']
        
        # Position information
        self.session_position = session_position or center_position
        self.relative_position = relative_position or (0, 0)
        self.center_position = center_position
        
        # Analysis parameters
        if self.is_tuning:
            # For tuning: sliding windows from t=0 to tmax-700ms
            self.tuning_time_info = {
                'offset': 0.025,
                'bin_size': 0.02,
                'align_events': ['stimOn'],
                'plot_names': ['Stim On'],
                'center_start': [-0.1],
                'center_stop': [2.2],
                'sigma': 0,
                'vel_profile_dt': 0.0083,
                'trial_start': -0.2,
                'trial_stop': 0.2
            }
            self.window_size_ms = 800

        else:
            # For regular task: -200 to 0 s before saccade
            self.analysis_window = [-200, 0]  # ms
        
        # Results storage
        self.unit_features = []
        self.n_units = self._get_n_units()
        
        print(f"Debug: Session {self.date} at position {self.session_position} (relative: {self.relative_position})")
        print(f"Debug: Alignment: {self.alignment}, Is tuning: {self.is_tuning}")
        print(f"Debug: Number of units determined: {self.n_units}")
    
    def _get_n_units(self):
        """Determine the number of units from the data structure"""
        try:
            for alignment in self.spikes_data.keys():
                spike_data = self.spikes_data[alignment]
                if hasattr(spike_data, 'shape'):
                    return spike_data.shape[0]
                elif hasattr(spike_data, '__len__'):
                    return len(spike_data)
            
            if hasattr(self.unit_info, 'shape'):
                return self.unit_info.shape[0]
            elif hasattr(self.unit_info, '__len__'):
                return len(self.unit_info)
            elif isinstance(self.unit_info, dict):
                for key in ['unit_id', 'channel', 'cluster_id']:
                    if key in self.unit_info and hasattr(self.unit_info[key], '__len__'):
                        return len(self.unit_info[key])
            
            total_units = 0
            for area, units in self.units_data.items():
                if hasattr(units, '__len__'):
                    total_units += len(units)
            
            return total_units if total_units > 0 else 0
            
        except Exception as e:
            print(f"Error determining number of units: {e}")
            return 0

    def get_time_windows_tuning(self):
        """Get sliding time windows for tuning data analysis"""
        try:
            time_axis = self.time_axes[self.alignment]
            bin_size_ms = self.tuning_time_info['bin_size'] * 1000  # 20ms
            window_bins = int(self.window_size_ms / bin_size_ms) 
            self.slide_bins = int(window_bins / 4)  

            # Find t=0 index (stimulus onset)
            t_zero_idx = np.argmin(np.abs(time_axis))
            
            # Find tmax-700ms index
            t_end = time_axis[-1]- 0.2
            t_end_idx = np.argmin(np.abs(time_axis - t_end))
            
            # Generate sliding windows
            windows = []
            start_idx = t_zero_idx
            end_idx = t_end_idx
            
            while end_idx - window_bins >= t_zero_idx:
                start_idx = end_idx - window_bins
                windows.append((start_idx, end_idx))
                end_idx -= self.slide_bins    
            return windows
            
        except Exception as e:
            print(f"Error in get_time_windows_tuning: {e}")
            # Fallback: single window
            return [(0, 25)]

    def get_time_window_indices_regular(self):
        """Get time window indices for regular task (-200 to 0ms before saccade)"""
        try:
   
            time_axis = self.time_axes[self.alignment]
            
            # Find indices for -200ms to 0ms
            start_idx = np.argmin(np.abs(time_axis*1000 - self.analysis_window[0]))
            end_idx = np.argmin(np.abs(time_axis*1000 - self.analysis_window[1]))
                
            return start_idx, end_idx
            
        except Exception as e:
            print(f"Error in get_time_window_indices_regular: {e}")
            return 0, 10
        
    def get_time_window_indices_tuning_RT(self, modality):
        """Get time window indices for RT tuning task based on mean RT for each modality"""
        try:
            time_axis = self.time_axes[self.alignment]
            
            # Determine which mean RT to use based on modality
            if modality == 1:
                rt_key = 'mod1_coh1'
            elif modality == 2:
                rt_key = 'mod2_coh2'
            elif modality == 3:
                rt_key = 'mod3_coh2'
            else:
                print(f"Unknown modality: {modality}")
                return 0, 10
            
            # Get the mean RT for this modality/coherence combination
            if rt_key not in self.mean_RT:
                print(f"Mean RT not found for {rt_key}")
                return 0, 10
                
            mean_rt_ms = (self.mean_RT[rt_key] + self.stim_on_t)  * 1000  # Convert to milliseconds
            
            # Define time window based on mean RT
            # You might want to adjust these values based on your analysis needs
            window_start = mean_rt_ms - 100  # 200ms before mean RT
            window_end = mean_rt_ms +100  # At mean RT
            
            # Find indices for the time window
            start_idx = np.argmin(np.abs(time_axis * 1000 - window_start))
            end_idx = np.argmin(np.abs(time_axis * 1000 - window_end))
            
            # Ensure valid indices
            if start_idx >= end_idx:
                print(f"Invalid time window indices for modality {modality}: start={start_idx}, end={end_idx}")
                return 0, 10
                
            return start_idx, end_idx
            
        except Exception as e:
            print(f"Error in get_time_window_indices_tuning_RT for modality {modality}: {e}")
            return 0, 10

    def _return_nan_result(self, n_headings, failure_reason):
        """Helper function for consistent NaN results"""
        return {
            'threshold': np.nan,
            'r_squared': np.nan,
            'fit_mu': np.nan,
            'fit_sigma': np.nan,
            'n_headings': n_headings,
            'failure_reason': failure_reason
        }

    def sort_trials_by_conditions(self):
        """Sort trials by heading and modality conditions"""
        try:
            conditions = {}
   
            headingInds = np.unique(self.behavior['headingInd'])
            modalities = np.unique(self.behavior['modality'])  
            headings = []

            if self.is_tuning:
                heading_values = {
                    1: -45, 2: -21.2, 3: -10, 4: -3.9,
                    5: 3.9, 6: 10, 7: 21.2, 8: 45
                }
                for headingInd in headingInds:
                    if headingInd in heading_values:
                        headings.append(heading_values[headingInd])
                headings = np.array(headings)
                
                for modality in modalities:
                    conditions[modality] = {}
                    for headingInd in headingInds:
                        if headingInd in heading_values:
                            heading = heading_values[headingInd]
                            mask = (self.behavior['headingInd'] == headingInd) & (self.behavior['modality'] == modality)
                            trial_indices = np.where(mask)[0]
                            conditions[modality][heading] = trial_indices
            else:
                heading_values = {
                    1: -10, 2: -3.9, 3: -1.5, 4: 0, 5: 1.5, 6: 3.9, 7: 10
                }

                for headingInd in headingInds:
                    if headingInd in heading_values:
                        headings.append(heading_values[headingInd])
                headings = np.array(headings)
                
                for modality in modalities:
                    if modality == 1:
                        coherence   =1
                    else:
                        coherence   = 2
                    conditions[modality] = {}
                    for headingInd in headingInds:
                        if headingInd in heading_values:
                            heading = heading_values[headingInd]
                            mask = (self.behavior['headingInd'] == headingInd) & (self.behavior['modality'] == modality) & (self.behavior['coherenceInd'] == coherence  )
                            trial_indices = np.where(mask)[0]
                            conditions[modality][heading] = trial_indices
                    
            return conditions, headings, modalities
            
        except Exception as e:
            print(f"Error in sort_trials_by_conditions: {e}")
            return {}, np.array([]), np.array([])

    def calculate_firing_rates_tuning(self, unit_idx):
        """Calculate firing rates for tuning data with sliding windows"""
        try:
            spike_data = self.spikes_data[self.alignment][unit_idx]
            windows = self.get_time_windows_tuning()
            
            conditions, headings, modalities = self.sort_trials_by_conditions()
            
            # Store results for all windows
            all_window_results = []
            neurothreshold_scores = []
            
            for window_idx, (start_idx, end_idx) in enumerate(windows):
                start_idx = max(0, min(start_idx, spike_data.shape[1] - 1))
                end_idx = max(start_idx + 1, min(end_idx, spike_data.shape[1]))
                
                window_spikes = spike_data[:, start_idx:end_idx]
                firing_rates = np.mean(window_spikes, axis=1)
                
                condition_rates = {}
                for modality in modalities:
                    condition_rates[modality] = {}
                    for heading in headings:
                        trial_indices = conditions[modality][heading]
                        if len(trial_indices) > 0:
                            condition_rates[modality][heading] = np.mean(firing_rates[trial_indices])
                        else:
                            condition_rates[modality][heading] = np.nan
                
                # Calculate neurometric thresholds for this window
                window_neurothresholds = {}
                window_score = 0
                
                for modality in modalities:
                    modality_name = {1.0: 'ves', 2.0: 'vis', 3.0: 'comb'}.get(modality, str(modality))
                    neurometric = self.calculate_neurometric_threshold_window(unit_idx, modality, start_idx, end_idx)
                    
                    # Store ALL neurometric results, not just threshold
                    window_neurothresholds[f'{modality_name}_threshold'] = neurometric['threshold']
                    window_neurothresholds[f'{modality_name}_r2'] = neurometric['r_squared']  # FIX: Store R²
                    window_neurothresholds[f'{modality_name}_mu'] = neurometric['fit_mu']     # FIX: Store mu
                    window_neurothresholds[f'{modality_name}_sigma'] = neurometric['fit_sigma'] # FIX: Store sigma
                    
                    # Score based on threshold quality (lower threshold = better discrimination)
                    if not np.isnan(neurometric['threshold']) and neurometric['r_squared'] > 0.1:
                        window_score += 1 / (neurometric['threshold'] + 1e-6)
                
                all_window_results.append({
                    'window_idx': window_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'condition_rates': condition_rates,
                    'firing_rates': firing_rates,
                    'neurothresholds': window_neurothresholds,
                    'score': window_score
                })
                
                neurothreshold_scores.append(window_score)
            
            # Find best window based on neurothreshold performance
            best_window_idx = np.argmax(neurothreshold_scores) if len(neurothreshold_scores) > 0 else 0
            best_window = all_window_results[best_window_idx] if len(all_window_results) > 0 else None
            
            return {
                'all_windows': all_window_results,
                'best_window': best_window,
                'best_window_idx': best_window_idx,
                'condition_rates': best_window['condition_rates'] if best_window else {},
                'firing_rates': best_window['firing_rates'] if best_window else np.array([])
            }
            
        except Exception as e:
            print(f"Error calculating firing rates for tuning unit {unit_idx}: {e}")
            return {'all_windows': [], 'best_window': None, 'condition_rates': {}, 'firing_rates': np.array([])}

    def calculate_firing_rates_regular(self, unit_idx):
        """Calculate firing rates for regular task data"""
        try:
            spike_data = self.spikes_data[self.alignment][unit_idx]
            start_idx, end_idx = self.get_time_window_indices_regular()
            
            start_idx = max(0, min(start_idx, spike_data.shape[1] - 1))
            end_idx = max(start_idx + 1, min(end_idx, spike_data.shape[1]))
            
            window_spikes = spike_data[:, start_idx:end_idx]
            firing_rates = np.mean(window_spikes, axis=1)
            
            conditions, headings, modalities = self.sort_trials_by_conditions()
            
            condition_rates = {}
            for modality in modalities:
                condition_rates[modality] = {}
                for heading in headings:
                    trial_indices = conditions[modality][heading]
                    if len(trial_indices) > 0:
                        condition_rates[modality][heading] = np.mean(firing_rates[trial_indices])
                    else:
                        condition_rates[modality][heading] = np.nan
                        
            return condition_rates, firing_rates
            
        except Exception as e:
            print(f"Error calculating firing rates for regular unit {unit_idx}: {e}")
            return {}, np.array([])
        
    def calculate_firing_rates_tuning_modality_specific(self, unit_idx):
        """Calculate firing rates for tuning data with modality-specific best windows"""
        try:
            spike_data = self.spikes_data[self.alignment][unit_idx]
            windows = self.get_time_windows_tuning()
            conditions, headings, modalities = self.sort_trials_by_conditions()
            
            modality_results = {}
            
            for modality in modalities:
                modality_name = {1.0: 'ves', 2.0: 'vis', 3.0: 'comb'}.get(modality, str(modality))
                
                best_score = -np.inf
                best_window_data = None
                best_window_idx = 0
                
                for window_idx, (start_idx, end_idx) in enumerate(windows):
                    start_idx = max(0, min(start_idx, spike_data.shape[1] - 1))
                    end_idx = max(start_idx + 1, min(end_idx, spike_data.shape[1]))
                    
                    window_spikes = spike_data[:, start_idx:end_idx]
                    firing_rates = np.mean(window_spikes, axis=1)
                    
                    condition_rates = {}
                    for m in modalities:
                        condition_rates[m] = {}
                        for heading in headings:
                            trial_indices = conditions[m][heading]
                            if len(trial_indices) > 0:
                                condition_rates[m][heading] = np.mean(firing_rates[trial_indices])
                            else:
                                condition_rates[m][heading] = np.nan
                    
                    neurometric = self.calculate_neurometric_threshold_window(unit_idx, modality, start_idx, end_idx)
                    
                    if not np.isnan(neurometric['threshold']) and neurometric['r_squared'] > 0.1:
                        score = 1 / (neurometric['threshold'] + 1e-6)
                        
                        if score > best_score:
                            best_score = score
                            best_window_idx = window_idx
                            best_window_data = {
                                'start_idx': start_idx,
                                'end_idx': end_idx,
                                'firing_rates': firing_rates,
                                'condition_rates': condition_rates,
                                'neurometric': neurometric
                            }
                
                modality_results[modality_name] = {
                    'best_window_idx': best_window_idx,
                    'best_window_data': best_window_data
                }
            
            return modality_results
            
        except Exception as e:
            print(f"Error calculating modality-specific firing rates for unit {unit_idx}: {e}")
            return {}

    def calculate_overall_firing_rate(self, firing_rates):
        """Calculate overall firing rate across all conditions"""
        if len(firing_rates) > 0:
            return np.mean(firing_rates)
        else:
            return 0.0

    def perform_anova_modality_specific(self, modality_results, modalities):
        """Perform ANOVA using modality-specific firing rates"""
        anova_results = {}
        
        try:
            conditions, headings, modality_types = self.sort_trials_by_conditions()
            
            for modality in modality_types:
                modality_name = {1.0: 'ves', 2.0: 'vis', 3.0: 'comb'}.get(modality, str(modality))
                
                if modality_name in modality_results and modality_results[modality_name]['best_window_data']:
                    firing_rates = modality_results[modality_name]['best_window_data']['firing_rates']
                    
                    modality_mask = self.behavior['modality'] == modality
                    modality_firing_rates = firing_rates[modality_mask]
                    modality_heading_inds = self.behavior['headingInd'][modality_mask]
                    
                    if len(np.unique(modality_heading_inds)) > 1 and len(modality_firing_rates) > 2:
                        groups = []
                        for heading_ind in np.unique(modality_heading_inds):
                            heading_rates = modality_firing_rates[modality_heading_inds == heading_ind]
                            if len(heading_rates) > 0:
                                groups.append(heading_rates)
                        
                        if len(groups) > 1:
                            f_stat, p_val = stats.f_oneway(*groups)
                            anova_results[f'{modality_name}_selective'] = 1 if p_val < 0.05 else 0
                            anova_results[f'{modality_name}_pval'] = p_val
                        else:
                            anova_results[f'{modality_name}_selective'] = 0
                            anova_results[f'{modality_name}_pval'] = np.nan
                    else:
                        anova_results[f'{modality_name}_selective'] = 0
                        anova_results[f'{modality_name}_pval'] = np.nan
                else:
                    anova_results[f'{modality_name}_selective'] = 0
                    anova_results[f'{modality_name}_pval'] = np.nan
                    
        except Exception as e:
            print(f"Error in modality-specific ANOVA: {e}")
            
        return anova_results
    
    def perform_anova(self, firing_rates, modalities):
        """Perform ANOVA for each modality to test selectivity"""
        anova_results = {}
        
        try:
            conditions, headings, modality_types = self.sort_trials_by_conditions()
            
            for modality in modality_types:
                try:
                    modality_mask = self.behavior['modality'] == modality
                    modality_firing_rates = firing_rates[modality_mask]
                    modality_heading_inds = self.behavior['headingInd'][modality_mask]
                    
                    if len(np.unique(modality_heading_inds)) > 1 and len(modality_firing_rates) > 2:
                        groups = []
                        for heading_ind in np.unique(modality_heading_inds):
                            heading_rates = modality_firing_rates[modality_heading_inds == heading_ind]
                            if len(heading_rates) > 0:
                                groups.append(heading_rates)
                        
                        if len(groups) > 1:
                            f_stat, p_val = stats.f_oneway(*groups)
                            modality_name = {1.0: 'ves', 2.0: 'vis', 3.0: 'comb'}.get(modality, str(modality))
                            anova_results[f'{modality_name}_selective'] = 1 if p_val < 0.05 else 0
                            anova_results[f'{modality_name}_pval'] = p_val
                        else:
                            modality_name = {1.0: 'ves', 2.0: 'vis', 3.0: 'comb'}.get(modality, str(modality))
                            anova_results[f'{modality_name}_selective'] = 0
                            anova_results[f'{modality_name}_pval'] = np.nan
                    else:
                        modality_name = {1.0: 'ves', 2.0: 'vis', 3.0: 'comb'}.get(modality, str(modality))
                        anova_results[f'{modality_name}_selective'] = 0
                        anova_results[f'{modality_name}_pval'] = np.nan
                        
                except Exception as e:
                    modality_name = {1.0: 'ves', 2.0: 'vis', 3.0: 'comb'}.get(modality, str(modality))
                    anova_results[f'{modality_name}_selective'] = 0
                    anova_results[f'{modality_name}_pval'] = np.nan
                    
        except Exception as e:
            print(f"Error in perform_anova: {e}")
            
        return anova_results
    
    def calculate_correlations_modality_specific(self, modality_results, headings):
        """Calculate correlations using modality-specific condition rates"""
        correlations = {'ves_corr_r': np.nan, 'ves_corr_p': np.nan, 'vis_corr_r': np.nan, 'vis_corr_p': np.nan}
        
        try:
            if 'ves' in modality_results and modality_results['ves']['best_window_data']:
                condition_rates = modality_results['ves']['best_window_data']['condition_rates']
                if 1.0 in condition_rates:
                    ves_rates = []
                    ves_headings = []
                    for heading in sorted(headings):
                        if heading in condition_rates[1.0] and not np.isnan(condition_rates[1.0][heading]):
                            ves_rates.append(condition_rates[1.0][heading])
                            ves_headings.append(heading)
                    
                    if len(ves_rates) > 2:
                        r_ves, p_ves = pearsonr(ves_rates, ves_headings)
                        correlations['ves_corr_r'] = r_ves
                        correlations['ves_corr_p'] = p_ves
            
            if 'vis' in modality_results and modality_results['vis']['best_window_data']:
                condition_rates = modality_results['vis']['best_window_data']['condition_rates']
                if 2.0 in condition_rates:
                    vis_rates = []
                    vis_headings = []
                    for heading in sorted(headings):
                        if heading in condition_rates[2.0] and not np.isnan(condition_rates[2.0][heading]):
                            vis_rates.append(condition_rates[2.0][heading])
                            vis_headings.append(heading)
                    
                    if len(vis_rates) > 2:
                        r_vis, p_vis = pearsonr(vis_rates, vis_headings)
                        correlations['vis_corr_r'] = r_vis
                        correlations['vis_corr_p'] = p_vis
                        
        except Exception as e:
            print(f"Error in modality-specific correlations: {e}")
            
        return correlations

    def calculate_correlations(self, condition_rates, headings):
        """Calculate Pearson correlations between firing rates and headings for each modality"""
        correlations = {'ves_corr_r': np.nan, 'ves_corr_p': np.nan, 'vis_corr_r': np.nan, 'vis_corr_p': np.nan}
        
        try:
            if 1.0 in condition_rates:
                ves_rates = []
                ves_headings = []
                for heading in sorted(headings):
                    if heading in condition_rates[1.0] and not np.isnan(condition_rates[1.0][heading]):
                        ves_rates.append(condition_rates[1.0][heading])
                        ves_headings.append(heading)
                
                if len(ves_rates) > 2:
                    r_ves, p_ves = pearsonr(ves_rates, ves_headings)
                    correlations['ves_corr_r'] = r_ves
                    correlations['ves_corr_p'] = p_ves
            
            if 2.0 in condition_rates:
                vis_rates = []
                vis_headings = []
                for heading in sorted(headings):
                    if heading in condition_rates[2.0] and not np.isnan(condition_rates[2.0][heading]):
                        vis_rates.append(condition_rates[2.0][heading])
                        vis_headings.append(heading)
                
                if len(vis_rates) > 2:
                    r_vis, p_vis = pearsonr(vis_rates, vis_headings)
                    correlations['vis_corr_r'] = r_vis
                    correlations['vis_corr_p'] = p_vis
                    
        except Exception as e:
            print(f"Error in calculate_correlations: {e}")
            
        return correlations

    def get_unit_area(self, unit_idx):
        """Determine the area for a given unit index based on relative depth"""
        try:
            positions = self.calculate_relative_positions()
            relative_depth = positions['relative_depth'][unit_idx]
            
            if relative_depth > 0:
                return 'VPS'
            elif relative_depth < 0:
                return 'MST'
            else:
                return 'MST'
            
        except Exception as e:
            print(f"Error determining area for unit {unit_idx} using depth: {e}")
            return 'unknown'
        
    def calculate_relative_positions(self):
        """Calculate positions and depths"""
        try:
            depths = None
            if hasattr(self.unit_info, 'columns') and 'depth' in self.unit_info.columns:
                depths = self.unit_info['depth'].values
            elif hasattr(self.unit_info, 'get') and 'depth' in self.unit_info:
                depths = np.array(self.unit_info['depth'])
            elif isinstance(self.unit_info, dict) and 'depth' in self.unit_info:
                depths = np.array(self.unit_info['depth'])
            
            if depths is None:
                depths = np.zeros(self.n_units)
            
            if len(depths) > 0 and isinstance(depths[0], str):
                cleaned_depths = []
                for depth_val in depths:
                    if isinstance(depth_val, str):
                        try:
                            cleaned_depths.append(float(depth_val.strip('[]')))
                        except:
                            cleaned_depths.append(0.0)
                    else:
                        cleaned_depths.append(float(depth_val))
                depths = np.array(cleaned_depths)
            
            if depths.ndim > 1:
                depths = depths.flatten()
            
            if len(depths) != self.n_units:
                if len(depths) > self.n_units:
                    depths = depths[:self.n_units]
                else:
                    depths = np.concatenate([depths, np.zeros(self.n_units - len(depths))])
            
            mst_units = self.units_data.get('MST', [])
            if len(mst_units) > 0:
                valid_mst_units = [idx for idx in mst_units if idx < len(depths)]
                if valid_mst_units:
                    max_mst_depth = np.max(depths[valid_mst_units])
                    relative_depths = depths - max_mst_depth
                else:
                    relative_depths = depths
            else:
                relative_depths = depths
            
            return {
                'absolute_x': np.full(self.n_units, self.session_position[0]),
                'absolute_y': np.full(self.n_units, self.session_position[1]),
                'relative_x': np.full(self.n_units, self.relative_position[0]),
                'relative_y': np.full(self.n_units, self.relative_position[1]),
                'absolute_depth': depths,
                'relative_depth': relative_depths
            }
        except:
            return {
                'absolute_x': np.full(self.n_units, self.session_position[0]),
                'absolute_y': np.full(self.n_units, self.session_position[1]),
                'relative_x': np.full(self.n_units, self.relative_position[0]),
                'relative_y': np.full(self.n_units, self.relative_position[1]),
                'absolute_depth': np.zeros(self.n_units),
                'relative_depth': np.zeros(self.n_units)
            }

    def calculate_roc_area(self, responses1, responses2):
        """Calculate ROC area between two response distributions"""
        try:
            if len(responses1) == 0 or len(responses2) == 0:
                return 0.5
            
            from sklearn.metrics import roc_auc_score
            
            all_responses = np.concatenate([responses1, responses2])
            labels = np.concatenate([np.zeros(len(responses1)), np.ones(len(responses2))])
            
            auc = roc_auc_score(labels, all_responses)
            ideal_auc = max(auc, 1 - auc)
            
            return ideal_auc
            
        except Exception as e:
            return 0.5

    def cumulative_gaussian(self, x, mu, sigma):
        """Cumulative Gaussian function for fitting neurometric curves"""
        if sigma <= 0:
            sigma = 1e-6
        return stats.norm.cdf(x, loc=mu, scale=sigma)

    def calculate_neurometric_threshold_window(self, unit_idx, modality, start_idx, end_idx):
        """Calculate neurometric threshold for a specific time window"""
        try:
            spike_data = self.spikes_data[self.alignment][unit_idx]
            window_spikes = spike_data[:, start_idx:end_idx]
            trial_firing_rates = np.mean(window_spikes, axis=1)
            
            heading_values_map = {
                1: -45, 2: -21.2, 3: -10, 4: -3.9,
                5: 3.9, 6: 10, 7: 21.2, 8: 45
            }
            
            modality_mask = self.behavior['modality'] == modality
            modality_heading_inds = self.behavior['headingInd'][modality_mask]
            modality_rates = trial_firing_rates[modality_mask]
            
            modality_headings = []
            valid_mask = []
            for i, heading_ind in enumerate(modality_heading_inds):
                if heading_ind in heading_values_map:
                    modality_headings.append(heading_values_map[heading_ind])
                    valid_mask.append(True)
                else:
                    valid_mask.append(False)
            
            modality_headings = np.array(modality_headings)
            modality_rates = modality_rates[valid_mask]
            
            unique_headings = np.unique(modality_headings)
            
            roc_values = []
            heading_values = []
            
            sorted_headings = np.sort(unique_headings)
            
            for heading in sorted_headings:
                if heading <= 0 or heading > 40:
                    continue  

                opposite_heading = -heading
                
                if opposite_heading in unique_headings:
                    heading_responses = modality_rates[modality_headings == heading]
                    opposite_responses = modality_rates[modality_headings == opposite_heading]
                    
                    if len(heading_responses) > 0 and len(opposite_responses) > 0:

                        # this returns the "ideal ROC", which is always >= 0.5, so we don't need to check if we need to invert
                        roc_area = self.calculate_roc_area(heading_responses, opposite_responses)
                        
                        roc_values.append(roc_area)
                        heading_values.append(abs(heading))
                        roc_values.append(1 - roc_area)
                        heading_values.append(-heading)
        
            heading_values = np.array(heading_values)
            roc_values = np.array(roc_values)

            sort_idx = np.argsort(heading_values)
            heading_values = heading_values[sort_idx]   
            roc_values = roc_values[sort_idx]
            

            if len(roc_values) < 3:
                return self._return_nan_result(len(unique_headings), "insufficient_data_points")
            
            if np.std(roc_values) < 0.01:
                return self._return_nan_result(len(unique_headings), "no_discrimination")

            try:           
                # Create multiple reasonable initial guesses
                initial_guesses = [
                    [0.0, 8],       # μ=0, σ=8
                    [0.0, 15],      # μ=0, σ=15  
                    [0.0, 50]       # μ=0, σ=50
                ]

                # IMPROVED BOUNDS - constrain μ to be near 0
                if self.is_tuning:
                    mu_lower = -1e-6     
                    mu_upper = 1e-6      
                    sigma_lower = 3.0
                    sigma_upper = 300
                    threshold_upper = 300  # σ * 0.994 ≈ 300 for max σ=300
                else:
                    mu_lower = -1e-6     
                    mu_upper = 1e-6     
                    sigma_lower = 1.0
                    sigma_upper = 300
                    threshold_upper = 300  # σ * 0.994 ≈ 300 for max σ=300
                
                bounds = ([mu_lower, sigma_lower], [mu_upper, sigma_upper])
                
                best_fit = None
                best_r_squared = -np.inf
                
                for i, p0 in enumerate(initial_guesses):
                    try:
                        # p0 is already [μ, σ] format, just clip to bounds
                        p0_clipped = [
                            np.clip(p0[0], mu_lower, mu_upper),  # μ ≈ 0
                            np.clip(p0[1], sigma_lower, sigma_upper)  # σ
                        ]
                        
                        popt, pcov = curve_fit(
                            self.cumulative_gaussian, 
                            heading_values, 
                            roc_values, 
                            p0=p0_clipped, 
                            bounds=bounds, 
                            maxfev=5000
                        )
                        
                        mu_fit, sigma_fit = popt
                        
                        y_pred = self.cumulative_gaussian(heading_values, mu_fit, sigma_fit)
                        ss_res = np.sum((roc_values - y_pred) ** 2)
                        ss_tot = np.sum((roc_values - np.mean(roc_values)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
                        
                        threshold_84 = mu_fit + sigma_fit * stats.norm.ppf(0.84)
                        # Since mu_fit ≈ 0, threshold_84 ≈ sigma_fit * 0.994

                        # Improved validation
                        if (threshold_84 > 2.0 and threshold_84 < threshold_upper and  # Minimum 2° threshold
                            sigma_fit > sigma_lower and sigma_fit < sigma_upper and
                            r_squared > -3 and  # More reasonable R² threshold
                            not np.isnan(threshold_84) and 
                            not np.isinf(threshold_84)):
                            
                            if r_squared > best_r_squared:
                                best_fit = {
                                    'threshold': threshold_84,
                                    'r_squared': r_squared,
                                    'fit_mu': mu_fit,
                                    'fit_sigma': sigma_fit,
                                    'n_headings': len(unique_headings),
                                    'attempt': i
                                }
                                best_r_squared = r_squared
                                
                    except Exception as fit_error:
                        continue
                
                if best_fit is not None:
                    return best_fit
                else:
                    return self._return_nan_result(len(unique_headings), "curve_fitting_failed")
                    
            except Exception as e:
                return self._return_nan_result(len(unique_headings), f"fitting_error_{str(e)[:20]}")
                
        except Exception as e:
            return self._return_nan_result(0, f"general_error_{str(e)[:20]}")

    def calculate_neurometric_threshold(self, unit_idx, modality):
        """Calculate neurometric threshold - wrapper for different data types"""
        if self.is_tuning and self.mean_RT is None:
            # For tuning, this is handled in calculate_firing_rates_tuning
            return self._return_nan_result(0, "use_tuning_method")
        else:
            if self.mean_RT is not None:
                start_idx, end_idx = self.get_time_window_indices_tuning_RT(modality)
            
            else:
                # For regular task or RT task, use single time window
                start_idx, end_idx = self.get_time_window_indices_regular()
            return self.calculate_neurometric_threshold_window(unit_idx, modality, start_idx, end_idx)

    def analyze_all_units(self):
        """Analyze all units and extract features"""
        print(f"Analyzing units for {self.subject} {self.date} with alignment {self.alignment}")
        
        try:
            positions = self.calculate_relative_positions()
            conditions, headings, modalities = self.sort_trials_by_conditions()
            
            print(f"Processing {self.n_units} units...")
            
            for unit_idx in range(self.n_units):
                try:    
                    if unit_idx % 10 == 0 or unit_idx < 5:
                        print(f"  Processing unit {unit_idx + 1}/{self.n_units}")
                    
                    if self.is_tuning and self.mean_RT is None:  # FIXED: use 'and' not '&'
                        modality_results = self.calculate_firing_rates_tuning_modality_specific(unit_idx)
                        
                        best_neurothresholds = {}
                        best_window_indices = {}
                        
                        for modality_name in ['ves', 'vis', 'comb']:
                            if modality_name in modality_results and modality_results[modality_name]['best_window_data']:
                                neurometric = modality_results[modality_name]['best_window_data']['neurometric']
                                best_neurothresholds[f'{modality_name}_threshold'] = neurometric['threshold']
                                best_neurothresholds[f'{modality_name}_r2'] = neurometric['r_squared']
                                best_neurothresholds[f'{modality_name}_mu'] = neurometric['fit_mu']
                                best_neurothresholds[f'{modality_name}_sigma'] = neurometric['fit_sigma']
                                best_window_indices[f'{modality_name}_best_window_idx'] = modality_results[modality_name]['best_window_idx']
                        
                        anova_results = self.perform_anova_modality_specific(modality_results, modalities)
                        correlations = self.calculate_correlations_modality_specific(modality_results, headings)
                        
                        overall_rate = 0
                        for modality_name in ['ves', 'vis', 'comb']:
                            if modality_name in modality_results and modality_results[modality_name]['best_window_data']:
                                firing_rates = modality_results[modality_name]['best_window_data']['firing_rates']
                                overall_rate += np.mean(firing_rates)
                        overall_rate /= 3
                        
                    else:
                        # Initialize variables
                        best_neurothresholds = {}
                        best_window_indices = {}
                        
                        # Calculate neurometric thresholds first
                        for modality in modalities:
                            modality_name = {1.0: 'ves', 2.0: 'vis', 3.0: 'comb'}.get(modality, str(modality))
                            neurometric = self.calculate_neurometric_threshold(unit_idx, modality)
                            best_neurothresholds[f'{modality_name}_threshold'] = neurometric['threshold']
                            best_neurothresholds[f'{modality_name}_r2'] = neurometric['r_squared']
                            best_neurothresholds[f'{modality_name}_mu'] = neurometric['fit_mu']
                            best_neurothresholds[f'{modality_name}_sigma'] = neurometric['fit_sigma']
                        
                        # Only do ANOVA and correlations for non-RT tuning
                        if not self.is_tuning_RT:
                            condition_rates, firing_rates = self.calculate_firing_rates_regular(unit_idx)
                            
                            if len(firing_rates) == 0:  # MOVED: check after firing_rates is defined
                                continue
                                
                            overall_rate = self.calculate_overall_firing_rate(firing_rates)
                            anova_results = self.perform_anova(firing_rates, modalities)
                            correlations = self.calculate_correlations(condition_rates, headings)
                        else:
                            # For RT tuning, minimal analysis
                            anova_results = {}
                            correlations = {}
                            overall_rate = 0.0  # FIXED: should be a number, not dict
                    
                    unit_area = self.get_unit_area(unit_idx)
                    
                    neurometric_results = {}
                    for key, value in best_neurothresholds.items():
                        if '_threshold' in key:
                            neurometric_results[key.replace('_threshold', '_neurometric_threshold')] = value
                        elif '_r2' in key:
                            neurometric_results[key.replace('_r2', '_neurometric_r2')] = value
                        elif '_mu' in key:
                            neurometric_results[key.replace('_mu', '_neurometric_mu')] = value
                        elif '_sigma' in key:
                            neurometric_results[key.replace('_sigma', '_neurometric_sigma')] = value
                        else:
                            neurometric_results[key] = value
                    
                    features = {
                        'subject': self.subject,
                        'date': self.date,
                        'alignment': self.alignment,
                        'is_tuning': self.is_tuning,
                        'is_tuning_RT': self.is_tuning_RT,  # Add this flag
                        'unit_idx': unit_idx,
                        'area': unit_area,
                        'absolute_x': positions['absolute_x'][unit_idx],
                        'absolute_y': positions['absolute_y'][unit_idx],
                        'relative_x': positions['relative_x'][unit_idx],
                        'relative_y': positions['relative_y'][unit_idx],
                        'absolute_depth': positions['absolute_depth'][unit_idx],
                        'relative_depth': positions['relative_depth'][unit_idx],
                        'overall_firing_rate': overall_rate,
                        **anova_results,
                        **correlations,
                        **neurometric_results,
                        **best_window_indices
                    }
                    
                    self.unit_features.append(features)
                    
                except Exception as e:
                    if unit_idx < 5:
                        print(f"    Error analyzing unit {unit_idx}: {e}")
                    continue
            
            print(f"Successfully analyzed {len(self.unit_features)} units")
            
        except Exception as e:
            print(f"Error in analyze_all_units: {e}")

    def get_features_dataframe(self):
        """Return features as a pandas DataFrame with clean numeric columns"""
        df = pd.DataFrame(self.unit_features)
        
        # Ensure position and rate columns are numeric
        numeric_cols = ['absolute_x', 'absolute_y', 'relative_x', 'relative_y', 
                    'absolute_depth', 'relative_depth', 'overall_firing_rate',
                    'ves_neurometric_threshold', 'vis_neurometric_threshold', 'comb_neurometric_threshold',
                    'ves_neurometric_r2', 'vis_neurometric_r2', 'comb_neurometric_r2',
                    'ves_neurometric_mu', 'vis_neurometric_mu', 'comb_neurometric_mu',
                    'ves_neurometric_sigma', 'vis_neurometric_sigma', 'comb_neurometric_sigma',
                    'ves_neurometric_n_headings', 'vis_neurometric_n_headings', 'comb_neurometric_n_headings',
                    'ves_corr_r', 'ves_corr_p', 'vis_corr_r', 'vis_corr_p',
                    'ves_pval', 'vis_pval', 'comb_pval',
                    # NEW: Add baseline columns
                    'ves_baseline_threshold', 'vis_baseline_threshold', 'comb_baseline_threshold',
                    'ves_baseline_r2', 'vis_baseline_r2', 'comb_baseline_r2',
                    'ves_baseline_mu', 'vis_baseline_mu', 'comb_baseline_mu',
                    'ves_baseline_sigma', 'vis_baseline_sigma', 'comb_baseline_sigma']
        
        # Add tuning-specific columns if present
        if self.is_tuning:
            numeric_cols.extend(['tuning_n_windows', 'tuning_best_window_idx'])
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def save_features(self, filename=None, save_dir=None):
        """Save features to CSV file with alignment-specific naming and error handling"""
        if save_dir is None:
            if self.is_tuning:
                save_dir = r'D:\Neural-Pipeline\results\analysis_single_neurons\dots3DMPtuning_neuralfeatures'
            else:
                save_dir = r'D:\Neural-Pipeline\results\analysis_single_neurons\dots3DMP_neuralfeatures'
        
        # Create directory if it doesn't exist
        import os
        import time
        from datetime import datetime
        os.makedirs(save_dir, exist_ok=True)
        
        if filename is None:
            # CHANGED: Add RT tuning as a separate suffix
            if self.is_tuning:
                if hasattr(self, 'mean_RT') and self.mean_RT is not None:
                    task_suffix = 'tuning_RT'  # Separate suffix for RT tuning
                else:
                    task_suffix = 'tuning'     # Regular tuning
            else:
                task_suffix = 'regular'        # Regular task
            filename = f"{self.subject}_{self.date}_{self.alignment}_{task_suffix}_neural_features.csv"
        
        # Full path
        full_path = os.path.join(save_dir, filename)
        
        df = self.get_features_dataframe()
        
        # Try to save with error handling
        max_attempts = 3
        saved_successfully = False
        
        for attempt in range(max_attempts):
            try:
                df.to_csv(full_path, index=False)
                print(f"Features saved to {full_path}")
                saved_successfully = True
                break
            except PermissionError:
                if attempt < max_attempts - 1:
                    print(f"Permission denied for {filename}, attempt {attempt + 1}/{max_attempts}. Waiting 2 seconds...")
                    time.sleep(2)
                else:
                    # Create alternative filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = filename.replace('.csv', '')
                    alt_filename = f"{base_name}_{timestamp}.csv"
                    alt_path = os.path.join(save_dir, alt_filename)
                    
                    try:
                        df.to_csv(alt_path, index=False)
                        print(f"Original file locked, features saved to {alt_path}")
                        full_path = alt_path  # Update return path
                        saved_successfully = True
                    except Exception as e:
                        print(f"Failed to save features: {e}")
                        return None
        
        # CHANGED: Only save dynamic data for regular tuning (not RT tuning)
        if self.is_tuning and (not hasattr(self, 'mean_RT') or self.mean_RT is None) and \
        hasattr(self, 'unit_features') and len(self.unit_features) > 0 and saved_successfully:
            # Save dynamic thresholds AND R² separately for detailed analysis
            dynamic_data = []
            for unit_features in self.unit_features:
                unit_idx = unit_features['unit_idx']
                for modality_name in ['ves', 'vis', 'comb']:
                    # Dynamic thresholds
                    threshold_key = f'{modality_name}_dynamic_thresholds'
                    r2_key = f'{modality_name}_dynamic_r2s'
                    mu_key = f'{modality_name}_dynamic_mus'
                    sigma_key = f'{modality_name}_dynamic_sigmas'
                    
                    if threshold_key in unit_features and isinstance(unit_features[threshold_key], list):
                        for window_idx, threshold in enumerate(unit_features[threshold_key]):
                            # Get corresponding R², mu, sigma values
                            r2_val = unit_features[r2_key][window_idx] if (r2_key in unit_features and 
                                                                        isinstance(unit_features[r2_key], list) and 
                                                                        window_idx < len(unit_features[r2_key])) else np.nan
                            mu_val = unit_features[mu_key][window_idx] if (mu_key in unit_features and 
                                                                        isinstance(unit_features[mu_key], list) and 
                                                                        window_idx < len(unit_features[mu_key])) else np.nan
                            sigma_val = unit_features[sigma_key][window_idx] if (sigma_key in unit_features and 
                                                                            isinstance(unit_features[sigma_key], list) and 
                                                                            window_idx < len(unit_features[sigma_key])) else np.nan
                            
                            dynamic_data.append({
                                'subject': unit_features['subject'],
                                'date': unit_features['date'],
                                'unit_idx': unit_idx,
                                'modality': modality_name,
                                'window_idx': window_idx,
                                'threshold': threshold,
                                'r_squared': r2_val,
                                'fit_mu': mu_val,
                                'fit_sigma': sigma_val,
                                'is_best_window': window_idx == unit_features.get('tuning_best_window_idx', -1)
                            })
            
            if dynamic_data:
                dynamic_df = pd.DataFrame(dynamic_data)
                dynamic_filename = f"{self.subject}_{self.date}_{self.alignment}_tuning_dynamic_thresholds.csv"
                dynamic_path = os.path.join(save_dir, dynamic_filename)
                
                # Try to save dynamic data with same error handling
                for attempt in range(max_attempts):
                    try:
                        dynamic_df.to_csv(dynamic_path, index=False)
                        print(f"Dynamic tuning data (including R²) saved to {dynamic_path}")
                        break
                    except PermissionError:
                        if attempt < max_attempts - 1:
                            print(f"Permission denied for dynamic data, attempt {attempt + 1}/{max_attempts}. Waiting 2 seconds...")
                            time.sleep(2)
                        else:
                            # Create alternative filename for dynamic data
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            base_name = dynamic_filename.replace('.csv', '')
                            alt_dynamic_filename = f"{base_name}_{timestamp}.csv"
                            alt_dynamic_path = os.path.join(save_dir, alt_dynamic_filename)
                            
                            try:
                                dynamic_df.to_csv(alt_dynamic_path, index=False)
                                print(f"Original dynamic file locked, saved to {alt_dynamic_path}")
                            except Exception as e:
                                print(f"Failed to save dynamic data: {e}")
        
        return full_path if saved_successfully else None