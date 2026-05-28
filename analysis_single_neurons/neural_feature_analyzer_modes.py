import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class NeuralFeatureAnalyzer:
    def __init__(self, session_data, subject, date, alignment='stimOn', session_position=None, relative_position=None, is_tuning=False,
            session_guidetube_depth=None, relative_guidetube_depth=None, center_position=(4, 4), corrected_depth=22, mean_RT=None, analysis_window=[-150, 150]):
    
        # Basic session information
        self.session_data = session_data
        self.subject = subject
        self.date = date
        self.alignment = alignment
        self.config = session_data['config']
        
        # Task type flags
        self.is_tuning = is_tuning
        self.mean_RT = mean_RT
        self.is_RT = True if mean_RT is not None else False
        
        # Time window parameter
        # For Modes 2 & 3 (RT-based): fixed [-200, 0]ms relative to mean RT per modality
        # For Mode 4 (saccade-aligned): fixed [-200, 0]ms relative to saccade onset
        self.analysis_window = analysis_window
        
        # Load appropriate data based on task type
        if self.is_tuning:
            self.behavior = session_data['behavior_tuning_converted']
            self.spikes_data = session_data['tuning_spikes_data'] 
            self.time_info = session_data['time_info_tuning']
            self.time_axes = session_data['time_axes_tuning']
            self.time_zero_shift_s = session_data['time_info']['trial_start_t'] 

            
            # Tuning-specific time window parameters (for Mode 1)
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
            self.neurometric_window_ms = 800
            self.neurometric_slide_ms = 400
            self.tuning_window_ms = 600
            self.tuning_slide_ms = 300
        else:
            self.behavior = session_data['behavior_converted']
            self.spikes_data = session_data['spikes_data']
            self.time_info = session_data['time_info']
            self.time_axes = session_data['time_axes_dots3DMP']
            self.time_zero_shift_s = 0
        
        # Unit information
        self.unit_info = session_data['unit_info']
        self.units_data = session_data['units_data']
        
        # Spatial positioning
        self.center_position = center_position
        self.session_position = session_position if session_position is not None else center_position
        self.relative_position = relative_position if relative_position is not None else (0, 0)
        
        # Depth information
        self.session_guidetube_depth = session_guidetube_depth if session_guidetube_depth is not None else (0, 0)
        self.relative_guidetube_depth = relative_guidetube_depth if relative_guidetube_depth is not None else 0.0
        
        # Initialize storage for unit features
        self.unit_features = []
        self.n_units = self._get_n_units()
        
        # Determine which mode we're in
        if self.is_tuning and not self.is_RT:
            mode_description = "Mode 1: Tuning with sliding windows"
        elif self.is_tuning and self.is_RT:
            mode_description = "Mode 2: Tuning with mean RT-based fixed windows"
        elif not self.is_tuning and self.is_RT:
            mode_description = "Mode 3: Regular task with mean RT-based fixed windows (stimOn aligned)"
        else:
            mode_description = "Mode 4: Regular task with saccade-aligned fixed windows"
        
        # Debug output
        print(f"Debug: Session {self.date} at position {self.session_position} (relative: {self.relative_position})")
        print(f"Debug: Guidetube depth: {self.session_guidetube_depth}, relative: {self.relative_guidetube_depth:.3f} mm")
        print(f"Debug: {mode_description}")
        print(f"Debug: Alignment: {self.alignment}")
        if self.is_RT and self.mean_RT is not None:
            print(f"Debug: Mean RTs provided: {self.mean_RT}")
        print(f"Debug: Analysis window: {self.analysis_window}ms")
        print(f"Debug: Number of units determined: {self.n_units}")
    
    def _get_n_units(self):
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

    def get_neurometric_windows(self):
        """For Mode 1 (Tuning sliding windows)"""
        try:
            time_axis = self.time_axes[self.alignment]
            bin_size_ms = self.tuning_time_info['bin_size'] * 1000
            window_bins = int(self.neurometric_window_ms / bin_size_ms)
            slide_bins = int(self.neurometric_slide_ms / bin_size_ms)

            t_zero_idx = np.argmin(np.abs(time_axis))
            t_end = time_axis[-1] - 0.2
            t_end_idx = np.argmin(np.abs(time_axis - t_end))
            
            windows = []
            start_idx = t_zero_idx
            
            while start_idx + window_bins <= t_end_idx:
                end_idx = start_idx + window_bins
                start_time_ms = time_axis[start_idx] * 1000
                end_time_ms = time_axis[end_idx] * 1000
                center_time_ms = (start_time_ms + end_time_ms) / 2
                
                windows.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time_ms': start_time_ms,
                    'end_time_ms': end_time_ms,
                    'center_time_ms': center_time_ms
                })
                
                start_idx += slide_bins
            
            return windows
            
        except Exception as e:
            print(f"Error in get_neurometric_windows: {e}")
            return []

    def get_tuning_windows(self):
        """For Mode 1 (Tuning sliding windows)"""
        try:
            time_axis = self.time_axes[self.alignment]
            bin_size_ms = self.tuning_time_info['bin_size'] * 1000
            window_bins = int(self.tuning_window_ms / bin_size_ms)
            slide_bins = int(self.tuning_slide_ms / bin_size_ms)

            t_zero_idx = np.argmin(np.abs(time_axis))
            t_end = time_axis[-1] - 0.2
            t_end_idx = np.argmin(np.abs(time_axis - t_end))
            
            windows = []
            start_idx = t_zero_idx
            
            while start_idx + window_bins <= t_end_idx:
                end_idx = start_idx + window_bins
                start_time_ms = time_axis[start_idx] * 1000
                end_time_ms = time_axis[end_idx] * 1000
                center_time_ms = (start_time_ms + end_time_ms) / 2
                
                windows.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time_ms': start_time_ms,
                    'end_time_ms': end_time_ms,
                    'center_time_ms': center_time_ms
                })
                
                start_idx += slide_bins
            
            return windows
            
        except Exception as e:
            print(f"Error in get_tuning_windows: {e}")
            return []

    def get_time_window_indices_fixed_saccade(self):
        """
        Get fixed time window indices for Mode 4 (Regular task, saccade-aligned)
        Returns window relative to saccade onset (e.g., [-200, 0]ms)
        """
        try:
            time_axis = self.time_axes[self.alignment]
            start_idx = np.argmin(np.abs(time_axis*1000 - self.analysis_window[0] -100)) #yc check point
            end_idx = np.argmin(np.abs(time_axis*1000 - self.analysis_window[1] - 100))
            
            print(f"    Mode 4 window: [{time_axis[start_idx]*1000:.1f}, {time_axis[end_idx]*1000:.1f}]ms relative to saccade")
            
            return start_idx, end_idx
            
        except Exception as e:
            print(f"Error in get_time_window_indices_fixed_saccade: {e}")
            return 0, 10

    def get_time_window_indices_RT(self, modality):
        """
        Get time window indices for Modes 2 & 3 (RT-based)
        Returns fixed window (e.g., [-200, 0]ms) relative to mean RT for each modality
        
        Parameters:
        -----------
        modality : int or str
            For tuning (Mode 2): 1, 2, or 3
            For regular (Mode 3): 'ves', 'vis', 'vis_low', 'comb', 'comb_low'
        
        Returns:
        --------
        tuple : (start_idx, end_idx, center_ms, mean_rt_ms)
        """
        try:
            time_axis = self.time_axes[self.alignment]
            
            # Map modality to RT key
            if self.is_tuning:
                # Mode 2: Tuning RT
                if modality == 1:
                    rt_key = 'mod1_coh1'
                elif modality == 2:
                    rt_key = 'mod2_coh2'
                elif modality == 3:
                    rt_key = 'mod3_coh2'
                else:
                    print(f"Unknown tuning modality: {modality}")
                    return 0, 10, 0, 0
            else:
                # Mode 3: Regular RT
                if modality == 'ves':
                    rt_key = 'mod1_coh1'
                elif modality == 'vis':
                    rt_key = 'mod2_coh2'
                elif modality == 'vis_low':
                    rt_key = 'mod2_coh1'
                elif modality == 'comb':
                    rt_key = 'mod3_coh2'
                elif modality == 'comb_low':
                    rt_key = 'mod3_coh1'
                else:
                    print(f"Unknown regular modality: {modality}")
                    return 0, 10, 0, 0
            
            # Get mean RT for this modality
            if rt_key not in self.mean_RT:
                print(f"Mean RT not found for {rt_key}")
                return 0, 10, 0, 0
            
            # Calculate mean RT in milliseconds
            # For both Mode 2 and Mode 3, alignment is 'stimOn', so we calculate RT from stimulus onset
            mean_rt_ms = (self.mean_RT[rt_key] + self.time_zero_shift_s) * 1000
            
            # Define window: analysis_window (e.g., [-200, 0]) relative to mean RT
            # Example: if mean_RT = 800ms, analysis_window = [-200, 0]
            #          then window is [600ms, 800ms] after stimulus
            window_start_ms = mean_rt_ms + self.analysis_window[0]  # e.g., 800 + (-200) = 600ms
            window_end_ms = mean_rt_ms + self.analysis_window[1]    # e.g., 800 + 0 = 800ms
            
            # Find indices in the time axis
            start_idx = np.argmin(np.abs(time_axis * 1000 - window_start_ms))
            end_idx = np.argmin(np.abs(time_axis * 1000 - window_end_ms))
            
            # Calculate center time for reporting
            center_ms = (window_start_ms + window_end_ms) / 2
            
            # Ensure valid indices
            if start_idx >= end_idx:
                print(f"Invalid time window indices for {rt_key}: start={start_idx}, end={end_idx}")
                return 0, 10, 0, 0
            
            return start_idx, end_idx, center_ms, mean_rt_ms
            
        except Exception as e:
            print(f"Error in get_time_window_indices_RT for modality {modality}: {e}")
            return 0, 10, 0, 0

    def get_baseline_window_tuning(self):
        """For Mode 1 baseline (pre-stimulus)"""
        try:
            time_axis = self.time_axes[self.alignment]
            t_zero_idx = np.argmin(np.abs(time_axis))
            
            start_idx = 0
            end_idx = t_zero_idx
            start_time_ms = time_axis[start_idx] * 1000
            end_time_ms = time_axis[end_idx] * 1000
            center_time_ms = (start_time_ms + end_time_ms) / 2
            
            return {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time_ms': start_time_ms,
                'end_time_ms': end_time_ms,
                'center_time_ms': center_time_ms
            }
            
        except Exception as e:
            print(f"Error in get_baseline_window_tuning: {e}")
            return {'start_idx': 0, 'end_idx': 10, 'center_time_ms': -100}

    def _return_nan_result(self, n_headings, failure_reason):
        return {
            'threshold': np.nan,
            'r_squared': np.nan,
            'fit_mu': np.nan,
            'fit_sigma': np.nan,
            'n_headings': n_headings,
            'failure_reason': failure_reason
        }

    def sort_trials_by_conditions(self):
        try:
            conditions = {}
            headingInds = np.unique(self.behavior['headingInd'])
            modalities_raw = np.unique(self.behavior['modality'])  
            headings = []

            if self.is_tuning:
                if self.date == '20250306':
                    heading_values = {
                        1: -90, 2: -45, 3: -21.5, 4: -10, 5: -3.9, 6: -1.5, 7: 0,
                        8: 1.5, 9: 3.9, 10: 10, 11: 21.5, 12: 45, 13: 90
                    }
                else:
                    heading_values = {
                        1: -45, 2: -21.5, 3: -10, 4: -3.9,
                        5: 3.9, 6: 10, 7: 21.5, 8: 45
                    }
                
                for headingInd in headingInds:
                    if headingInd in heading_values:
                        headings.append(heading_values[headingInd])
                headings = np.array(headings)
                
                # Tuning task: use original modality numbers (NO coherence split)
                modalities = modalities_raw
                for modality in modalities:
                    conditions[modality] = {}
                    for headingInd in headingInds:
                        if headingInd in heading_values:
                            heading = heading_values[headingInd]
                            mask = (self.behavior['headingInd'] == headingInd) & (self.behavior['modality'] == modality)
                            trial_indices = np.where(mask)[0]
                            conditions[modality][heading] = trial_indices
            else:
                # Regular task headings
                heading_values = {
                    1: -10, 2: -3.9, 3: -1.5, 4: 0, 5: 1.5, 6: 3.9, 7: 10
                }

                for headingInd in headingInds:
                    if headingInd in heading_values:
                        headings.append(heading_values[headingInd])
                headings = np.array(headings)
                
                # Regular task: split visual and combined into high and low coherence
                modalities = []
                
                for modality in modalities_raw:
                    if modality == 1:  # Vestibular
                        modalities.append('ves')
                        conditions['ves'] = {}
                        for headingInd in headingInds:
                            if headingInd in heading_values:
                                heading = heading_values[headingInd]
                                mask = (self.behavior['headingInd'] == headingInd) & \
                                    (self.behavior['modality'] == modality) & \
                                    (self.behavior['coherenceInd'] == 1)
                                trial_indices = np.where(mask)[0]
                                conditions['ves'][heading] = trial_indices
                                
                    elif modality == 2:  # Visual
                        if 'vis' not in modalities:
                            modalities.append('vis')
                            modalities.append('vis_low')
                        
                        conditions['vis'] = {}
                        for headingInd in headingInds:
                            if headingInd in heading_values:
                                heading = heading_values[headingInd]
                                mask = (self.behavior['headingInd'] == headingInd) & \
                                    (self.behavior['modality'] == modality) & \
                                    (self.behavior['coherenceInd'] == 2)
                                trial_indices = np.where(mask)[0]
                                conditions['vis'][heading] = trial_indices
                        
                        conditions['vis_low'] = {}
                        for headingInd in headingInds:
                            if headingInd in heading_values:
                                heading = heading_values[headingInd]
                                mask = (self.behavior['headingInd'] == headingInd) & \
                                    (self.behavior['modality'] == modality) & \
                                    (self.behavior['coherenceInd'] == 1)
                                trial_indices = np.where(mask)[0]
                                conditions['vis_low'][heading] = trial_indices
                                
                    elif modality == 3:  # Combined
                        if 'comb' not in modalities:
                            modalities.append('comb')
                            modalities.append('comb_low')
                        
                        conditions['comb'] = {}
                        for headingInd in headingInds:
                            if headingInd in heading_values:
                                heading = heading_values[headingInd]
                                mask = (self.behavior['headingInd'] == headingInd) & \
                                    (self.behavior['modality'] == modality) & \
                                    (self.behavior['coherenceInd'] == 2)
                                trial_indices = np.where(mask)[0]
                                conditions['comb'][heading] = trial_indices
                        
                        conditions['comb_low'] = {}
                        for headingInd in headingInds:
                            if headingInd in heading_values:
                                heading = heading_values[headingInd]
                                mask = (self.behavior['headingInd'] == headingInd) & \
                                    (self.behavior['modality'] == modality) & \
                                    (self.behavior['coherenceInd'] == 1)
                                trial_indices = np.where(mask)[0]
                                conditions['comb_low'][heading] = trial_indices
                        
            return conditions, headings, modalities
            
        except Exception as e:
            print(f"Error in sort_trials_by_conditions: {e}")
            return {}, np.array([]), np.array([])

    def calculate_overall_firing_rate(self, firing_rates):
        if len(firing_rates) > 0:
            return np.mean(firing_rates)
        else:
            return 0.0

    def calculate_anova_for_window(self, unit_idx, modality, start_idx, end_idx):
        """Calculate ANOVA using only headings with 3.9 <= |heading| <= 45"""
        try:
            spike_data = self.spikes_data[self.alignment][unit_idx]
            window_spikes = spike_data[:, start_idx:end_idx]
            firing_rates = np.mean(window_spikes, axis=1)
            
            # Map modality to raw modality number and coherence for filtering
            if self.is_tuning:
                raw_modality = modality
                modality_mask = self.behavior['modality'] == raw_modality
            else:
                # Regular task with coherence split
                if modality == 'ves':
                    modality_mask = (self.behavior['modality'] == 1) & (self.behavior['coherenceInd'] == 1)
                elif modality == 'vis':
                    modality_mask = (self.behavior['modality'] == 2) & (self.behavior['coherenceInd'] == 2)
                elif modality == 'vis_low':
                    modality_mask = (self.behavior['modality'] == 2) & (self.behavior['coherenceInd'] == 1)
                elif modality == 'comb':
                    modality_mask = (self.behavior['modality'] == 3) & (self.behavior['coherenceInd'] == 2)
                elif modality == 'comb_low':
                    modality_mask = (self.behavior['modality'] == 3) & (self.behavior['coherenceInd'] == 1)
                else:
                    return {'F': np.nan, 'p': np.nan, 'selective': 0}
            
            modality_firing_rates = firing_rates[modality_mask]
            modality_heading_inds = self.behavior['headingInd'][modality_mask]
            
            if self.is_tuning and self.date == '20250306':
                heading_values_map = {
                    1: -90, 2: -45, 3: -21.5, 4: -10, 5: -3.9, 6: -1.5, 7: 0,
                    8: 1.5, 9: 3.9, 10: 10, 11: 21.5, 12: 45, 13: 90
                }
            elif self.is_tuning:
                heading_values_map = {
                    1: -45, 2: -21.5, 3: -10, 4: -3.9,
                    5: 3.9, 6: 10, 7: 21.5, 8: 45
                }
            else:
                heading_values_map = {
                    1: -10, 2: -3.9, 3: -1.5, 4: 0, 5: 1.5, 6: 3.9, 7: 10
                }
            
            groups = []
            for heading_ind in np.unique(modality_heading_inds):
                if heading_ind in heading_values_map:
                    heading = heading_values_map[heading_ind]
                    if abs(heading) >= 3.9 and abs(heading) <= 45:
                        heading_rates = modality_firing_rates[modality_heading_inds == heading_ind]
                        if len(heading_rates) > 0:
                            groups.append(heading_rates)
            
            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups)
                return {'F': f_stat, 'p': p_val, 'selective': 1 if p_val < 0.05 else 0}
            
            return {'F': np.nan, 'p': np.nan, 'selective': 0}
            
        except Exception as e:
            return {'F': np.nan, 'p': np.nan, 'selective': 0}

    def calculate_correlation_for_window(self, unit_idx, modality, start_idx, end_idx):
        """Calculate correlation using only headings with 3.9 <= |heading| <= 45"""
        try:
            spike_data = self.spikes_data[self.alignment][unit_idx]
            window_spikes = spike_data[:, start_idx:end_idx]
            firing_rates = np.mean(window_spikes, axis=1)
            
            conditions, headings, _ = self.sort_trials_by_conditions()
            
            if modality not in conditions:
                return {'r': np.nan, 'p': np.nan}
            
            condition_rates = {}
            for heading in headings:
                if abs(heading) >= 3.9 and abs(heading) <= 45:
                    if heading in conditions[modality]:
                        trial_indices = conditions[modality][heading]
                        if len(trial_indices) > 0:
                            condition_rates[heading] = np.mean(firing_rates[trial_indices])
            
            rates = []
            heading_vals = []
            for heading in sorted(condition_rates.keys()):
                if not np.isnan(condition_rates[heading]):
                    rates.append(condition_rates[heading])
                    heading_vals.append(heading)
            
            if len(rates) > 2:
                r, p = pearsonr(rates, heading_vals)
                return {'r': r, 'p': p}
            else:
                return {'r': np.nan, 'p': np.nan}
                
        except Exception as e:
            return {'r': np.nan, 'p': np.nan}

    def get_unit_area(self, unit_idx):
        if unit_idx in self.units_data.get('MST', []):
            return 'MST'
        elif unit_idx in self.units_data.get('VPS', []):
            return 'VPS'
        elif unit_idx in self.units_data.get('dual', []):
            return 'MT'
        else:
            return 'unknown'
        
    def calculate_relative_positions(self):
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
            
            depths_mm = depths / 1000.0
            relative_depths = self.relative_guidetube_depth - depths_mm
            
            return {
                'absolute_x': np.full(self.n_units, self.session_position[0]),
                'absolute_y': np.full(self.n_units, self.session_position[1]),
                'relative_x': np.full(self.n_units, self.relative_position[0]),
                'relative_y': np.full(self.n_units, self.relative_position[1]),
                'absolute_depth': depths_mm,
                'relative_depth': relative_depths
            }
        except Exception as e:
            print(f"Error in calculate_relative_positions: {e}")
            return {
                'absolute_x': np.full(self.n_units, self.session_position[0]),
                'absolute_y': np.full(self.n_units, self.session_position[1]),
                'relative_x': np.full(self.n_units, self.relative_position[0]),
                'relative_y': np.full(self.n_units, self.relative_position[1]),
                'absolute_depth': np.zeros(self.n_units),
                'relative_depth': np.zeros(self.n_units)
            }

    def calculate_roc_area(self, responses1, responses2):
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
        if sigma <= 0:
            sigma = 1e-6
        return stats.norm.cdf(x, loc=mu, scale=sigma)

    def calculate_neurometric_threshold_window(self, unit_idx, modality, start_idx, end_idx):
        """Calculate neurometric threshold using headings where 0 < |heading| <= 40"""
        try:
            spike_data = self.spikes_data[self.alignment][unit_idx]
            window_spikes = spike_data[:, start_idx:end_idx]
            trial_firing_rates = np.mean(window_spikes, axis=1)
            heading_n_trials = []
            
            # Get heading values map based on task type
            if self.is_tuning and self.date == '20250306':
                heading_values_map = {
                    1: -90, 2: -45, 3: -21.5, 4: -10, 5: -3.9, 6: -1.5, 7: 0,
                    8: 1.5, 9: 3.9, 10: 10, 11: 21.5, 12: 45, 13: 90
                }
            elif self.is_tuning:
                heading_values_map = {
                    1: -45, 2: -21.5, 3: -10, 4: -3.9,
                    5: 3.9, 6: 10, 7: 21.5, 8: 45
                }
            else:
                heading_values_map = {
                    1: -10, 2: -3.9, 3: -1.5, 4: 0, 5: 1.5, 6: 3.9, 7: 10
                }
            
            # Map modality to raw modality number and coherence for filtering
            if self.is_tuning:
                raw_modality = modality
                modality_mask = self.behavior['modality'] == raw_modality
            else:
                if modality == 'ves':
                    modality_mask = (self.behavior['modality'] == 1) & (self.behavior['coherenceInd'] == 1)
                elif modality == 'vis':
                    modality_mask = (self.behavior['modality'] == 2) & (self.behavior['coherenceInd'] == 2)
                elif modality == 'vis_low':
                    modality_mask = (self.behavior['modality'] == 2) & (self.behavior['coherenceInd'] == 1)
                elif modality == 'comb':
                    modality_mask = (self.behavior['modality'] == 3) & (self.behavior['coherenceInd'] == 2)
                elif modality == 'comb_low':
                    modality_mask = (self.behavior['modality'] == 3) & (self.behavior['coherenceInd'] == 1)
                else:
                    return self._return_nan_result(0, "unknown_modality")
            
            modality_heading_inds = self.behavior['headingInd'][modality_mask]
            modality_rates = trial_firing_rates[modality_mask]
            
            # Convert heading indices to actual heading values
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
            
            # Calculate ROC for opposing headings
            roc_values = []
            heading_values = []
            sorted_headings = np.sort(unique_headings)
            
            # TASK-SPECIFIC HEADING FILTERING (this part stays different)
            # Tuning: uses ±3.9°, ±10°, ±21.5°, ±45° (larger range)
            # Regular: uses ±1.5°, ±3.9°, ±10° (smaller range)
            for heading in sorted_headings:
                if heading <= 0 or heading > 50:
                    continue  

                opposite_heading = -heading
                
                if opposite_heading in unique_headings:
                    heading_responses = modality_rates[modality_headings == heading]
                    opposite_responses = modality_rates[modality_headings == opposite_heading]
                    
                    if len(heading_responses) > 0 and len(opposite_responses) > 0:

                        roc_area = self.calculate_roc_area(heading_responses, opposite_responses)

                        roc_values.append(roc_area)
                        heading_values.append(heading)
                        roc_values.append(1 - roc_area)
                        heading_values.append(-heading)

                        n_total = len(heading_responses) + len(opposite_responses)
                        heading_n_trials.extend([n_total, n_total]) # mirrored points share the same N
        
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
                # ============================================
                # BEST SOLUTION: Use ALL initial guesses for BOTH tasks
                # This ensures identical procedure and maximum robustness
                # ============================================
                initial_guesses = [
                    [0.0, 2.5],    # Good for fine discrimination (regular task)
                    [0.0, 5],
                    [0.0, 10],   # Good for moderate tuning
                    [0.0, 40]  # Good for broad tuning

                ]

                # ============================================
                # Bounds for both tasks
                # ============================================
                mu_lower = -1e-6     
                mu_upper = 1e-6   
                sigma_lower = 0.1       # SAME for both tasks
                sigma_upper = 300        # SAME for both tasks
                threshold_upper = 300    # SAME for both tasks
                
                bounds = ([mu_lower, sigma_lower], [mu_upper, sigma_upper])
                
                best_fit = None
                best_r_squared = -np.inf
                
                # Try ALL initial guesses and keep the best fit
                for i, p0 in enumerate(initial_guesses):
                    try:
                        p0_clipped = [
                            np.clip(p0[0], mu_lower, mu_upper),
                            np.clip(p0[1], sigma_lower, sigma_upper)
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

                        if (threshold_84 > sigma_lower and threshold_84 < threshold_upper and
                            sigma_fit > sigma_lower and sigma_fit < sigma_upper and
                            r_squared > -3 and
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

    def calculate_baseline_results(self, unit_idx, modalities):
        """Calculate baseline (pre-stimulus) results for all measures (tuning task Mode 1 only)"""
        baseline_results = {}
        
        try:
            if not self.is_tuning or self.is_RT:
                return baseline_results
            
            baseline_window = self.get_baseline_window_tuning()
            baseline_start_idx = baseline_window['start_idx']
            baseline_end_idx = baseline_window['end_idx']
            baseline_center_ms = baseline_window['center_time_ms']
            
            spike_data = self.spikes_data[self.alignment][unit_idx]
            baseline_spikes = spike_data[:, baseline_start_idx:baseline_end_idx]
            baseline_firing_rates = np.mean(baseline_spikes, axis=1)
            
            conditions, headings, _ = self.sort_trials_by_conditions()
            
            for modality in modalities:
                modality_name = {1.0: 'ves', 2.0: 'vis', 3.0: 'comb'}.get(modality, str(modality))
                
                neurometric = self.calculate_neurometric_threshold_window(
                    unit_idx, modality, baseline_start_idx, baseline_end_idx
                )
                baseline_results[f'{modality_name}_baseline_neurometric_threshold'] = neurometric['threshold']
                baseline_results[f'{modality_name}_baseline_neurometric_r2'] = neurometric['r_squared']
                baseline_results[f'{modality_name}_baseline_neurometric_mu'] = neurometric['fit_mu']
                baseline_results[f'{modality_name}_baseline_neurometric_sigma'] = neurometric['fit_sigma']
                
                anova = self.calculate_anova_for_window(unit_idx, modality, baseline_start_idx, baseline_end_idx)
                baseline_results[f'{modality_name}_baseline_anova_F'] = anova['F']
                baseline_results[f'{modality_name}_baseline_anova_p'] = anova['p']
                
                correlation = self.calculate_correlation_for_window(unit_idx, modality, baseline_start_idx, baseline_end_idx)
                baseline_results[f'{modality_name}_baseline_correlation_r'] = correlation['r']
                baseline_results[f'{modality_name}_baseline_correlation_p'] = correlation['p']
            
            baseline_results['baseline_center_ms'] = baseline_center_ms
            
        except Exception as e:
            print(f"Error in calculate_baseline_results: {e}")
        
        return baseline_results

    def analyze_all_units(self):
        print(f"\nAnalyzing units for {self.subject} {self.date} with alignment {self.alignment}")
        
        # Determine mode
        if self.is_tuning and not self.is_RT:
            mode_num = 1
            mode_desc = "Tuning with sliding windows"
        elif self.is_tuning and self.is_RT:
            mode_num = 2
            mode_desc = "Tuning with mean RT-based windows"
        elif not self.is_tuning and self.is_RT:
            mode_num = 3
            mode_desc = "Regular task with mean RT-based windows (stimOn aligned)"
        else:
            mode_num = 4
            mode_desc = "Regular task with saccade-aligned windows"
        
        print(f"MODE {mode_num}: {mode_desc}")
        print(f"Analysis window: {self.analysis_window}ms")
        
        try:
            positions = self.calculate_relative_positions()
            conditions, headings, modalities = self.sort_trials_by_conditions()
            
            print(f"Modalities to analyze: {modalities}")
            print(f"Processing {self.n_units} units...")
            
            for unit_idx in range(self.n_units):
                try:    
                    if unit_idx % 10 == 0 or unit_idx < 5:
                        print(f"  Processing unit {unit_idx + 1}/{self.n_units}")

                    # Convert numpy type to Python int
                    real_cluster_id = int(self.unit_info['cluster_id'][unit_idx])
          
                    unit_area = self.get_unit_area(unit_idx)
                    
                    features = {
                        'unit_id': real_cluster_id,
                        'subject': self.subject,
                        'date': self.date,
                        'alignment': self.alignment,
                        'is_tuning': self.is_tuning,
                        'is_RT': self.is_RT,
                        'mode': mode_num,
                        'unit_idx': unit_idx,
                        'area': unit_area,
                        'absolute_x': positions['absolute_x'][unit_idx],
                        'absolute_y': positions['absolute_y'][unit_idx],
                        'relative_x': positions['relative_x'][unit_idx],
                        'relative_y': positions['relative_y'][unit_idx],
                        'absolute_depth': positions['absolute_depth'][unit_idx],
                        'relative_depth': positions['relative_depth'][unit_idx],
                    }
                    
                    # ============================================================
                    # MODE 1: Tuning with sliding windows
                    # ============================================================
                    if mode_num == 1:
                        neurometric_windows = self.get_neurometric_windows()
                        tuning_windows = self.get_tuning_windows()
                        
                        baseline_results = self.calculate_baseline_results(unit_idx, modalities)
                        baseline_center = int(baseline_results.get('baseline_center_ms', -100))
                        
                        for modality in modalities:
                            modality_name = {1.0: 'ves', 2.0: 'vis', 3.0: 'comb'}.get(modality, str(modality))
                            
                            # Neurometric analysis
                            neurometric_centers = [baseline_center]
                            neurometric_thresholds = [baseline_results.get(f'{modality_name}_baseline_neurometric_threshold', np.nan)]
                            neurometric_r2s = [baseline_results.get(f'{modality_name}_baseline_neurometric_r2', np.nan)]
                            neurometric_mus = [baseline_results.get(f'{modality_name}_baseline_neurometric_mu', np.nan)]
                            neurometric_sigmas = [baseline_results.get(f'{modality_name}_baseline_neurometric_sigma', np.nan)]
                            
                            best_neuro_score = -np.inf
                            best_neuro_center = None
                            
                            for window in neurometric_windows:
                                center_ms = int(window['center_time_ms'])
                                neurometric = self.calculate_neurometric_threshold_window(
                                    unit_idx, modality, window['start_idx'], window['end_idx']
                                )
                                
                                neurometric_centers.append(center_ms)
                                neurometric_thresholds.append(neurometric['threshold'])
                                neurometric_r2s.append(neurometric['r_squared'])
                                neurometric_mus.append(neurometric['fit_mu'])
                                neurometric_sigmas.append(neurometric['fit_sigma'])
                                
                                if not np.isnan(neurometric['threshold']) and neurometric['r_squared'] > 0.1:
                                    score = 1 / (neurometric['threshold'] + 1e-6)
                                    if score > best_neuro_score:
                                        best_neuro_score = score
                                        best_neuro_center = center_ms
                            
                            features[f'neurometric_{modality_name}_centers'] = neurometric_centers
                            features[f'neurometric_{modality_name}_thresholds'] = neurometric_thresholds
                            features[f'neurometric_{modality_name}_r2'] = neurometric_r2s
                            features[f'neurometric_{modality_name}_mu'] = neurometric_mus
                            features[f'neurometric_{modality_name}_sigma'] = neurometric_sigmas
                            features[f'neurometric_{modality_name}_best_center_ms'] = best_neuro_center
                            
                            # ANOVA analysis
                            anova_centers = [baseline_center]
                            anova_Fs = [baseline_results.get(f'{modality_name}_baseline_anova_F', np.nan)]
                            anova_ps = [baseline_results.get(f'{modality_name}_baseline_anova_p', np.nan)]
                            
                            best_anova_score = -np.inf
                            best_anova_center = None
                            
                            for window in tuning_windows:
                                center_ms = int(window['center_time_ms'])
                                anova = self.calculate_anova_for_window(
                                    unit_idx, modality, window['start_idx'], window['end_idx']
                                )
                                
                                anova_centers.append(center_ms)
                                anova_Fs.append(anova['F'])
                                anova_ps.append(anova['p'])
                                
                                if not np.isnan(anova['F']):
                                    if anova['F'] > best_anova_score:
                                        best_anova_score = anova['F']
                                        best_anova_center = center_ms
                            
                            features[f'anova_{modality_name}_centers'] = anova_centers
                            features[f'anova_{modality_name}_F'] = anova_Fs
                            features[f'anova_{modality_name}_p'] = anova_ps
                            features[f'anova_{modality_name}_best_center_ms'] = best_anova_center
                            
                            # Correlation analysis
                            corr_centers = [baseline_center]
                            corr_rs = [baseline_results.get(f'{modality_name}_baseline_correlation_r', np.nan)]
                            corr_ps = [baseline_results.get(f'{modality_name}_baseline_correlation_p', np.nan)]
                            
                            best_corr_score = -np.inf
                            best_corr_center = None
                            
                            for window in tuning_windows:
                                center_ms = int(window['center_time_ms'])
                                correlation = self.calculate_correlation_for_window(
                                    unit_idx, modality, window['start_idx'], window['end_idx']
                                )
                                
                                corr_centers.append(center_ms)
                                corr_rs.append(correlation['r'])
                                corr_ps.append(correlation['p'])
                                
                                if not np.isnan(correlation['r']):
                                    if abs(correlation['r']) > best_corr_score:
                                        best_corr_score = abs(correlation['r'])
                                        best_corr_center = center_ms
                            
                            features[f'correlation_{modality_name}_centers'] = corr_centers
                            features[f'correlation_{modality_name}_r'] = corr_rs
                            features[f'correlation_{modality_name}_p'] = corr_ps
                            features[f'correlation_{modality_name}_best_center_ms'] = best_corr_center
                        
                        # Calculate overall firing rate from best windows
                        overall_rates = []
                        for modality in modalities:
                            modality_name = {1.0: 'ves', 2.0: 'vis', 3.0: 'comb'}.get(modality, str(modality))
                            best_center = features.get(f'neurometric_{modality_name}_best_center_ms')
                            if best_center is not None:
                                centers = features[f'neurometric_{modality_name}_centers']
                                if best_center in centers:
                                    for window in neurometric_windows:
                                        if int(window['center_time_ms']) == best_center:
                                            spike_data = self.spikes_data[self.alignment][unit_idx]
                                            window_spikes = spike_data[:, window['start_idx']:window['end_idx']]
                                            overall_rates.append(np.mean(window_spikes))
                                            break
                        
                        features['overall_firing_rate'] = np.mean(overall_rates) if len(overall_rates) > 0 else 0.0
                    
                    # ============================================================
                    # MODE 2: Tuning with mean RT-based windows
                    # ============================================================
                    elif mode_num == 2:
                        if unit_idx < 5:
                            print(f"    Mode 2: Using RT-based windows (stimOn aligned)")
                        
                        for modality in modalities:
                            modality_name = {1.0: 'ves', 2.0: 'vis', 3.0: 'comb'}.get(modality, str(modality))
                            
                            # Get RT-specific window for this modality
                            start_idx, end_idx, center_ms, mean_rt_ms = self.get_time_window_indices_RT(modality)
                            
                            if unit_idx < 5:
                                w0, w1 = self.analysis_window
                                print(f"{modality}: meanRT={mean_rt_ms:.1f}ms, window=[{mean_rt_ms+w0:.1f}, {mean_rt_ms+w1:.1f}]ms (center {center_ms:.1f})")
                            
                            # Neurometric threshold
                            neurometric = self.calculate_neurometric_threshold_window(
                                unit_idx, modality, start_idx, end_idx
                            )
                            
                            features[f'neurometric_{modality_name}_centers'] = [int(center_ms)]
                            features[f'neurometric_{modality_name}_thresholds'] = [neurometric['threshold']]
                            features[f'neurometric_{modality_name}_r2'] = [neurometric['r_squared']]
                            features[f'neurometric_{modality_name}_mu'] = [neurometric['fit_mu']]
                            features[f'neurometric_{modality_name}_sigma'] = [neurometric['fit_sigma']]
                            features[f'neurometric_{modality_name}_best_center_ms'] = int(center_ms)
                            features[f'neurometric_{modality_name}_mean_rt_ms'] = mean_rt_ms
                            
                            # ANOVA
                            anova = self.calculate_anova_for_window(unit_idx, modality, start_idx, end_idx)
                            features[f'anova_{modality_name}_centers'] = [int(center_ms)]
                            features[f'anova_{modality_name}_F'] = [anova['F']]
                            features[f'anova_{modality_name}_p'] = [anova['p']]
                            features[f'anova_{modality_name}_best_center_ms'] = int(center_ms)
                            
                            # Correlation
                            correlation = self.calculate_correlation_for_window(unit_idx, modality, start_idx, end_idx)
                            features[f'correlation_{modality_name}_centers'] = [int(center_ms)]
                            features[f'correlation_{modality_name}_r'] = [correlation['r']]
                            features[f'correlation_{modality_name}_p'] = [correlation['p']]
                            features[f'correlation_{modality_name}_best_center_ms'] = int(center_ms)
                        
                        features['overall_firing_rate'] = 0.0
                    
                    # ============================================================
                    # MODE 3: Regular with mean RT-based windows (stimOn aligned)
                    # ============================================================
                    elif mode_num == 3:
                        if unit_idx < 5:
                            print(f"    Mode 3: Using RT-based windows (stimOn aligned)")
                        
                        spike_data = self.spikes_data[self.alignment][unit_idx]
                        
                        for modality in modalities:
                            # Get RT-specific window for this modality
                            start_idx, end_idx, center_ms, mean_rt_ms = self.get_time_window_indices_RT(modality)
                            
                            if unit_idx < 5:
                                print(f"      {modality}: mean_RT={mean_rt_ms:.1f}ms, window center={center_ms:.1f}ms")
                            
                            # Neurometric threshold
                            neurometric = self.calculate_neurometric_threshold_window(
                                unit_idx, modality, start_idx, end_idx
                            )
                            features[f'neurometric_{modality}_thresholds'] = [neurometric['threshold']]
                            features[f'neurometric_{modality}_r2'] = [neurometric['r_squared']]
                            features[f'neurometric_{modality}_mu'] = [neurometric['fit_mu']]
                            features[f'neurometric_{modality}_sigma'] = [neurometric['fit_sigma']]
                            features[f'neurometric_{modality}_best_center_ms'] = int(center_ms)
                            features[f'neurometric_{modality}_center_ms'] = int(center_ms)
                            features[f'neurometric_{modality}_mean_rt_ms'] = mean_rt_ms
                            
                            # ANOVA
                            anova = self.calculate_anova_for_window(unit_idx, modality, start_idx, end_idx)
                            features[f'anova_{modality}_F'] = [anova['F']]
                            features[f'anova_{modality}_p'] = [anova['p']]
                            features[f'anova_{modality}_best_center_ms'] = int(center_ms)
                            features[f'anova_{modality}_center_ms'] = int(center_ms)
                            
                            # Correlation
                            correlation = self.calculate_correlation_for_window(unit_idx, modality, start_idx, end_idx)
                            features[f'correlation_{modality}_r'] = [correlation['r']]
                            features[f'correlation_{modality}_p'] = [correlation['p']]
                            features[f'correlation_{modality}_best_center_ms'] = int(center_ms)
                            features[f'correlation_{modality}_center_ms'] = int(center_ms)
                        
                        features['overall_firing_rate'] = 0.0
                    
                    # ============================================================
                    # MODE 4: Regular with saccade-aligned fixed windows
                    # ============================================================
                    else:  # mode_num == 4
                        time_axis = self.time_axes[self.alignment]
                        start_idx, end_idx = self.get_time_window_indices_fixed_saccade()
                        center_ms = int((time_axis[start_idx] + time_axis[end_idx]) / 2 * 1000)
                        
                        spike_data = self.spikes_data[self.alignment][unit_idx]
                        window_spikes = spike_data[:, start_idx:end_idx]
                        firing_rates = np.mean(window_spikes, axis=1)
                        
                        features['overall_firing_rate'] = np.mean(firing_rates)
                        
                        features['neurometric_centers'] = [center_ms]
                        features['anova_centers'] = [center_ms]
                        features['correlation_centers'] = [center_ms]
                        
                        for modality in modalities:
                            neurometric = self.calculate_neurometric_threshold_window(
                                unit_idx, modality, start_idx, end_idx
                            )
                            features[f'neurometric_{modality}_thresholds'] = [neurometric['threshold']]
                            features[f'neurometric_{modality}_r2'] = [neurometric['r_squared']]
                            features[f'neurometric_{modality}_mu'] = [neurometric['fit_mu']]
                            features[f'neurometric_{modality}_sigma'] = [neurometric['fit_sigma']]
                            features[f'neurometric_{modality}_best_center_ms'] = center_ms
                            
                            anova = self.calculate_anova_for_window(unit_idx, modality, start_idx, end_idx)
                            features[f'anova_{modality}_F'] = [anova['F']]
                            features[f'anova_{modality}_p'] = [anova['p']]
                            features[f'anova_{modality}_best_center_ms'] = center_ms
                            
                            correlation = self.calculate_correlation_for_window(unit_idx, modality, start_idx, end_idx)
                            features[f'correlation_{modality}_r'] = [correlation['r']]
                            features[f'correlation_{modality}_p'] = [correlation['p']]
                            features[f'correlation_{modality}_best_center_ms'] = center_ms
                    
                    self.unit_features.append(features)
                    
                except Exception as e:
                    if unit_idx < 5:
                        print(f"    Error analyzing unit {unit_idx}: {e}")
                    continue
            
            print(f"\nSuccessfully analyzed {len(self.unit_features)} units")
            
        except Exception as e:
            print(f"Error in analyze_all_units: {e}")
            import traceback
            traceback.print_exc()

    def get_features_dataframe(self):
        df = pd.DataFrame(self.unit_features)
        
        for col in df.columns:
            if df[col].dtype == object:
                if isinstance(df[col].iloc[0] if len(df) > 0 else None, list):
                    df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
        
        return df

    def save_features(self, filename=None, save_dir=None):
        if save_dir is None:
            if self.is_tuning:
                save_dir = r'D:\Neural-Pipeline\results\analysis_single_neurons\dots3DMPtuning_neuralfeatures'
            else:
                save_dir = r'D:\Neural-Pipeline\results\analysis_single_neurons\dots3DMP_neuralfeatures'
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if filename is None:
            if self.is_tuning:
                task_suffix = 'tuning_RT' if self.is_RT else 'tuning'
            else:
                task_suffix = 'regular_RT' if self.is_RT else 'regular'
            filename = f"{self.subject}_{self.date}_{self.alignment}_{task_suffix}_neural_features.csv"
        
        full_path = os.path.join(save_dir, filename)
        df = self.get_features_dataframe()
        df.to_csv(full_path, index=False)
        print(f"Features saved to {full_path}")
        
        return full_path