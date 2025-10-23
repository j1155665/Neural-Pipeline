import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from dPCA.dPCA import dPCA

class SlidingWindowdPCA:
    
    def __init__(self, subject, date, save_dir=r'D:\Neural-Pipeline\results\analysis_population\dpca_results'):
        self.subject = subject
        self.date = date
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.setup_dpca()

    def setup_dpca(self, brain_area=None):
        """Setup dPCA parameters"""
        
        # dPCA specific parameters
        self.dpca_params = {
            'labels': 'sd',  # stimulus (s) and decision/choice (d)
            'regularizer': None,
            'n_components': 15  # Number of components to keep
        }

    def prepare_dpca_data(self, spikes, behavior, valid_units, 
                        trial_mod=3, trial_coh=2, trial_del=0,
                        min_trials_per_condition=3):  # Reduced from 5 to 3
        """
        Prepare data for dPCA analysis using real heading values
        
        Parameters:
        -----------
        spikes : array, shape (n_neurons, n_trials, n_time)
        behavior : dict with behavioral data
        valid_units : array, boolean mask for valid units
        min_trials_per_condition : int, minimum trials needed per condition
        
        Returns:
        --------
        dpca_data : array, shape (n_neurons, n_time, n_stimuli, n_choices)
        condition_info : dict with condition details
        """
        
        if valid_units is not None:
            spikes = spikes[valid_units, :, :]
        
        # Filter trials by modality, coherence, and delta
        if isinstance(trial_del, (list, np.ndarray)):
            del_mask = np.isin(behavior['delta'], trial_del)
        else:
            del_mask = (behavior['delta'] == trial_del)
            
        mask = (behavior['modality'] == trial_mod) & (behavior['coherenceInd'] == trial_coh) & del_mask
        
        # Apply mask
        filtered_spikes = spikes[:, mask, :]
        filtered_behavior = {key: val[mask] for key, val in behavior.items()}
        
        # Use actual heading values instead of indices
        stimulus_conditions = filtered_behavior['heading']  # Real headings (e.g., -10, -5, 0, 5, 10)
        choice_conditions = (filtered_behavior['choice'].astype(int) - 1)  # 0-1 (convert from 1-2)
        
        # Get dimensions
        n_neurons, n_trials, n_time = filtered_spikes.shape
        unique_headings = np.sort(np.unique(stimulus_conditions))
        n_stimuli = len(unique_headings)
        n_choices = 2  # left/right choices
        
        # Initialize dPCA data array
        dpca_data = np.full((n_neurons, n_time, n_stimuli, n_choices), np.nan)
        trial_counts = np.zeros((n_stimuli, n_choices), dtype=int)
        
        # Organize data by conditions and average across trials
        print("Organizing data by heading and choice conditions...")
        
        for stim_idx, heading in enumerate(unique_headings):  # FIXED: proper enumerate usage
            for choice_idx in range(2):  # choice indices 0-1
                
                # Find trials matching this condition - FIXED: use heading value, not index
                condition_mask = (stimulus_conditions == heading) & (choice_conditions == choice_idx)
                n_condition_trials = np.sum(condition_mask)
                
                if n_condition_trials >= min_trials_per_condition:
                    # Average across trials for this condition
                    condition_spikes = filtered_spikes[:, condition_mask, :]  # (n_neurons, n_condition_trials, n_time)
                    mean_response = np.mean(condition_spikes, axis=1)  # (n_neurons, n_time)
                    
                    # Store in dPCA array - FIXED: use stim_idx directly (already 0-indexed)
                    dpca_data[:, :, stim_idx, choice_idx] = mean_response
                    trial_counts[stim_idx, choice_idx] = n_condition_trials
                
                # FIXED: show actual heading value
                choice_label = "Left" if choice_idx == 0 else "Right"
                print(f"  Heading {heading:+.1f}°, Choice {choice_label}: {n_condition_trials} trials")
        
        # Handle missing conditions - IMPROVED
        nan_mask = np.isnan(dpca_data)
        if np.any(nan_mask):
            n_missing = np.sum(nan_mask)
            print(f"Found {n_missing} missing condition combinations")
            
            # Check how many complete conditions we have
            complete_conditions = 0
            for s in range(n_stimuli):
                for d in range(n_choices):
                    if not np.any(np.isnan(dpca_data[:, :, s, d])):
                        complete_conditions += 1
            
            print(f"Complete conditions: {complete_conditions}/{n_stimuli * n_choices}")
            
            if complete_conditions < 4:  # Need at least 4 complete conditions
                raise ValueError(f"Too few complete conditions ({complete_conditions}). "
                            f"Try different parameters or reduce min_trials_per_condition.")
            
            print("Filling missing conditions with mean across available conditions...")
            
            # Fill missing values more carefully
            for n in range(n_neurons):
                for t in range(n_time):
                    slice_data = dpca_data[n, t, :, :]
                    if np.any(np.isnan(slice_data)):
                        # Use mean of available conditions for this neuron and time
                        available_mean = np.nanmean(slice_data)
                        if not np.isnan(available_mean):
                            dpca_data[n, t, :, :] = np.where(np.isnan(dpca_data[n, t, :, :]), 
                                                        available_mean, 
                                                        dpca_data[n, t, :, :])
                        else:
                            # If still NaN, use overall mean for this neuron
                            neuron_mean = np.nanmean(dpca_data[n, :, :, :])
                            dpca_data[n, t, :, :] = np.where(np.isnan(dpca_data[n, t, :, :]), 
                                                        neuron_mean if not np.isnan(neuron_mean) else 0, 
                                                        dpca_data[n, t, :, :])
        
        condition_info = {
            'trial_counts': trial_counts,
            'stimulus_labels': unique_headings,  # Now contains real heading values!
            'choice_labels': ['Left', 'Right'],
            'total_trials': n_trials,
            'valid_conditions': np.sum(trial_counts >= min_trials_per_condition),
            'trial_mod': trial_mod,
            'trial_coh': trial_coh,
            'trial_del': trial_del,
            'heading_range': (np.min(unique_headings), np.max(unique_headings)),
            'n_headings': len(unique_headings)
        }
        
        return dpca_data, condition_info
        

    def zscore_normalize_data(self, dpca_data):
        """
        Simple z-score normalization using scipy
        """
        from scipy import stats
        
        print("Applying z-score normalization across all conditions...")
        
        n_neurons, n_time, n_stimuli, n_choices = dpca_data.shape
        dpca_data_zscore = np.zeros_like(dpca_data)
        
        # Z-score each neuron independently
        for neuron_idx in range(n_neurons):
            neuron_data = dpca_data[neuron_idx, :, :, :]  # (time, stimuli, choices)
            
            # Flatten to 1D, apply zscore, then reshape back
            flat_data = neuron_data.flatten()
            
            # Handle NaN values
            valid_mask = ~np.isnan(flat_data)
            if np.sum(valid_mask) > 1:  # Need at least 2 points for std
                flat_zscore = np.full_like(flat_data, np.nan)
                flat_zscore[valid_mask] = stats.zscore(flat_data[valid_mask], nan_policy='omit')
                dpca_data_zscore[neuron_idx, :, :, :] = flat_zscore.reshape(neuron_data.shape)
            else:
                print(f"Warning: Neuron {neuron_idx} has insufficient valid data")
                dpca_data_zscore[neuron_idx, :, :, :] = 0
        
        zscore_info = {
            'method': 'scipy_zscore',
            'final_data_mean': np.nanmean(dpca_data_zscore),
            'final_data_std': np.nanstd(dpca_data_zscore),
            'final_data_range': (np.nanmin(dpca_data_zscore), np.nanmax(dpca_data_zscore))
        }
        
        print(f"Z-score normalization completed:")
        print(f"  Final data mean: {zscore_info['final_data_mean']:.6f}")
        print(f"  Final data std: {zscore_info['final_data_std']:.6f}")
        
        return dpca_data_zscore, zscore_info

    def run_dpca_analysis(self, spikes_data, behavior_data, time_axes, area, 
                        train_mod=3, train_coh=2, train_delta=0,
                        valid_units=None, save_results=True):
        """
        Run time-resolved dPCA analysis - FIXED VERSION
        """
        
        print(f"\n{'='*60}")
        print(f"RUNNING TIME-RESOLVED dPCA ANALYSIS")
        print(f"Subject: {self.subject}, Date: {self.date}")
        print(f"Area: {area}, Condition: mod{train_mod}_coh{train_coh}_del{train_delta}")
        print(f"{'='*60}")
        
        all_alignment_results = {}
        
        # Loop through alignments
        for alignment in ['stimOn', 'saccOnset', 'postTargHold']:
            if alignment not in spikes_data:
                print(f"Skipping {alignment} - not found in spikes_data")
                continue
                
            print(f"\n{'-'*40}")
            print(f"Processing alignment: {alignment}")
            print(f"{'-'*40}")
            
            try:
                spikes = spikes_data[alignment]
                time_axis = time_axes[alignment]
                
                # Prepare data for dPCA
                dpca_data, condition_info = self.prepare_dpca_data(
                    spikes, behavior_data, valid_units,
                    trial_mod=train_mod, trial_coh=train_coh, trial_del=train_delta
                )
                
                if dpca_data.size == 0:
                    print(f"No valid data for {alignment}, skipping...")
                    continue
                
                print(f"dPCA data shape: {dpca_data.shape}")
                print(f"Valid conditions: {condition_info['valid_conditions']}/14")
                
                # Z-score normalize
                dpca_data_processed, zscore_info = self.zscore_normalize_data(dpca_data)
                
                # Run time-resolved dPCA
                print("Running time-resolved dPCA...")
                n_neurons, n_time, n_stimuli, n_choices = dpca_data_processed.shape
                
                # Initialize storage for time-resolved results - FIXED
                max_components = min(5, n_stimuli-1, n_choices-1)  # Ensure we don't exceed data dimensions
                
                time_resolved_Z = {
                    's': np.zeros((n_time, max_components)),
                    'd': np.zeros((n_time, max_components)),
                    'sd': np.zeros((n_time, max_components))
                }
                
                time_resolved_var = {
                    's': np.zeros(n_time),
                    'd': np.zeros(n_time), 
                    'sd': np.zeros(n_time)
                }
                
                print(f"Processing {n_time} time points...")
                successful_timepoints = 0
                
                for t_idx in range(n_time):
                    if t_idx % 10 == 0:
                        print(f"  Time point {t_idx+1}/{n_time} ({time_axis[t_idx]:.3f}s)")
                    
                    # Extract data for this time point
                    time_point_data = dpca_data_processed[:, t_idx, :, :]
                    
                    # Check if this time point has valid data
                    if np.any(np.isnan(time_point_data)) or np.all(time_point_data == 0):
                        continue
                    
                    try:
                        # Run dPCA for this time point
                        dpca_model = dPCA(labels='sd', regularizer=None)
                        
                        # Fit and transform for this time point
                        Z_t = dpca_model.fit_transform(time_point_data)
                        var_explained_t = dpca_model.explained_variance_ratio_
                        
                        # Store results for this time point - FIXED
                        for component in ['s', 'd', 'sd']:
                            if component in Z_t and len(Z_t[component]) > 0:
                                # Handle different Z_t structures
                                z_component = Z_t[component]
                                
                                if z_component.ndim == 3:
                                    # Shape: (n_components, n_stimuli, n_choices)
                                    # Take first component, average across conditions
                                    if z_component.shape[0] > 0:
                                        time_resolved_Z[component][t_idx, 0] = np.mean(z_component[0])
                                elif z_component.ndim == 2:
                                    # Shape: (n_components, n_conditions)
                                    if z_component.shape[0] > 0:
                                        time_resolved_Z[component][t_idx, 0] = np.mean(z_component[0])
                                elif z_component.ndim == 1:
                                    # Shape: (n_components,)
                                    if len(z_component) > 0:
                                        time_resolved_Z[component][t_idx, 0] = z_component[0]
                                
                                # Store variance explained - FIXED
                                if component in var_explained_t:
                                    var_val = var_explained_t[component]
                                    if isinstance(var_val, (list, np.ndarray)):
                                        time_resolved_var[component][t_idx] = np.sum(var_val) if len(var_val) > 0 else 0
                                    else:
                                        time_resolved_var[component][t_idx] = var_val
                        
                        successful_timepoints += 1
                        
                    except Exception as e:
                        print(f"    dPCA failed at time point {t_idx} ({time_axis[t_idx]:.3f}s): {e}")
                        continue
                
                print(f"Time-resolved dPCA completed! Successful timepoints: {successful_timepoints}/{n_time}")
                
                # Calculate average variance explained
                avg_var_explained = {}
                for component in ['s', 'd', 'sd']:
                    avg_var_explained[component] = np.mean(time_resolved_var[component])
                
                print("\nAverage variance explained across time:")
                total_var = 0
                for component, variance in avg_var_explained.items():
                    print(f"  {component}: {variance:.1%}")
                    total_var += variance
                print(f"  Total: {total_var:.1%}")
                
                # Store results - ENSURE all required keys are present
                alignment_results = {
                    'time_resolved_Z': time_resolved_Z,  # This key MUST be present
                    'time_resolved_variance': time_resolved_var,
                    'average_variance_explained': avg_var_explained,
                    'condition_info': condition_info,
                    'zscore_info': zscore_info,
                    'time_axes': time_axis,
                    'area': area,
                    'alignment': alignment,
                    'train_mod': train_mod,
                    'train_coh': train_coh,
                    'train_delta': train_delta,
                    'dpca_params': self.dpca_params,
                    'subject': self.subject,
                    'date': self.date,
                    'successful_timepoints': successful_timepoints,
                    'total_timepoints': n_time
                }
                
                all_alignment_results[alignment] = alignment_results
                
                # Save results
                if save_results:
                    self.save_dpca_results(alignment_results, area, alignment, 
                                        train_mod, train_coh, train_delta)
                
                # Generate summary plots
                self.plot_dpca_summary_timeresolved(alignment_results)
                
            except Exception as e:
                print(f"ERROR in alignment {alignment}: {e}")
                import traceback
                traceback.print_exc()
                
                # Create empty results to avoid KeyError
                empty_results = {
                    'time_resolved_Z': {'s': np.array([]), 'd': np.array([]), 'sd': np.array([])},
                    'time_resolved_variance': {'s': np.array([]), 'd': np.array([]), 'sd': np.array([])},
                    'average_variance_explained': {'s': 0, 'd': 0, 'sd': 0},
                    'condition_info': {},
                    'zscore_info': {},
                    'time_axes': np.array([]),
                    'area': area,
                    'alignment': alignment,
                    'train_mod': train_mod,
                    'train_coh': train_coh,
                    'train_delta': train_delta,
                    'error': str(e)
                }
                all_alignment_results[alignment] = empty_results
                continue
        
        return all_alignment_results

  

    def save_dpca_results(self, results, area, alignment, train_mod, train_coh, train_delta):
        """
        Save time-resolved dPCA results to file
        """
        filename = f"{self.subject}_{self.date}_{area}_dpca_timeresolved_{alignment}_mod{train_mod}_coh{train_coh}_del{train_delta}_results.npy"
        filepath = self.save_dir / filename
        
        # Prepare data for saving - updated for time-resolved results
        save_data = {
            'subject': self.subject,
            'date': self.date,
            'area': area,
            'alignment': alignment,
            'train_mod': train_mod,
            'train_coh': train_coh,
            'train_delta': train_delta,
            
            # Time-resolved dPCA results
            'time_resolved_Z': results['time_resolved_Z'],
            'time_resolved_variance': results['time_resolved_variance'],
            'average_variance_explained': results['average_variance_explained'],
            
            # Metadata
            'condition_info': results['condition_info'],
            'zscore_info': results['zscore_info'],
            'time_axes': results['time_axes'],
            'dpca_params': results['dpca_params'],
            
            # Data shape information
            'n_time_points': len(results['time_axes']),
            'n_components_saved': {
                's': results['time_resolved_Z']['s'].shape[1],
                'd': results['time_resolved_Z']['d'].shape[1], 
                'sd': results['time_resolved_Z']['sd'].shape[1]
            },
            
            # Analysis metadata
            'analysis_type': 'time_resolved_dpca',
            'labels_used': results['dpca_params']['labels'],
            'regularizer_used': results['dpca_params']['regularizer']
        }
        
        np.save(filepath, save_data, allow_pickle=True)
        print(f"Time-resolved dPCA results saved to: {filepath}")
        
        return filepath

    def plot_dpca_summary_timeresolved(self, results):
        """
        Plot time-resolved dPCA results - FIXED VERSION
        """
        # Check if results contain the required keys
        if 'time_resolved_Z' not in results:
            print(f"Warning: No time_resolved_Z in results. Available keys: {list(results.keys())}")
            return None
        
        if 'error' in results:
            print(f"Skipping plot due to error: {results['error']}")
            return None
        
        time_resolved_Z = results['time_resolved_Z']
        time_resolved_var = results['time_resolved_variance']
        time_axis = results['time_axes']
        alignment = results['alignment']
        
        # Check if we have valid data
        if len(time_axis) == 0:
            print("No valid time axis data for plotting")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Time-Resolved dPCA - {alignment} alignment\n'
                    f'Area: {results["area"]}, Condition: mod{results["train_mod"]}_coh{results["train_coh"]}', 
                    fontsize=14)
        
        try:
            # Plot stimulus component over time
            if len(time_resolved_Z['s']) > 0 and len(time_resolved_Z['s']) == len(time_axis):
                axes[0,0].plot(time_axis, time_resolved_Z['s'][:, 0], 'b-', linewidth=2)
            axes[0,0].set_title('Stimulus Component Over Time')
            axes[0,0].set_ylabel('dPC Activity')
            axes[0,0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            axes[0,0].grid(True, alpha=0.3)
            
            # Plot choice component over time
            if len(time_resolved_Z['d']) > 0 and len(time_resolved_Z['d']) == len(time_axis):
                axes[0,1].plot(time_axis, time_resolved_Z['d'][:, 0], 'r-', linewidth=2)
            axes[0,1].set_title('Choice Component Over Time')
            axes[0,1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot interaction component over time
            if len(time_resolved_Z['sd']) > 0 and len(time_resolved_Z['sd']) == len(time_axis):
                axes[1,0].plot(time_axis, time_resolved_Z['sd'][:, 0], 'g-', linewidth=2)
            axes[1,0].set_title('Stimulus × Choice Interaction Over Time')
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('dPC Activity')
            axes[1,0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            axes[1,0].grid(True, alpha=0.3)
            
            # Plot variance explained over time
            if len(time_resolved_var['s']) > 0 and len(time_resolved_var['s']) == len(time_axis):
                axes[1,1].plot(time_axis, time_resolved_var['s'], 'b-', label='Stimulus', linewidth=2)
                axes[1,1].plot(time_axis, time_resolved_var['d'], 'r-', label='Choice', linewidth=2)
                axes[1,1].plot(time_axis, time_resolved_var['sd'], 'g-', label='Interaction', linewidth=2)
            axes[1,1].set_title('Variance Explained Over Time')
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].set_ylabel('Fraction of Variance')
            axes[1,1].legend()
            axes[1,1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{self.subject}_{self.date}_{results['area']}_dpca_timeresolved_{alignment}_mod{results['train_mod']}_coh{results['train_coh']}.png"
            plot_path = self.save_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            return fig
            
        except Exception as e:
            print(f"Plotting error: {e}")
            plt.close(fig)
            return None
        

    ################# add PDW #########################################
    def prepare_dpca_data_5d(self, spikes, behavior, valid_units, 
                            trial_mod=3, trial_coh=2, trial_del=0,
                            min_trials_per_condition=2):  # Reduced since we have more conditions now
        """
        Prepare data for 5D dPCA analysis: (neurons, time, stimulus, choice, confidence)
        
        Returns:
        --------
        dpca_data : array, shape (n_neurons, n_time, n_stimuli, n_choices, n_confidence_levels)
        condition_info : dict with condition details
        """
        
        if valid_units is not None:
            spikes = spikes[valid_units, :, :]
        
        # Filter trials by modality, coherence, and delta
        if isinstance(trial_del, (list, np.ndarray)):
            del_mask = np.isin(behavior['delta'], trial_del)
        else:
            del_mask = (behavior['delta'] == trial_del)
            
        mask = (behavior['modality'] == trial_mod) & (behavior['coherenceInd'] == trial_coh) & del_mask
        
        # Apply mask
        filtered_spikes = spikes[:, mask, :]
        filtered_behavior = {key: val[mask] for key, val in behavior.items()}
        
        # Define conditions
        stimulus_conditions = filtered_behavior['heading']  # Real headings
        choice_conditions = (filtered_behavior['choice'].astype(int) - 1)  # 0-1
        
        # Get or create confidence conditions
        confidence_conditions = self._get_confidence_conditions(filtered_behavior)
        
        # Get dimensions
        n_neurons, n_trials, n_time = filtered_spikes.shape
        unique_headings = np.sort(np.unique(stimulus_conditions))
        unique_confidence = np.sort(np.unique(confidence_conditions))
        
        n_stimuli = len(unique_headings)
        n_choices = 2
        n_confidence = len(unique_confidence)
        
        print(f"5D dPCA dimensions:")
        print(f"  Neurons: {n_neurons}")
        print(f"  Time points: {n_time}")
        print(f"  Stimuli: {n_stimuli} (headings: {unique_headings})")
        print(f"  Choices: {n_choices}")
        print(f"  Confidence levels: {n_confidence} (levels: {unique_confidence})")
        
        # Initialize 5D dPCA data array
        dpca_data = np.full((n_neurons, n_time, n_stimuli, n_choices, n_confidence), np.nan)
        trial_counts = np.zeros((n_stimuli, n_choices, n_confidence), dtype=int)
        
        # Organize data by all 5 conditions
        print("Organizing data by stimulus × choice × confidence conditions...")
        
        for stim_idx, heading in enumerate(unique_headings):
            for choice_idx in range(n_choices):
                for conf_idx, conf_level in enumerate(unique_confidence):
                    
                    # Find trials matching this condition combination
                    condition_mask = (stimulus_conditions == heading) & \
                                (choice_conditions == choice_idx) & \
                                (confidence_conditions == conf_level)
                    
                    n_condition_trials = np.sum(condition_mask)
                    
                    if n_condition_trials >= min_trials_per_condition:
                        # Average across trials for this condition
                        condition_spikes = filtered_spikes[:, condition_mask, :]
                        mean_response = np.mean(condition_spikes, axis=1)  # (n_neurons, n_time)
                        
                        # Store in 5D dPCA array
                        dpca_data[:, :, stim_idx, choice_idx, conf_idx] = mean_response
                        trial_counts[stim_idx, choice_idx, conf_idx] = n_condition_trials
                    
                    # Progress reporting
                    choice_label = "Left" if choice_idx == 0 else "Right"
                    print(f"  Heading {heading:+.1f}°, {choice_label}, Conf {conf_level}: {n_condition_trials} trials")
        
        # Handle missing conditions for 5D data
        nan_mask = np.isnan(dpca_data)
        if np.any(nan_mask):
            n_missing = np.sum(nan_mask)
            total_possible = np.prod(dpca_data.shape)
            print(f"Found {n_missing}/{total_possible} ({n_missing/total_possible:.1%}) missing condition combinations")
            
            # Fill missing values
            print("Filling missing conditions with mean across available conditions...")
            self._fill_missing_5d(dpca_data, n_neurons, n_time, n_stimuli, n_choices, n_confidence)
        
        condition_info = {
            'trial_counts': trial_counts,
            'stimulus_labels': unique_headings,
            'choice_labels': ['Left', 'Right'],
            'confidence_labels': unique_confidence,
            'total_trials': n_trials,
            'valid_conditions': np.sum(trial_counts >= min_trials_per_condition),
            'total_possible_conditions': n_stimuli * n_choices * n_confidence,
            'trial_mod': trial_mod,
            'trial_coh': trial_coh,
            'trial_del': trial_del
        }
        
        return dpca_data, condition_info


    def _get_confidence_conditions(self, behavior_data):
        """
        Get confidence conditions from behavior data
        """
        
        # Try to find confidence data
        confidence_key = None
        for key in ['confidence', 'wager', 'PDW', 'rating', 'certainty']:
            if key in behavior_data:
                confidence_key = key
                break
        
        if confidence_key is not None:
            confidence_raw = behavior_data[confidence_key]
            print(f"Using {confidence_key} for confidence conditions")
        else:
            print("No confidence data found, creating synthetic confidence from RT")
            # Create binary confidence from RT (fast = high confidence)
            rt = behavior_data['RT']
            rt_median = np.median(rt)
            confidence_raw = (rt < rt_median).astype(int) + 1  # 1=low, 2=high confidence
        
        # Ensure we have reasonable number of confidence levels (2-4)
        unique_conf = np.unique(confidence_raw)
        if len(unique_conf) > 4:
            # Bin into quartiles if too many levels
            quartiles = np.percentile(confidence_raw, [25, 50, 75])
            confidence_binned = np.ones(len(confidence_raw))
            confidence_binned[confidence_raw > quartiles[0]] = 2
            confidence_binned[confidence_raw > quartiles[1]] = 3
            confidence_binned[confidence_raw > quartiles[2]] = 4
            return confidence_binned.astype(int)
        else:
            return confidence_raw.astype(int)

    def _fill_missing_5d(self, dpca_data, n_neurons, n_time, n_stimuli, n_choices, n_confidence):
        """
        Fill missing values in 5D dPCA data
        """
        
        for n in range(n_neurons):
            for t in range(n_time):
                # Get slice for this neuron and time point
                slice_data = dpca_data[n, t, :, :, :]  # (stimuli, choices, confidence)
                
                if np.any(np.isnan(slice_data)):
                    # Fill with mean across available conditions
                    available_mean = np.nanmean(slice_data)
                    if not np.isnan(available_mean):
                        dpca_data[n, t, :, :, :] = np.where(
                            np.isnan(dpca_data[n, t, :, :, :]), 
                            available_mean, 
                            dpca_data[n, t, :, :, :]
                        )
                    else:
                        # Use overall neuron mean if still NaN
                        neuron_mean = np.nanmean(dpca_data[n, :, :, :, :])
                        dpca_data[n, t, :, :, :] = np.where(
                            np.isnan(dpca_data[n, t, :, :, :]), 
                            neuron_mean if not np.isnan(neuron_mean) else 0, 
                            dpca_data[n, t, :, :, :]
                        )

    def setup_dpca_5d(self, brain_area=None):
        """Setup dPCA parameters for 5D analysis"""
        
        self.dpca_params = {
            'labels': 'tsdp',  # time, stimulus, decision, post-decision-wager
            'regularizer': None,
            'n_components': 10  # Might need more components for 5D
        }
        
        print("5D dPCA setup:")
        print(f"  Labels: {self.dpca_params['labels']}")
        print("  t = time, s = stimulus, d = decision, p = confidence (PDW)")
    def run_dpca_5d_analysis(self, spikes_data, behavior_data, time_axes, area, 
                            train_mod=3, train_coh=2, train_delta=0,
                            valid_units=None, save_results=True):

        
        print(f"\n{'='*60}")
        print(f"RUNNING 5D dPCA ANALYSIS")
        print(f"Subject: {self.subject}, Date: {self.date}")
        print(f"Area: {area}, Condition: mod{train_mod}_coh{train_coh}_del{train_delta}")
        print(f"Including CONFIDENCE as 5th dimension")
        print(f"{'='*60}")
        
        # Setup 5D parameters
        self.setup_dpca_5d()
        
        all_alignment_results = {}
        
        # Loop through alignments
        for alignment in ['stimOn', 'saccOnset', 'postTargHold']:
            if alignment not in spikes_data:
                print(f"Skipping {alignment} - not found in spikes_data")
                continue
                
            print(f"\n{'-'*40}")
            print(f"Processing alignment: {alignment}")
            print(f"{'-'*40}")
            
            try:
                spikes = spikes_data[alignment]
                time_axis = time_axes[alignment]
                
                # Prepare 5D data
                dpca_data_5d, condition_info = self.prepare_dpca_data_5d(
                    spikes, behavior_data, valid_units,
                    trial_mod=train_mod, trial_coh=train_coh, trial_del=train_delta
                )
                
                if dpca_data_5d.size == 0:
                    print(f"No valid data for {alignment}, skipping...")
                    continue
                
                print(f"5D dPCA data shape: {dpca_data_5d.shape}")
                print(f"Valid conditions: {condition_info['valid_conditions']}/{condition_info['total_possible_conditions']}")
                
                # Z-score normalize
                dpca_data_processed, zscore_info = self.zscore_normalize_data_5d(dpca_data_5d)
                
                # Run time-resolved 5D dPCA
                print("Running time-resolved 5D dPCA...")
                n_neurons, n_time, n_stimuli, n_choices, n_confidence = dpca_data_processed.shape
                
                # Storage for time-resolved results
                time_resolved_Z = {
                    's': np.zeros((n_time, 5)),    # stimulus
                    'd': np.zeros((n_time, 5)),    # decision/choice  
                    'p': np.zeros((n_time, 5)),    # confidence (PDW)
                    'sd': np.zeros((n_time, 5)),   # stimulus × decision
                    'sp': np.zeros((n_time, 5)),   # stimulus × confidence
                    'dp': np.zeros((n_time, 5)),   # decision × confidence
                    'sdp': np.zeros((n_time, 5))   # stimulus × decision × confidence
                }
                
                time_resolved_var = {comp: np.zeros(n_time) for comp in time_resolved_Z.keys()}
                
                successful_timepoints = 0
                
                for t_idx in range(n_time):
                    if t_idx % 10 == 0:
                        print(f"  Time point {t_idx+1}/{n_time} ({time_axis[t_idx]:.3f}s)")
                    
                    # Extract data for this time point: (neurons, stimulus, choice, confidence)
                    time_point_data = dpca_data_processed[:, t_idx, :, :, :]
                    
                    # Check for valid data
                    if np.any(np.isnan(time_point_data)) or np.all(time_point_data == 0):
                        continue
                    
                    try:
                        # Run 5D dPCA for this time point
                        dpca_model = dPCA(labels='sdp', regularizer=None)  # 3D at each timepoint
                        
                        # Fit and transform
                        Z_t = dpca_model.fit_transform(time_point_data)
                        var_explained_t = dpca_model.explained_variance_ratio_
                        
                        # Store results
                        for component in time_resolved_Z.keys():
                            if component in Z_t and len(Z_t[component]) > 0:
                                # Store first component
                                z_comp = Z_t[component]
                                if z_comp.ndim >= 1 and z_comp.size > 0:
                                    time_resolved_Z[component][t_idx, 0] = np.mean(z_comp.flatten()[:1])
                            
                            # Store variance explained
                            if component in var_explained_t:
                                var_val = var_explained_t[component]
                                if isinstance(var_val, (list, np.ndarray)):
                                    time_resolved_var[component][t_idx] = np.sum(var_val) if len(var_val) > 0 else 0
                                else:
                                    time_resolved_var[component][t_idx] = var_val
                        
                        successful_timepoints += 1
                        
                    except Exception as e:
                        print(f"    5D dPCA failed at time point {t_idx}: {e}")
                        continue
                
                print(f"5D dPCA completed! Successful timepoints: {successful_timepoints}/{n_time}")
                
                # Calculate average variance explained
                avg_var_explained = {comp: np.mean(time_resolved_var[comp]) for comp in time_resolved_var.keys()}
                
                print("\nAverage variance explained across time (5D dPCA):")
                total_var = 0
                for component, variance in avg_var_explained.items():
                    print(f"  {component}: {variance:.1%}")
                    total_var += variance
                print(f"  Total: {total_var:.1%}")
                
                # Store results
                alignment_results = {
                    'time_resolved_Z': time_resolved_Z,
                    'time_resolved_variance': time_resolved_var,
                    'average_variance_explained': avg_var_explained,
                    'condition_info': condition_info,
                    'zscore_info': zscore_info,
                    'time_axes': time_axis,
                    'area': area,
                    'alignment': alignment,
                    'train_mod': train_mod,
                    'train_coh': train_coh,
                    'train_delta': train_delta,
                    'dpca_params': self.dpca_params,
                    'subject': self.subject,
                    'date': self.date,
                    'analysis_type': '5D_dpca',
                    'successful_timepoints': successful_timepoints
                }
                
                all_alignment_results[alignment] = alignment_results
                
                if save_results:
                    self.save_dpca_5d_results(alignment_results, area, alignment, 
                                            train_mod, train_coh, train_delta)
                
                # Plot 5D results
                self.plot_dpca_5d_summary(alignment_results)
                
            except Exception as e:
                print(f"ERROR in 5D analysis for {alignment}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_alignment_results

    def zscore_normalize_data_5d(self, dpca_data_5d):
        """
        Z-score normalize 5D dPCA data
        """
        from scipy import stats
        
        print("Applying z-score normalization to 5D data...")
        
        n_neurons = dpca_data_5d.shape[0]
        dpca_data_zscore = np.zeros_like(dpca_data_5d)
        
        for neuron_idx in range(n_neurons):
            neuron_data = dpca_data_5d[neuron_idx, :, :, :, :]  # All other dimensions
            flat_data = neuron_data.flatten()
            
            valid_mask = ~np.isnan(flat_data)
            if np.sum(valid_mask) > 1:
                flat_zscore = np.full_like(flat_data, np.nan)
                flat_zscore[valid_mask] = stats.zscore(flat_data[valid_mask], nan_policy='omit')
                dpca_data_zscore[neuron_idx, :, :, :, :] = flat_zscore.reshape(neuron_data.shape)
            else:
                dpca_data_zscore[neuron_idx, :, :, :, :] = 0
        
        zscore_info = {
            'method': 'scipy_zscore_5d',
            'final_data_mean': np.nanmean(dpca_data_zscore),
            'final_data_std': np.nanstd(dpca_data_zscore)
        }
        
        print(f"5D Z-score completed: mean={zscore_info['final_data_mean']:.6f}, std={zscore_info['final_data_std']:.6f}")
        
        return dpca_data_zscore, zscore_info
    
    def plot_dpca_5d_summary(self, results):
        """
        Plot 5D dPCA results showing all components including confidence
        """
        
        time_resolved_Z = results['time_resolved_Z']
        time_resolved_var = results['time_resolved_variance']
        time_axis = results['time_axes']
        alignment = results['alignment']
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'5D dPCA Results - {alignment} alignment\n'
                    f'Area: {results["area"]}, Condition: mod{results["train_mod"]}_coh{results["train_coh"]}\n'
                    f'Including Confidence (PDW) Dimension', fontsize=16)
        
        # Component definitions
        components = [
            ('s', 'Stimulus', 'blue'),
            ('d', 'Choice', 'red'), 
            ('p', 'Confidence (PDW)', 'purple'),
            ('sd', 'Stimulus × Choice', 'green'),
            ('sp', 'Stimulus × Confidence', 'orange'),
            ('dp', 'Choice × Confidence', 'brown'),
            ('sdp', 'Stimulus × Choice × Confidence', 'pink')
        ]
        
        # Plot component activities
        for i, (comp, name, color) in enumerate(components):
            if i < 6:  # First 6 plots
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                if comp in time_resolved_Z and len(time_resolved_Z[comp]) > 0:
                    ax.plot(time_axis, time_resolved_Z[comp][:, 0], color=color, linewidth=2)
                
                ax.set_title(f'{name} Component')
                ax.set_ylabel('dPC Activity')
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                ax.grid(True, alpha=0.3)
        
        # Plot variance explained comparison
        ax = axes[2, 0]
        comp_names = [comp[1] for comp in components[:6]]  # First 6 components
        comp_vars = [results['average_variance_explained'].get(comp[0], 0) for comp in components[:6]]
        comp_colors = [comp[2] for comp in components[:6]]
        
        bars = ax.bar(range(len(comp_names)), comp_vars, color=comp_colors, alpha=0.7)
        ax.set_title('Average Variance Explained')
        ax.set_ylabel('Fraction of Variance')
        ax.set_xticks(range(len(comp_names)))
        ax.set_xticklabels(comp_names, rotation=45, ha='right')
        
        # Add percentage labels
        for bar, var in zip(bars, comp_vars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{var:.1%}', ha='center', va='bottom')
        
        # Plot variance over time for key components
        ax = axes[2, 1]
        key_components = [('s', 'Stimulus', 'blue'), ('d', 'Choice', 'red'), ('p', 'Confidence', 'purple')]
        
        for comp, name, color in key_components:
            if comp in time_resolved_var:
                ax.plot(time_axis, time_resolved_var[comp], color=color, linewidth=2, label=name)
        
        ax.set_title('Key Components Variance Over Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fraction of Variance')
        ax.legend()
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Summary statistics
        ax = axes[2, 2]
        summary_text = "5D dPCA Summary:\n\n"
        summary_text += f"Successful timepoints: {results['successful_timepoints']}/{len(time_axis)}\n\n"
        
        summary_text += "Top variance components:\n"
        sorted_components = sorted(results['average_variance_explained'].items(), 
                                key=lambda x: x[1], reverse=True)
        
        for comp, var in sorted_components[:5]:
            comp_name = next((name for c, name, _ in components if c == comp), comp)
            summary_text += f"  {comp_name}: {var:.1%}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save the figure
        figure_filename = f"{self.subject}_{self.date}_{results['area']}_5d_dpca_summary_{alignment}_mod{results['train_mod']}_coh{results['train_coh']}_del{results['train_delta']}.png"
        figure_path = self.save_dir / figure_filename
        
        # Save as PNG with high DPI
        plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"5D dPCA figure saved to: {figure_path}")
        
        # Also save as PDF for publications
        pdf_path = figure_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"5D dPCA figure (PDF) saved to: {pdf_path}")
        
        plt.show()
        
        return fig
    
    def save_dpca_5d_results(self, results, area, alignment, train_mod, train_coh, train_delta):
        """
        Save 5D dPCA results to file
        """
        filename = f"{self.subject}_{self.date}_{area}_dpca_5d_timeresolved_{alignment}_mod{train_mod}_coh{train_coh}_del{train_delta}_results.npy"
        filepath = self.save_dir / filename
        
        # Prepare data for saving - updated for 5D time-resolved results
        save_data = {
            'subject': self.subject,
            'date': self.date,
            'area': area,
            'alignment': alignment,
            'train_mod': train_mod,
            'train_coh': train_coh,
            'train_delta': train_delta,
            
            # 5D Time-resolved dPCA results
            'time_resolved_Z': results['time_resolved_Z'],
            'time_resolved_variance': results['time_resolved_variance'],
            'average_variance_explained': results['average_variance_explained'],
            
            # Metadata
            'condition_info': results['condition_info'],
            'zscore_info': results['zscore_info'],
            'time_axes': results['time_axes'],
            'dpca_params': results['dpca_params'],
            
            # 5D-specific information
            'analysis_type': '5D_dpca_timeresolved',
            'dimensions': {
                'neurons': 'n_neurons (varies by area)',
                'time': 'time points in alignment window',
                'stimulus': 'heading directions',
                'choice': 'left vs right choice',
                'confidence': 'confidence/PDW levels'
            },
            'components_included': list(results['time_resolved_Z'].keys()),
            'component_meanings': {
                's': 'Pure stimulus coding',
                'd': 'Pure choice coding',
                'p': 'Pure confidence coding',
                'sd': 'Stimulus × choice interaction',
                'sp': 'Stimulus × confidence interaction', 
                'dp': 'Choice × confidence interaction',
                'sdp': 'Three-way stimulus × choice × confidence interaction'
            },
            
            # Data shape information
            'n_time_points': len(results['time_axes']),
            'n_components_saved': {
                comp: results['time_resolved_Z'][comp].shape[1] 
                for comp in results['time_resolved_Z'].keys()
            },
            'successful_timepoints': results['successful_timepoints'],
            'success_rate': results['successful_timepoints'] / len(results['time_axes']),
            
            # Condition information
            'n_stimulus_conditions': len(results['condition_info']['stimulus_labels']),
            'n_choice_conditions': len(results['condition_info']['choice_labels']),
            'n_confidence_conditions': len(results['condition_info']['confidence_labels']),
            'total_condition_combinations': results['condition_info']['total_possible_conditions'],
            'valid_condition_combinations': results['condition_info']['valid_conditions'],
            
            # Labels for interpretation
            'stimulus_labels': results['condition_info']['stimulus_labels'],
            'choice_labels': results['condition_info']['choice_labels'],
            'confidence_labels': results['condition_info']['confidence_labels'],
            
            # Analysis parameters used
            'labels_used': results['dpca_params']['labels'],
            'regularizer_used': results['dpca_params']['regularizer'],
            'n_components_requested': results['dpca_params']['n_components'],
            
            # Save timestamp
            'save_timestamp': np.datetime64('now'),
            'analysis_version': '5D_dpca_v1.0'
        }
        
        # Save the data
        np.save(filepath, save_data, allow_pickle=True)
        print(f"5D dPCA results saved to: {filepath}")
        
        return filepath