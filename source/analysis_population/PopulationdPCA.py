import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from dPCA.dPCA import dPCA

class SlidingWindowdPCA:
    
    def __init__(self, subject, date, save_dir=r'D:\Neural-Pipeline\results\analysis_population\dpca_results'):
        self.subject = subject
        self.date = date
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def fit_linear_model_missing_data(self, dpca_data, stimulus_labels, choice_labels):
        n_neurons, n_time, n_stimuli, n_choices = dpca_data.shape
        filled_data = dpca_data.copy()
        
        print("Fitting linear models to fill missing data...")
        
        available_conditions = []
        available_responses = []
        
        for s_idx, stimulus in enumerate(stimulus_labels):
            for d_idx, choice in enumerate([0, 1]):
                if not np.any(np.isnan(dpca_data[:, :, s_idx, d_idx])):
                    available_conditions.append([stimulus, choice])
                    available_responses.append((s_idx, d_idx))
        
        if len(available_conditions) < 3:
            print("Warning: Too few conditions for linear model fitting, using simple mean imputation")
            return self._simple_mean_imputation(dpca_data)
        
        for neuron_idx in range(n_neurons):
            for time_idx in range(n_time):
                X_available = []
                y_available = []
                
                for i, (s_idx, d_idx) in enumerate(available_responses):
                    stimulus, choice = available_conditions[i]
                    response = dpca_data[neuron_idx, time_idx, s_idx, d_idx]
                    if not np.isnan(response):
                        X_available.append([1, stimulus, choice])
                        y_available.append(response)
                
                if len(y_available) >= 3:
                    X_available = np.array(X_available)
                    y_available = np.array(y_available)
                    
                    try:
                        coefficients = np.linalg.pinv(X_available) @ y_available
                        alpha, beta, gamma = coefficients
                        
                        for s_idx, stimulus in enumerate(stimulus_labels):
                            for d_idx, choice in enumerate([0, 1]):
                                if np.isnan(dpca_data[neuron_idx, time_idx, s_idx, d_idx]):
                                    predicted_value = alpha + beta * stimulus + gamma * choice
                                    filled_data[neuron_idx, time_idx, s_idx, d_idx] = predicted_value
                    
                    except np.linalg.LinAlgError:
                        mean_val = np.nanmean(y_available)
                        for s_idx in range(n_stimuli):
                            for d_idx in range(n_choices):
                                if np.isnan(dpca_data[neuron_idx, time_idx, s_idx, d_idx]):
                                    filled_data[neuron_idx, time_idx, s_idx, d_idx] = mean_val
                else:
                    if len(y_available) > 0:
                        mean_val = np.mean(y_available)
                        for s_idx in range(n_stimuli):
                            for d_idx in range(n_choices):
                                if np.isnan(dpca_data[neuron_idx, time_idx, s_idx, d_idx]):
                                    filled_data[neuron_idx, time_idx, s_idx, d_idx] = mean_val
                    else:
                        neuron_mean = np.nanmean(dpca_data[neuron_idx, :, :, :])
                        if not np.isnan(neuron_mean):
                            for s_idx in range(n_stimuli):
                                for d_idx in range(n_choices):
                                    if np.isnan(dpca_data[neuron_idx, time_idx, s_idx, d_idx]):
                                        filled_data[neuron_idx, time_idx, s_idx, d_idx] = neuron_mean
                        else:
                            for s_idx in range(n_stimuli):
                                for d_idx in range(n_choices):
                                    if np.isnan(dpca_data[neuron_idx, time_idx, s_idx, d_idx]):
                                        filled_data[neuron_idx, time_idx, s_idx, d_idx] = 0
        
        remaining_nans = np.sum(np.isnan(filled_data))
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} NaN values remain after linear model fitting")
            filled_data = np.nan_to_num(filled_data, nan=0.0)
        
        print("Linear model fitting completed!")
        return filled_data
    
    def _simple_mean_imputation(self, dpca_data):
        n_neurons, n_time, n_stimuli, n_choices = dpca_data.shape
        filled_data = dpca_data.copy()
        
        for n in range(n_neurons):
            for t in range(n_time):
                slice_data = dpca_data[n, t, :, :]
                if np.any(np.isnan(slice_data)):
                    available_mean = np.nanmean(slice_data)
                    if not np.isnan(available_mean):
                        filled_data[n, t, :, :] = np.where(np.isnan(dpca_data[n, t, :, :]), 
                                                    available_mean, 
                                                    dpca_data[n, t, :, :])
                    else:
                        neuron_mean = np.nanmean(dpca_data[n, :, :, :])
                        filled_data[n, t, :, :] = np.where(np.isnan(dpca_data[n, t, :, :]), 
                                                    neuron_mean if not np.isnan(neuron_mean) else 0, 
                                                    dpca_data[n, t, :, :])
        return filled_data

    def prepare_dpca_data(self, spikes, behavior, valid_units, 
                        trial_mod=3, trial_coh=2, trial_del=0,
                        min_trials_per_condition=3):
        
        if valid_units is not None:
            spikes = spikes[valid_units, :, :]
        
        if isinstance(trial_del, (list, np.ndarray)):
            del_mask = np.isin(behavior['delta'], trial_del)
        else:
            del_mask = (behavior['delta'] == trial_del)
            
        mask = (behavior['modality'] == trial_mod) & (behavior['coherenceInd'] == trial_coh) & del_mask
        
        filtered_spikes = spikes[:, mask, :]
        filtered_behavior = {key: val[mask] for key, val in behavior.items()}
        
        stimulus_conditions = filtered_behavior['heading']
        choice_conditions = (filtered_behavior['choice'].astype(int) - 1)
        
        n_neurons, n_trials, n_time = filtered_spikes.shape
        unique_headings = np.sort(np.unique(stimulus_conditions))
        n_stimuli = len(unique_headings)
        n_choices = 2
        
        dpca_data = np.full((n_neurons, n_time, n_stimuli, n_choices), np.nan)
        trial_counts = np.zeros((n_stimuli, n_choices), dtype=int)
        
        print("Organizing data by heading and choice conditions...")
        
        for stim_idx, heading in enumerate(unique_headings):
            for choice_idx in range(2):
                condition_mask = (stimulus_conditions == heading) & (choice_conditions == choice_idx)
                n_condition_trials = np.sum(condition_mask)
                
                if n_condition_trials >= min_trials_per_condition:
                    condition_spikes = filtered_spikes[:, condition_mask, :]
                    mean_response = np.mean(condition_spikes, axis=1)
                    dpca_data[:, :, stim_idx, choice_idx] = mean_response
                    trial_counts[stim_idx, choice_idx] = n_condition_trials
        
        nan_mask = np.isnan(dpca_data)
        if np.any(nan_mask):
            complete_conditions = 0
            for s in range(n_stimuli):
                for d in range(n_choices):
                    if not np.any(np.isnan(dpca_data[:, :, s, d])):
                        complete_conditions += 1
            
            print(f"Complete conditions: {complete_conditions}/{n_stimuli * n_choices}")
            
            if complete_conditions < 3:
                raise ValueError(f"Too few complete conditions ({complete_conditions}). "
                            f"Need at least 3 for parametric fitting.")
            
            print("Using parametric linear model to fill missing conditions...")
            dpca_data = self.fit_linear_model_missing_data(dpca_data, unique_headings, [0, 1])
        
        condition_info = {
            'trial_counts': trial_counts,
            'stimulus_labels': unique_headings,
            'choice_labels': ['Left', 'Right'],
            'total_trials': n_trials,
            'valid_conditions': np.sum(trial_counts >= min_trials_per_condition),
            'trial_mod': trial_mod,
            'trial_coh': trial_coh,
            'trial_del': trial_del,
            'heading_range': (np.min(unique_headings), np.max(unique_headings)),
            'n_headings': len(unique_headings),
            'missing_data_method': 'parametric_linear_model'
        }
        
        return dpca_data, condition_info

    def cross_validate_regularization(self, spikes, behavior, valid_units, 
                                    trial_mod=3, trial_coh=2, trial_del=0,
                                    lambda_range=None, n_cv_folds=10, n_components_per_marg=10):
        
        if lambda_range is None:
            lambda_range = np.logspace(-7, -3, 20)
        
        print(f"Cross-validating regularization with {len(lambda_range)} values, {n_cv_folds} folds...")
        
        if valid_units is not None:
            spikes = spikes[valid_units, :, :]
        
        if isinstance(trial_del, (list, np.ndarray)):
            del_mask = np.isin(behavior['delta'], trial_del)
        else:
            del_mask = (behavior['delta'] == trial_del)
            
        mask = (behavior['modality'] == trial_mod) & (behavior['coherenceInd'] == trial_coh) & del_mask
        
        filtered_spikes = spikes[:, mask, :]
        filtered_behavior = {key: val[mask] for key, val in behavior.items()}
        
        stimulus_conditions = filtered_behavior['heading']
        choice_conditions = (filtered_behavior['choice'].astype(int) - 1)
        
        n_neurons, n_trials, n_time = filtered_spikes.shape
        unique_headings = np.sort(np.unique(stimulus_conditions))
        n_stimuli = len(unique_headings)
        n_choices = 2
        
        cv_errors_all_folds = {lam: [] for lam in lambda_range}
        
        print("Running cross-validation folds...")
        
        for fold in range(n_cv_folds):
            print(f"  Fold {fold+1}/{n_cv_folds}")
            
            X_train, X_test = self._create_cv_split(
                filtered_spikes, stimulus_conditions, choice_conditions, 
                unique_headings, n_stimuli, n_choices, n_time
            )
            
            if X_train is None or X_test is None:
                print(f"    Skipping fold {fold+1} - insufficient data")
                continue
            
            for lam in lambda_range:
                try:
                    if lam > 0:
                        data_norm = np.linalg.norm(X_train)
                        regularizer = (lam * data_norm) ** 2
                    else:
                        regularizer = None
                    
                    dpca_model = dPCA(labels='tsd', regularizer=regularizer)
                    Z_train = dpca_model.fit_transform(X_train)
                    
                    cv_error = self._compute_cv_error(dpca_model, X_train, X_test, Z_train)
                    cv_errors_all_folds[lam].append(cv_error)
                    
                except Exception as e:
                    print(f"    Lambda {lam:.2e} failed in fold {fold+1}: {e}")
                    continue
        
        mean_cv_errors = {}
        std_cv_errors = {}
        
        for lam in lambda_range:
            if len(cv_errors_all_folds[lam]) > 0:
                mean_cv_errors[lam] = np.mean(cv_errors_all_folds[lam])
                std_cv_errors[lam] = np.std(cv_errors_all_folds[lam])
            else:
                mean_cv_errors[lam] = np.inf
                std_cv_errors[lam] = 0
        
        valid_lambdas = [lam for lam in lambda_range if mean_cv_errors[lam] < np.inf]
        if len(valid_lambdas) == 0:
            print("Warning: No valid lambda values found, using no regularization")
            return 0, cv_errors_all_folds
        
        optimal_lambda = min(valid_lambdas, key=lambda x: mean_cv_errors[x])
        
        print(f"Cross-validation results:")
        print(f"  Optimal λ = {optimal_lambda:.2e}")
        print(f"  CV error = {mean_cv_errors[optimal_lambda]:.6f} ± {std_cv_errors[optimal_lambda]:.6f}")
        
        return optimal_lambda, cv_errors_all_folds

    def _create_cv_split(self, filtered_spikes, stimulus_conditions, choice_conditions, 
                        unique_headings, n_stimuli, n_choices, n_time):
        n_neurons = filtered_spikes.shape[0]
        
        X_train = np.full((n_neurons, n_time, n_stimuli, n_choices), np.nan)
        X_test = np.full((n_neurons, n_time, n_stimuli, n_choices), np.nan)
        
        for stim_idx, heading in enumerate(unique_headings):
            for choice_idx in range(n_choices):
                condition_mask = (stimulus_conditions == heading) & (choice_conditions == choice_idx)
                condition_trials = np.where(condition_mask)[0]
                
                if len(condition_trials) < 2:
                    continue
                
                condition_spikes = filtered_spikes[:, condition_mask, :]
                
                for neuron_idx in range(n_neurons):
                    neuron_trials = condition_spikes[neuron_idx, :, :]
                    
                    if neuron_trials.shape[0] >= 2:
                        test_trial_idx = np.random.randint(0, neuron_trials.shape[0])
                        test_trial = neuron_trials[test_trial_idx, :]
                        
                        train_trials = np.delete(neuron_trials, test_trial_idx, axis=0)
                        train_avg = np.mean(train_trials, axis=0)
                        
                        X_test[neuron_idx, :, stim_idx, choice_idx] = test_trial
                        X_train[neuron_idx, :, stim_idx, choice_idx] = train_avg
        
        if np.all(np.isnan(X_train)) or np.all(np.isnan(X_test)):
            return None, None
        
        if np.any(np.isnan(X_train)):
            X_train = self.fit_linear_model_missing_data(X_train, unique_headings, [0, 1])
        if np.any(np.isnan(X_test)):
            X_test = self.fit_linear_model_missing_data(X_test, unique_headings, [0, 1])
        
        return X_train, X_test

    def _compute_cv_error(self, dpca_model, X_train, X_test, Z_train):
        try:
            Z_test = dpca_model.transform(X_test)
            X_train_reconstructed = dpca_model.inverse_transform(Z_test)
            
            reconstruction_error = np.sum((X_train - X_train_reconstructed) ** 2)
            total_variance = np.sum(X_train ** 2)
            
            if total_variance > 0:
                normalized_error = reconstruction_error / total_variance
            else:
                normalized_error = np.inf
            
            return normalized_error
            
        except Exception as e:
            return np.inf

    def run_dpca_analysis(self, spikes_data, behavior_data, time_axes, area, 
                        train_mod=3, train_coh=2, train_delta=0,
                        valid_units=None, save_results=True, 
                        use_cv_regularization=False):
        
        print(f"\n{'='*60}")
        print(f"RUNNING FULL SPIKE TRAIN dPCA ANALYSIS")
        print(f"Subject: {self.subject}, Date: {self.date}")
        print(f"Area: {area}, Condition: mod{train_mod}_coh{train_coh}_del{train_delta}")
        print(f"{'='*60}")
        
        all_alignment_results = {}
        
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
                
                if use_cv_regularization:
                    optimal_lambda, cv_results = self.cross_validate_regularization(
                        spikes, behavior_data, valid_units,
                        trial_mod=train_mod, trial_coh=train_coh, trial_del=train_delta
                    )
                    
                    if optimal_lambda > 0:
                        print(f"Using CV-selected regularization: λ = {optimal_lambda:.2e}")
                    else:
                        print("CV selected no regularization")
                else:
                    optimal_lambda = 10^-2
                    cv_results = None
                
                dpca_data, condition_info = self.prepare_dpca_data(
                    spikes, behavior_data, valid_units,
                    trial_mod=train_mod, trial_coh=train_coh, trial_del=train_delta
                )
                
                if dpca_data.size == 0:
                    print(f"No valid data for {alignment}, skipping...")
                    continue
                
                print(f"dPCA data shape: {dpca_data.shape}")
                print(f"Valid conditions: {condition_info['valid_conditions']}/14")
                
                if optimal_lambda > 0:
                    data_norm = np.linalg.norm(dpca_data)
                    regularizer = (optimal_lambda * data_norm) ** 2
                    print(f"Final regularizer μ = {regularizer:.2e}")
                else:
                    regularizer = None
                
                dpca_model = dPCA(labels='tsd', regularizer=regularizer)
                Z = dpca_model.fit_transform(dpca_data)
                
                results_Z = {}
                for component in Z.keys():
                    if component == 's' and 'ts' in Z:
                        results_Z[component] = Z['ts'] + Z[component]
                    elif component == 'd' and 'td' in Z:
                        results_Z[component] = Z['td'] + Z[component]
                    else:
                        results_Z[component] = Z[component]
                
                variance_explained = dpca_model.explained_variance_ratio_
                
                alignment_results = {
                    'Z_components': results_Z,
                    'variance_explained': variance_explained,
                    'condition_info': condition_info,
                    'time_axes': time_axis,
                    'area': area,
                    'alignment': alignment,
                    'train_mod': train_mod,
                    'train_coh': train_coh,
                    'train_delta': train_delta,
                    'dpca_params': {'labels': 'tsd', 'regularizer': regularizer},
                    'subject': self.subject,
                    'date': self.date,
                    'analysis_type': 'full_spike_train',
                    'optimal_lambda': optimal_lambda,
                    'cv_results': cv_results,
                    'transformation_weights': dpca_model.D  # Added this line
                }
                
                all_alignment_results[alignment] = alignment_results

                if save_results:
                    self.save_dpca_results(alignment_results, area, alignment, 
                                        train_mod, train_coh, train_delta)
                
            except Exception as e:
                print(f"ERROR in alignment {alignment}: {e}")
                continue
        
        return all_alignment_results

    def plot_dpca_summary_fullspike(self, results):
        if 'Z_components' not in results:
            print(f"Warning: No Z_components in results. Available keys: {list(results.keys())}")
            return None
        
        if 'error' in results:
            print(f"Skipping plot due to error: {results['error']}")
            return None
        
        Z_components = results['Z_components']
        variance_explained = results['variance_explained']
        time_axis = results['time_axes']
        alignment = results['alignment']
        
        def sum_variance(*var_keys):
            total = 0
            for key in var_keys:
                if key in variance_explained:
                    var_val = variance_explained[key]
                    if isinstance(var_val, (list, np.ndarray)):
                        total += np.sum(var_val)
                    else:
                        total += var_val
            return total
        
        pooled_Z = {}
        pooled_var = {}
        
        stimulus_components = []
        if 's' in Z_components:
            s_comp = Z_components['s'][0]
            stimulus_components.append(s_comp)
        if 'ts' in Z_components:
            ts_comp = Z_components['ts'][0]
            stimulus_components.append(ts_comp)
        
        if stimulus_components:
            if len(stimulus_components) == 1:
                pooled_Z['stimulus'] = stimulus_components[0]
            else:
                pooled_Z['stimulus'] = np.sum(stimulus_components, axis=0)
            pooled_var['stimulus'] = sum_variance('s', 'ts')
        
        choice_components = []
        if 'd' in Z_components:
            d_comp = Z_components['d'][0]
            choice_components.append(d_comp)
        if 'td' in Z_components:
            td_comp = Z_components['td'][0]
            choice_components.append(td_comp)
        
        if choice_components:
            if len(choice_components) == 1:
                pooled_Z['choice'] = choice_components[0]
            else:
                pooled_Z['choice'] = np.sum(choice_components, axis=0)
            pooled_var['choice'] = sum_variance('d', 'td')
        
        interaction_components = []
        if 'sd' in Z_components:
            sd_comp = Z_components['sd'][0]
            interaction_components.append(sd_comp)
        if 'tsd' in Z_components:
            tsd_comp = Z_components['tsd'][0]
            interaction_components.append(tsd_comp)
        
        if interaction_components:
            if len(interaction_components) == 1:
                pooled_Z['interaction'] = interaction_components[0]
            else:
                pooled_Z['interaction'] = np.sum(interaction_components, axis=0)
            pooled_var['interaction'] = sum_variance('sd', 'tsd')
        
        condition_indep_components = []
        if 't' in Z_components:
            t_comp = Z_components['t'][0]
            condition_indep_components.append(t_comp)
        
        if condition_indep_components:
            pooled_Z['condition_independent'] = condition_indep_components[0]
            pooled_var['condition_independent'] = sum_variance('t')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Pooled dPCA Components - {alignment} alignment\n'
                    f'Area: {results["area"]}, Condition: mod{results["train_mod"]}_coh{results["train_coh"]}', 
                    fontsize=14)
        
        try:
            if 'stimulus' in pooled_Z:
                stim_data = pooled_Z['stimulus']
                stim_avg = np.mean(stim_data, axis=(1, 2))
                axes[0,0].plot(time_axis, stim_avg, 'b-', linewidth=2)
                axes[0,0].set_title(f'Stimulus Component (s + ts)\nVar: {pooled_var.get("stimulus", 0):.1%}')
            else:
                axes[0,0].set_title('Stimulus Component (no data)')
            axes[0,0].set_ylabel('dPC Activity')
            axes[0,0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            axes[0,0].grid(True, alpha=0.3)
            
            if 'choice' in pooled_Z:
                choice_data = pooled_Z['choice']
                choice_avg = np.mean(choice_data, axis=(1, 2))
                axes[0,1].plot(time_axis, choice_avg, 'r-', linewidth=2)
                axes[0,1].set_title(f'Choice Component (d + td)\nVar: {pooled_var.get("choice", 0):.1%}')
            else:
                axes[0,1].set_title('Choice Component (no data)')
            axes[0,1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            axes[0,1].grid(True, alpha=0.3)
            
            if 'interaction' in pooled_Z:
                int_data = pooled_Z['interaction']
                int_avg = np.mean(int_data, axis=(1, 2))
                axes[1,0].plot(time_axis, int_avg, 'g-', linewidth=2)
                axes[1,0].set_title(f'Interaction Component (sd + tsd)\nVar: {pooled_var.get("interaction", 0):.1%}')
            else:
                axes[1,0].set_title('Interaction Component (no data)')
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('dPC Activity')
            axes[1,0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            axes[1,0].grid(True, alpha=0.3)
            
            if 'condition_independent' in pooled_Z:
                ci_data = pooled_Z['condition_independent']
                ci_avg = np.mean(ci_data, axis=(1, 2))
                axes[1,1].plot(time_axis, ci_avg, 'k-', linewidth=2)
                axes[1,1].set_title(f'Condition Independent (t + rest)\nVar: {pooled_var.get("condition_independent", 0):.1%}')
            else:
                axes[1,1].set_title('Condition Independent (no data)')
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_filename = f"{self.subject}_{self.date}_{results['area']}_dpca_pooled_{alignment}_mod{results['train_mod']}_coh{results['train_coh']}.png"
            plot_path = self.save_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\nPooled variance explained:")
            for category, var_val in pooled_var.items():
                print(f"  {category}: {var_val:.1%}")
            
            return fig
            
        except Exception as e:
            print(f"Plotting error: {e}")
            import traceback
            traceback.print_exc()
            plt.close(fig)
            return None

    def save_dpca_results(self, results, area, alignment, train_mod, train_coh, train_delta):
        filename = f"{self.subject}_{self.date}_{area}_dpca_fullspike_{alignment}_mod{train_mod}_coh{train_coh}_del{train_delta}_results.npy"
        filepath = self.save_dir / filename
        
        save_data = {
            'subject': self.subject,
            'date': self.date,
            'area': area,
            'alignment': alignment,
            'train_mod': train_mod,
            'train_coh': train_coh,
            'train_delta': train_delta,
            'Z_components': results['Z_components'],
            'variance_explained': results['variance_explained'],
            'condition_info': results['condition_info'],
            'time_axes': results['time_axes'],
            'dpca_params': results['dpca_params'],
            'analysis_type': 'full_spike_train_dpca',
            'labels_used': results['dpca_params']['labels'],
            'transformation_weights': results['transformation_weights']  # Added this line
        }
        
        np.save(filepath, save_data, allow_pickle=True)
        print(f"Full spike train dPCA results saved to: {filepath}")
        
        return filepath