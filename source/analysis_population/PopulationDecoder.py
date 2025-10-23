import numpy as np
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class Defineparameters:
    
    def __init__(self, subject, date, save_dir=r'D:\Neural-Pipeline\results\analysis_population\hyperparameters'):
        self.subject = subject
        self.date = date
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.decode_configs = {
            'choice': {'time_window': [-0.2, 0.0], 'alignment': 'saccOnset'},
            'stimulus': {'time_window': [0.5, 0.7], 'alignment': 'stimOn'}, 
            'PDW': {'time_window': [-0.2, 0.0], 'alignment': 'postTargHold'}
        }

    def check_unit_balance(self, dual_units, MST_units, vps_units, balance_threshold=1.5):

        dual_n = dual_units.shape[0]
        mst_n = MST_units.shape[0]
        vps_n = vps_units.shape[0]

        print(f"Unit counts - Dual: {dual_n}, MST: {mst_n}, VPS: {vps_n}")

        # Find min between MST and VPS
        min_units = min(mst_n, vps_n)
        max_units = max(mst_n, vps_n)

        if min_units > 0:
            balance_ratio = max_units / min_units
        else:
            balance_ratio = float('inf')

        is_balanced = balance_ratio <= balance_threshold
        print(f"Is balanced: {is_balanced} (Ratio: {balance_ratio:.2f})")

        if is_balanced:
            target_n = [min_units, max_units]
        else:
            target_n = [min_units, int(balance_threshold * min_units)] 

        balance_info = {}

        for area_name, units, n_units in [('dual', dual_units, dual_n), ('MST', MST_units, mst_n), ('VPS', vps_units, vps_n)]:
            if units is None and area_name == 'dual':
                # Special case for dual (all units)
                balance_info[area_name] = {
                    'units': units,
                    'target_n': target_n[0] + target_n[1],
                    'n_reps': 3 if not is_balanced else 1,
                    'subsampling': True if not is_balanced else False
                }

            elif n_units == target_n[0]:
                # minimun units, no subsampling needed
                balance_info[area_name] = {
                    'units': units,
                    'original_n': n_units,
                    'target_n': target_n[0],
                    'n_reps': 1,
                    'subsampling': False
                }
            elif n_units >= target_n[1]:
                # maximum units, subsampling needed based on target_n
                balance_info[area_name] = {
                    'units': units,
                    'original_n': n_units,
                    'target_n': target_n[1],
                    'n_reps': 3 if not is_balanced  else 1,
                    'subsampling': True if not is_balanced else False
                }

        return balance_info

    def prepare_decoder_data(self, spikes, behavior, time_axis, decode_target, time_window, valid_units, modality=3, coherence=2):
        # Filter valid units first
        if valid_units is not None:
            spikes = spikes[valid_units, :, :]
           
        
        # Filter for modality == 3 and coherence == 2
        mask = (behavior['modality'] == modality) & (behavior['coherenceInd'] == coherence)
        print(f"Total trials: {spikes.shape[1]}, After filtering: {np.sum(mask)}")

        filtered_spikes = spikes[:, mask, :]  
        filtered_behavior = {key: val[mask] for key, val in behavior.items()}

        time_mask = (time_axis >= time_window[0]) & (time_axis <= time_window[1])
        time_indices = np.where(time_mask)[0]

        X = np.mean(filtered_spikes[:, :, time_indices], axis=2).T  # Shape: [trials, neurons]

        if decode_target == 'choice':
            y = filtered_behavior['choice'].astype(int) - 1
            target_mask = np.isin(y, [0, 1])
        elif decode_target == 'PDW':
            y = filtered_behavior['PDW'].astype(int)
            target_mask = np.isin(y, [0, 1])
        elif decode_target == 'stimulus':
            stim_dirs = filtered_behavior['headingInd']
            y = np.zeros_like(stim_dirs, dtype=int)
            class_1_mask = np.isin(stim_dirs, [1, 2, 3])
            class_2_mask = np.isin(stim_dirs, [5, 6, 7])
            y[class_1_mask] = 0
            y[class_2_mask] = 1
            target_mask = class_1_mask | class_2_mask

        X = X[target_mask]
        y = y[target_mask]

        return X, y, np.sum(target_mask)
    
    def train_decoder_with_cv(self, X, y):
        param_grid = {
            'classifier__penalty': ['elasticnet'],  
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'classifier__solver': ['saga'], 
            'classifier__l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            'classifier__class_weight': [
                None, 'balanced', 
                {0: 0.2, 1: 0.8}, {0: 0.5, 1: 0.5}, {0: 0.8, 1: 0.2}
            ]
        }
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            
        ])
        
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Use multiple scoring metrics
        scoring = ['accuracy', 'f1', 'roc_auc', 'balanced_accuracy']
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, 
            scoring=scoring, refit='balanced_accuracy',  # Optimize for balanced accuracy
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X, y)
        
        return grid_search
    
    def run_decoding_analysis(self, decode_target, spikes_data, behavior_data, time_axes, balance_info, valid_units=None, save_results=False):

        if decode_target not in ['choice', 'stimulus', 'PDW']:
            raise ValueError(f"Invalid decode_target: {decode_target}")
        
        config = self.decode_configs[decode_target]
        spikes = spikes_data[config['alignment']]
        time_axis = time_axes[config['alignment']]
        
        print(f"Decoding {decode_target}...")
        
        # Store results from all repetitions
        all_scores = []
        all_results = []
        
        for rep in range(balance_info.get('n_reps', 1)):
            print(f"  Repetition {rep + 1}/{balance_info.get('n_reps', 1)}")
            
            if balance_info.get('subsampling', True):
                # Subsampling case - randomly select from available units
                np.random.seed(42 + rep)
                if valid_units is not None:
                    unit_indices = np.where(valid_units)[0] 
                else:
                    unit_indices = np.arange(spikes.shape[0])  
                
                selected_indices = np.random.choice(
                    unit_indices, 
                    size=balance_info['target_n'], 
                    replace=False
                )
 
                final_units = valid_units[selected_indices] if valid_units is not None else np.zeros(spikes.shape[0], dtype=bool)
   
            else:
                # No subsampling - use all valid units
                final_units = valid_units
            
            if rep == 0:
                print(f" Total units: {spikes.shape[0]}, Valid units: {valid_units.shape[0]},  Using {final_units.shape[0]} units for decoding.")

            # Prepare data for this repetition
            X, y, n_trials = self.prepare_decoder_data(spikes, behavior_data, time_axis, decode_target, config['time_window'], final_units)
            
            if len(np.unique(y)) < 2 or X.shape[0] < 20:
                print(f"Insufficient data for {decode_target} in repetition {rep + 1}")
                continue
            
            # Train decoder for this repetition
            grid_search = self.train_decoder_with_cv(X, y)
            
            # Store results from this repetition
            all_scores.append(grid_search.best_score_)
            all_results.append({
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'n_trials': n_trials,
                'n_neurons': X.shape[1],
                'class_distribution': np.bincount(y).tolist()
            })
        
        # Check if we got any valid results
        if not all_scores:
            print(f"No valid results for {decode_target}")
            return None
        
        # Calculate averaged results
        avg_score = np.mean(all_scores)
        std_score = np.std(all_scores) if len(all_scores) > 1 else 0
        
        # Use parameters from the best-performing repetition
        best_rep_idx = np.argmax(all_scores)
        best_result = all_results[best_rep_idx]
        
        # Create final averaged results
        results = {
            'decode_target': decode_target,
            'subject': self.subject,
            'date': self.date,
            'best_params': best_result['best_params'],
            'best_score': avg_score,
            'score_std': std_score,
            'n_trials': best_result['n_trials'],
            'n_neurons': best_result['n_neurons'],
            'n_valid_units': balance_info.get('target_n', spikes.shape[0]),
            'n_repetitions': len(all_scores),
            'all_scores': all_scores,
            'class_distribution': best_result['class_distribution'],
            'time_window': config['time_window'],
            'alignment': config['alignment']
        }
        
        if len(all_scores) > 1:
            print(f"Average accuracy: {avg_score:.3f} ± {std_score:.3f} ({len(all_scores)} reps)")
        else:
            print(f"Accuracy: {avg_score:.3f}, Trials: {results['n_trials']}, Valid units: {results['n_valid_units']}")
        
        if save_results:
            self.save_results(results, decode_target)
        
        return results
    
    def save_results(self, results, decode_target):
        filename = f"{self.subject}_{self.date}_{decode_target}_decoding_results.npy"
        filepath = self.save_dir / filename
        np.save(filepath, results, allow_pickle=True)
        print(f"Saved: {filename}")    



class SlidingWindowDecoder:
    
    def __init__(self, subject, date, save_dir=r'D:\Neural-Pipeline\results\analysis_population\decoders_partialregression'):
        self.subject = subject
        self.date = date
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_reps = 1  # Add default n_reps
        self.setup_decoder()

    def setup_decoder(self, brain_area=None, decode_target=None): # for now, just use fixed hyperparam, but still load it to get brain info
        file_path = f"D:\\Neural-Pipeline\\results\\analysis_population\\hyperparameters\\{self.subject}_{self.date}_all_areas_all_targets_combined.npy"
        combined_filepath = Path(file_path)
        
        if not combined_filepath.exists():
            raise FileNotFoundError(f"File not found: {combined_filepath}")
        
        # Load hyperparameters (we still need brain area and target to get n_neurons and n_valid_units)
        hyperparams = np.load(file_path, allow_pickle=True).item()
        self.all_hyperparams = hyperparams
        
        if brain_area is None:
            brain_area = list(hyperparams.keys())[0]
        if decode_target is None:
            decode_target = list(hyperparams[brain_area].keys())[0]
        
        if brain_area not in hyperparams:
            raise ValueError(f"Brain area '{brain_area}' not found in hyperparameters")
        if decode_target not in hyperparams[brain_area]:
            raise ValueError(f"Decode target '{decode_target}' not found for area '{brain_area}'")
        
        params = hyperparams[brain_area][decode_target]
                
        self.brain_area = brain_area
        self.decode_target = decode_target
        self.n_neurons = params['n_neurons']
        self.n_valid_units = params['n_valid_units']
        # self.best_params = params['best_params'] # not used for now, just use fixed hyperparam


        self.decoder_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=0.1,                    # Moderate regularization #when C is 1, the pattern is good but variation huge, 0.01 is too strong
                l1_ratio=0.5,            # Equal L1 and L2 penalty
                penalty='elasticnet',     # Keep elasticnet
                solver='saga',
                multi_class='auto',           # Keep saga
                random_state=42,
                max_iter=1000,
                class_weight='balanced'   # Keep balanced for class imbalance
            ))
        ])

    def prepare_decoder_data(self, spikes, behavior, decode_target, valid_units, 
                            trial_mod=3, trial_coh=2, trial_del=0):
        if valid_units is not None:
            spikes = spikes[valid_units, :]
        if isinstance(trial_del, (list, np.ndarray)):
            del_mask = np.isin(behavior['delta'], trial_del)
        else:
            del_mask = (behavior['delta'] == trial_del)
        mask = (behavior['modality'] == trial_mod) & (behavior['coherenceInd'] == trial_coh) & del_mask

        original_indices = np.where(mask)[0]
        filtered_spikes = spikes[:, mask]
        filtered_behavior = {key: val[mask] for key, val in behavior.items()}

        if decode_target == 'choice':
            y_all = filtered_behavior['choice'].astype(int) - 1
            target_mask = np.isin(y_all, [0, 1])
        elif decode_target == 'PDW':
            y_all = filtered_behavior['PDW'].astype(int)
            target_mask = np.isin(y_all, [0, 1])
        elif decode_target == 'stimulus':
            stim_dirs = filtered_behavior['headingInd']
            y_all = np.zeros_like(stim_dirs, dtype=int)
            class_1_mask = np.isin(stim_dirs, [1, 2, 3])
            class_2_mask = np.isin(stim_dirs, [5, 6, 7])
            zero_heading_mask = (filtered_behavior['headingInd'] == 4)
            y_all[class_1_mask] = 0
            y_all[class_2_mask] = 1
            y_all[zero_heading_mask] = -1  # Mark zero heading trials as -1
            target_mask = class_1_mask | class_2_mask | zero_heading_mask
 

        valid_spikes = filtered_spikes[:, target_mask]
        y_valid = y_all[target_mask]
        final_original_indices = original_indices[target_mask]

        valid_behavior = {key: filtered_behavior[key][target_mask] for key in [
            'choice', 'PDW', 'modality', 'headingInd', 'coherenceInd', 
            'goodtrial', 'deltaInd', 'correct', 'oneTargChoice', 'oneTargConf', 
            'heading', 'coherence', 'delta', 'RT'] if key in filtered_behavior}

        if len(y_valid) == 0:
            return np.array([]), np.array([]), np.array([]), {}


        return valid_spikes.T, y_valid, final_original_indices, valid_behavior
    
    def preprocess_spikes(self, train_spikes, test_spikes, decode_target, train_behaviors, test_behaviors, zero_trials_spikes, zero_trials_behaviors):
        
        # if decode_target in ['choice', 'PDW']:  
        #     stim_dirs = train_behaviors['heading']
            
        #     # Fit regression using ONLY training data
        #     X_design = np.column_stack([np.ones(len(stim_dirs)), stim_dirs])  # (n_train_trials, 2)
        #     coeffs = np.linalg.lstsq(X_design, train_spikes, rcond=None)[0]  # (2, n_neurons)
            
        #     # Apply to training data
        #     train_stim_pred = X_design @ coeffs  # (n_train_trials, n_neurons)
            
        #     # Apply SAME coefficients to test data
        #     test_stim_dirs = test_behaviors['heading']
        #     X_test_design = np.column_stack([np.ones(len(test_stim_dirs)), test_stim_dirs])  # (n_test_trials, 2)
        #     test_stim_pred = X_test_design @ coeffs  # (n_test_trials, n_neurons)
            
        #     return (train_spikes - train_stim_pred), (test_spikes - test_stim_pred)
        
        # elif decode_target == 'stimulus':
        #     train_choices = (np.array(train_behaviors['choice']) - 1).astype(int)  # Convert 1,2 -> 0,1
 
        #     X_design = np.column_stack([np.ones(len(train_choices)), train_choices])  # (n_train_trials, 2)
        #     coeffs = np.linalg.lstsq(X_design, train_spikes, rcond=None)[0]  # (2, n_neurons)

        #     train_choice_pred = X_design @ coeffs  # (n_train_trials, n_neurons)

        #     test_choices = (np.array(test_behaviors['choice']) - 1).astype(int)  # Convert 1,2 -> 0,1
        #     X_test_design = np.column_stack([np.ones(len(test_choices)), test_choices])  # (n_test_trials, 2)
        #     test_choice_pred = X_test_design @ coeffs  # (n_test_trials, n_neurons)
            
        #     return (train_spikes - train_choice_pred), (test_spikes - test_choice_pred)
        
        # else:
        #     return train_spikes, test_spikes

            # Fit simultaneous model for both cases: neural_activity = β₀ + β₁×stimulus + β₂×choice
        stim_dirs = train_behaviors['heading']
        train_choices = (np.array(train_behaviors['choice']) - 1).astype(int)  # Convert 1,2 -> 0,1
        
        X_design = np.column_stack([np.ones(len(stim_dirs)), stim_dirs, train_choices])  # (n_train_trials, 3)
        coeffs = np.linalg.lstsq(X_design, train_spikes, rcond=None)[0]  # (3, n_neurons)
        
        # Prepare test data
        test_stim_dirs = test_behaviors['heading']
        test_choices = (np.array(test_behaviors['choice']) - 1).astype(int)  # Convert 1,2 -> 0,1
        
        if decode_target in ['choice', 'PDW']:
            # For choice decoding: regress out stimulus
            regress_coeffs = coeffs[[0, 1], :]  # intercept + stimulus
            X_train_regress = np.column_stack([np.ones(len(stim_dirs)), stim_dirs])
            X_test_regress = np.column_stack([np.ones(len(test_stim_dirs)), test_stim_dirs])
            
        elif decode_target == 'stimulus':
            # For stimulus decoding: regress out choice
            regress_coeffs = coeffs[[0, 2], :]  # intercept + choice
            X_train_regress = np.column_stack([np.ones(len(train_choices)), train_choices])
            X_test_regress = np.column_stack([np.ones(len(test_choices)), test_choices])
        
        # Apply regression
        train_regress_pred = X_train_regress @ regress_coeffs
        test_regress_pred = X_test_regress @ regress_coeffs
            
        return (train_spikes - train_regress_pred), (test_spikes - test_regress_pred)

    

    def run_decoding_analysis_cv(self, spikes_data, behavior_data, time_axes, area, target, 
                                train_mod=3, train_coh=2, train_delta=0, test_mod=3, test_coh=2, test_delta=0,
                                valid_units=None, n_folds=10, save_results=True):
        
        if target not in ['choice', 'stimulus', 'PDW']:
            raise ValueError(f"Invalid target: {target}")
        
        config = self.all_hyperparams[area][target]
        same_condition = (train_mod == test_mod) and (train_coh == test_coh) and (train_delta == test_delta)
        condition_str = f"train_mod{train_mod}_coh{train_coh}_test_mod{test_mod}_coh{test_coh}"
        
        print(f"Decoding {target} in {area}")
        print(f"Condition: {condition_str} ({'same' if same_condition else 'cross'}-condition)")
        
        # Define training conditions for same_condition case
        if same_condition:
            if target == 'choice':
                training_conditions = ['non_zero']  # Skip 'zero_only' due to insufficient data
            elif target == 'PDW':
                training_conditions = ['small_heading_single_target']
            elif target == 'stimulus':
                training_conditions = ['correct_non_zero']
        else:
            training_conditions = ['original']  # Keep original behavior for cross-condition
        
        # Store all results for different training conditions
        all_training_results = {}
        
        # Loop through training conditions
        for train_condition in training_conditions:
            print(f"\n=== Running training condition: {train_condition} ===")
            
            all_alignment_results = {}
            
            # Loop through alignments
            for alignment in ['stimOn', 'saccOnset', 'postTargHold']:
                print(f"\n  Processing alignment: {alignment}")
                
                spikes = spikes_data[alignment]
                all_results = []
                all_coefficients = []    
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                
                # Loop through time points
                for t in range(spikes.shape[2]):  
                    spikes_time = spikes[:, :, t]
                    time_results = []
                    time_coefficients = []

                    # Loop through CV folds
                    for cv_fold in range(n_folds):
                        np.random.seed(42 + cv_fold)
                        
                        if valid_units is not None:
                            unit_indices = np.where(valid_units)[0] 
                        else:
                            unit_indices = np.arange(spikes.shape[0])  
                        
                        selected_indices = np.random.choice(
                            unit_indices, 
                            size=min(config['n_valid_units'], len(unit_indices)), 
                            replace=False
                        )

                        final_units = np.zeros(spikes.shape[0], dtype=bool)
                        final_units[selected_indices] = True

                        train_X_full, train_y_full, original_indices_full, behavior_full = self.prepare_decoder_data(
                                spikes_time, behavior_data, target, final_units,
                                trial_mod=train_mod, trial_coh=train_coh, trial_del=train_delta
                            )

                        
                        if len(train_y_full) == 0 or len(np.unique(train_y_full)) < 2:
                            continue

                        train_idxs, test_idxs = list(skf.split(train_X_full, train_y_full))[cv_fold]
                        
                        # Define masks
                        non_zero_mask = np.isin(behavior_full['headingInd'], [1,2,3,5,6,7])
                        small_heading_mask = np.isin(behavior_full['headingInd'], [2,3,5,6])
                        correct_mask = behavior_full['correct'] == 1
                        zero_mask = behavior_full['headingInd'] == 4
                        
                        if same_condition:
                            # Apply training condition-specific masking
                            if train_condition == 'non_zero':
                                valid_train_mask = non_zero_mask[train_idxs]
                                train_idxs_filtered = train_idxs[valid_train_mask]
                            elif train_condition == 'small_heading_single_target':
                                # For PDW: small heading AND single target (oneTargConf == 0)
                                single_target_mask = behavior_full['oneTargConf'] == 0
                                valid_train_mask = (small_heading_mask & single_target_mask)[train_idxs]
                                train_idxs_filtered = train_idxs[valid_train_mask]
                            elif train_condition == 'correct_non_zero':
                                # valid_train_mask = (correct_mask & non_zero_mask)[train_idxs]
                                valid_train_mask =  non_zero_mask[train_idxs]
                                train_idxs_filtered = train_idxs[valid_train_mask]

                            else:
                                train_idxs_filtered = train_idxs
                            
                            # Check for minimum training data
                            if len(train_idxs_filtered) < 10:
                                print(f"Insufficient training data for {train_condition} ({len(train_idxs_filtered)} samples), skipping")
                                continue

                            test_idxs_filtered = ~train_idxs_filtered
                            
                            # Test on all trials (no masking for test set)
                            X_train = train_X_full[train_idxs_filtered]
                            y_train = train_y_full[train_idxs_filtered]
                            train_original_indices = original_indices_full[train_idxs_filtered]
                            train_behavior = {key: behavior_full[key][train_idxs_filtered] for key in behavior_full}
                            
                            X_test = train_X_full[test_idxs_filtered]
                            y_test = train_y_full[test_idxs_filtered]
                            test_original_indices = original_indices_full[test_idxs_filtered]
                            test_behavior = {key: behavior_full[key][test_idxs_filtered] for key in behavior_full}

                            X_zero = train_X_full[zero_mask]
                            zero_behavior = {key: behavior_full[key][zero_mask] for key in behavior_full}

                            X_train, X_test = self.preprocess_spikes(X_train, X_test, target, train_behavior, test_behavior, X_zero, zero_behavior)

                        else:
                            # Original cross-condition logic
                            X_train = train_X_full[train_idxs]
                            y_train = train_y_full[train_idxs]
                            train_original_indices = original_indices_full[train_idxs]
                            train_behavior = {key: behavior_full[key][train_idxs] for key in behavior_full}
                            
                            X_test, y_test, test_original_indices, test_behavior = self.prepare_decoder_data(
                                spikes_time, behavior_data, target, final_units,
                                trial_mod=test_mod, trial_coh=test_coh, trial_del=test_delta
                            )
                            X_train, X_test = self.preprocess_spikes(X_train, X_test, target, train_behavior, test_behavior, X_zero, zero_behavior)
                        
                        if len(X_train) == 0 or len(X_test) == 0:
                            continue
                        
                        unique_train = np.unique(y_train)
                        unique_test = np.unique(y_test)
                        if len(unique_train) < 2 or len(unique_test) < 2:
                            continue
                        
                        self.decoder_pipeline.fit(X_train, y_train)
                        y_proba = self.decoder_pipeline.predict_proba(X_test)[:, 1]
                        y_pred = self.decoder_pipeline.predict(X_test)
                        
                        y_test_valid = y_test[np.isin(y_test, [0, 1])]
                        y_proba_valid = y_proba[np.isin(y_test, [0, 1])]
                        y_pred_valid = y_pred[np.isin(y_test, [0, 1])]

                        accuracy = np.mean(y_pred_valid == y_test_valid)
                        auc = roc_auc_score(y_test_valid, y_proba_valid)
                        
                        classifier_coef = self.decoder_pipeline.named_steps['classifier'].coef_[0]
                        full_coefficients = np.zeros(spikes.shape[0])
                        full_coefficients[selected_indices] = classifier_coef
                        
                        # Trial results
                        trial_results = {
                            'time': t,
                            'cv_fold': cv_fold,
                            'y_proba': y_proba,
                            'y_pred': y_pred,
                            'y_test': y_test,
                            'accuracy': accuracy,
                            'auc': auc,
                            'selected_units': selected_indices,
                            'coefficients': full_coefficients,
                            'n_train': len(X_train),
                            'n_test': len(X_test),
                            'train_condition': train_condition,
                            'train_indices': train_original_indices,
                            'test_indices': test_original_indices,
                            'test_behavior': {key: test_behavior[key] for key in test_behavior},
                            'train_behavior': {key: train_behavior[key] for key in train_behavior}
                        }
                        
                        time_results.append(trial_results)
                        time_coefficients.append(full_coefficients)
                    
                    all_results.extend(time_results)
                    if time_coefficients:
                        all_coefficients.append(time_coefficients)
                
                if all_coefficients:
                    all_coefficients = np.array(all_coefficients)
                    coeff_mean = np.mean(all_coefficients, axis=1)
                    coeff_std = np.std(all_coefficients, axis=1)
                else:
                    coeff_mean = np.array([])
                    coeff_std = np.array([])
                
                alignment_results = {
                    'trial_results': all_results,
                    'coefficients_mean': coeff_mean,
                    'coefficients_std': coeff_std,
                    'time_axes': time_axes,
                    'config': config,
                    'area': area,
                    'target': target,
                    'alignment': alignment,
                    'train_mod': train_mod,
                    'train_coh': train_coh,
                    'test_mod': test_mod,
                    'test_coh': test_coh,
                    'n_folds': n_folds,
                    'train_condition': train_condition
                }
                
                all_alignment_results[alignment] = alignment_results
                
                if save_results:
                    self.save_results_cv(alignment_results, area, target, alignment, 
                                        train_mod, train_coh, test_mod, test_coh, train_condition)
            
            all_training_results[train_condition] = all_alignment_results
        
        return all_training_results

    def save_results_cv(self, results, area, target, alignment, train_mod, train_coh, test_mod, test_coh, train_condition='original'):
        """Save cross-validation results with training condition in filename"""
        
        # Include train_condition in filename to distinguish different training conditions
        filename = f"{self.subject}_{self.date}_{area}_{target}_{alignment}_train_mod{train_mod}_coh{train_coh}_test_mod{test_mod}_coh{test_coh}_{train_condition}_cv_results.npy"
        filepath = self.save_dir / filename
        
        save_data = {
            'subject': self.subject,
            'date': self.date,
            'area': area,
            'target': target,
            'alignment': alignment,
            'train_mod': train_mod,
            'train_coh': train_coh,
            'test_mod': test_mod,
            'test_coh': test_coh,
            'train_condition': train_condition,
            'trial_results': results['trial_results'],
            'coefficients_mean': results['coefficients_mean'],
            'coefficients_std': results['coefficients_std'],
            'time_axes': results['time_axes'],
            'config': results['config'],
            'n_folds': results['n_folds'],
        }
        
        np.save(filepath, save_data, allow_pickle=True)
        print(f"Results saved to: {filepath}")
        
        return filepath

        
      