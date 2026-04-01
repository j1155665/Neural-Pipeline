import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class DecoderCrossTester:
    """
    Load pre-trained decoder coefficients and test on new conditions.
    Works with existing saved results - NO need to retrain or modify original code!
    """
    
    def __init__(self, subject, date, partial_regress=True):
        self.subject = subject
        self.date = date
        self.partial_regress = partial_regress
        self.regress = 'partialregress' if partial_regress else 'noregress'
        self.load_dir = Path(rf'D:\Neural-Pipeline\results\analysis_pseudopopulation\decoders_{self.regress}')
        self.save_dir = Path(rf'D:\Neural-Pipeline\results\analysis_pseudopopulation\decoders_{self.regress}_crosstest')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def load_trained_results(self, area, target, alignment, train_mod, train_coh, test_mod=None, test_coh=None):
        """Load the trained decoder results"""
        if test_mod is None:
            test_mod = train_mod
        if test_coh is None:
            test_coh = train_coh
            
        filename = f"{self.subject}_{self.date}_{area}_{target}_{alignment}_train_mod{train_mod}_coh{train_coh}_test_mod{test_mod}_coh{test_coh}_results.npy"
        filepath = self.load_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Trained model not found: {filepath}")
        
        print(f"Loading trained decoder from: {filepath}")
        results = np.load(filepath, allow_pickle=True).item()
        
        return results
    
    def prepare_decoder_data(self, spikes, behavior, valid_units, trial_mod=3, trial_coh=2, trial_del=0):
        """Same as original - prepare data for a specific condition"""
        if valid_units is not None:
            spikes = spikes[valid_units, :]
        
        del_mask = np.isin(behavior['delta'], trial_del) if isinstance(trial_del, (list, np.ndarray)) else (behavior['delta'] == trial_del)
        mask = (behavior['modality'] == trial_mod) & (behavior['coherenceInd'] == trial_coh) & del_mask
        
        if np.sum(mask) == 0:
            return np.array([]), {}, np.array([])
        
        filtered_spikes = spikes[:, mask].T
        behavior_keys = ['choice', 'PDW', 'modality', 'headingInd', 'coherenceInd', 'goodtrial', 
                        'deltaInd', 'correct', 'oneTargChoice', 'oneTargConf', 'heading', 'coherence', 'delta', 'RT']
        filtered_behavior = {k: behavior[k][mask] for k in behavior_keys if k in behavior}
        
        return filtered_spikes, filtered_behavior, np.where(mask)[0]
    
    def prepare_decoder_data_flexible(self, spikes, behavior, valid_units, 
                                     trial_mod=None, trial_coh=None, trial_del=None,
                                     heading_filter=None):
        """
        More flexible data preparation that doesn't require all filters.
        Useful for testing on different deltas without filtering them out first.
        
        Parameters:
        -----------
        trial_mod : int or None
            If None, don't filter by modality
        trial_coh : int or None
            If None, don't filter by coherence
        trial_del : int, list, or None
            If None, don't filter by delta
        heading_filter : list or None
            If provided, filter by specific headings
        """
        if valid_units is not None:
            spikes = spikes[valid_units, :]
        
        # Start with all trials
        mask = np.ones(len(behavior['modality']), dtype=bool)
        
        # Apply modality filter if specified
        if trial_mod is not None:
            mask &= (behavior['modality'] == trial_mod)
        
        # Apply coherence filter if specified
        if trial_coh is not None:
            mask &= (behavior['coherenceInd'] == trial_coh)
        
        # Apply delta filter if specified
        if trial_del is not None:
            if isinstance(trial_del, (list, np.ndarray)):
                mask &= np.isin(behavior['delta'], trial_del)
            else:
                mask &= (behavior['delta'] == trial_del)
        
        # Apply heading filter if specified
        if heading_filter is not None:
            mask &= np.isin(behavior['headingInd'], heading_filter)
        
        if np.sum(mask) == 0:
            return np.array([]), {}, np.array([])
        
        filtered_spikes = spikes[:, mask].T
        behavior_keys = ['choice', 'PDW', 'modality', 'headingInd', 'coherenceInd', 'goodtrial', 
                        'deltaInd', 'correct', 'oneTargChoice', 'oneTargConf', 'heading', 'coherence', 'delta', 'RT']
        filtered_behavior = {k: behavior[k][mask] for k in behavior_keys if k in behavior}
        
        return filtered_spikes, filtered_behavior, np.where(mask)[0]
    
    def get_available_delta_values(self, behavior, exclude_delta=None):
        """Get all available delta values in the dataset"""
        if 'delta' not in behavior:
            return []
        
        unique_deltas = np.unique(behavior['delta'])
        valid_deltas = [d for d in unique_deltas 
                        if d is not None 
                        and not (isinstance(d, float) and np.isnan(d))]
        
        if exclude_delta is not None:
            valid_deltas = [d for d in valid_deltas if d != exclude_delta]
        
        return sorted(valid_deltas)
    
    def preprocess_spikes(self, spikes, behavior, decode_target):
        """Apply partial regression - same as original"""
        stim = behavior['heading']
        choice = (behavior['choice'] - 1).astype(int)
        pdw = behavior['PDW'].astype(int)
        n = len(stim)
        
        X_full = np.column_stack([np.ones(n), stim, choice, pdw])
        coeffs = np.linalg.lstsq(X_full, spikes, rcond=None)[0]
        
        regress_map = {
            'choice': ([0, 1, 3], [np.ones(n), stim, pdw]),
            'stimulus': ([0, 2, 3], [np.ones(n), choice, pdw]),
            'PDW': ([0, 1, 2], [np.ones(n), stim, choice])
        }
        idx, regressors = regress_map[decode_target]
        return spikes - np.column_stack(regressors) @ coeffs[idx, :]
    
    def _get_test_labels(self, behavior, decode_target):
        """Generate test labels based on decode target"""
        if decode_target == 'choice':
            return (behavior['choice'] - 1).astype(int)
        elif decode_target == 'PDW':
            return behavior['PDW'].astype(int)
        else:  # stimulus
            return np.where(np.isin(behavior['headingInd'], [2, 3]), 0, 
                          np.where(np.isin(behavior['headingInd'], [5, 6]), 1, -1))
    
    def _compute_dv(self, y_proba, y_pred, behavior, decode_target):
        """Compute decision variable - same as original"""
        y_proba_clipped = np.clip(y_proba, 1e-10, 1 - 1e-10)
        raw_logodds = np.log(y_proba_clipped / (1 - y_proba_clipped))
        
        if decode_target == 'choice':
            choice_binary = (behavior['choice'] - 1).astype(int) if np.max(behavior['choice']) > 1 else behavior['choice']
            return np.where(choice_binary == 1, raw_logodds, -raw_logodds)
        elif decode_target == 'PDW':
            pdw = behavior['PDW'].astype(int)
            return np.where(pdw == 1, raw_logodds, -raw_logodds)
        else:
            return raw_logodds
    
    def _reconstruct_predictions_from_coefficients(self, X_test, coefficients, train_behavior, unit_idx):
        """Fixed version"""
        from sklearn.preprocessing import StandardScaler
        
        # Extract coefficients for training units
        coef_subset = coefficients[unit_idx]  # Gets 39 values
        
        # DO NOT re-index X_test! It's already filtered to 39 columns
        # Just use it directly
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)  # No [:, unit_idx]!
        
        logits = X_test_scaled @ coef_subset
        y_proba = 1 / (1 + np.exp(-logits))
        y_pred = (y_proba > 0.5).astype(int)
        
        return y_proba, y_pred
    
    def extract_training_stats(self, trained_results):
        """Extract training statistics organized by time and fold"""
        training_info = {}
        
        for result in trained_results['trial_results']:
            t = result['time']
            fold = result['fold']
            
            if t not in training_info:
                training_info[t] = {}
            
            training_info[t][fold] = {
                'coefficients': result['coefficients'],
                'unit_idx': result['unit_idx'],
                'n_train': result['n_train'],
                'train_behavior': result['train_behavior']
            }
        
        return training_info
    
    def test_on_new_condition(self, trained_results, spikes_data, behavior_data, 
                             test_mod=None, test_coh=None, test_delta=None, 
                             valid_units=None, heading_filter=None, save_results=True):
        """
        Test pre-trained decoder on a new condition using saved coefficients.
        
        Parameters:
        -----------
        trained_results : dict
            Loaded results from load_trained_results()
        spikes_data : dict
            Neural data with keys like 'stimOn', 'saccOnset', etc.
        behavior_data : dict
            Behavioral data
        test_mod : int or None
            Test modality (None = don't filter by modality)
        test_coh : int or None
            Test coherence (None = don't filter by coherence)
        test_delta : int, list, or None
            Test delta value(s) (None = don't filter by delta)
        valid_units : array-like
            Unit indices to use (should match training)
        heading_filter : list, optional
            Specific headings to test (e.g., [2,3,5,6] or [1,7] or [4])
        save_results : bool
            Whether to save results
        
        Returns:
        --------
        dict : Test results with predictions and metrics
        """
        
        area = trained_results['area']
        target = trained_results['target']
        alignment = trained_results['alignment']
        train_mod = trained_results['train_mod']
        train_coh = trained_results['train_coh']
        train_delta = trained_results.get('train_delta', 0)
        
        print(f"{'='*60}")
        print(f"Testing {target} decoder on NEW condition")
        print(f"Original training: mod={train_mod}, coh={train_coh}, delta={train_delta}")
        print(f"New test: mod={test_mod}, coh={test_coh}, delta={test_delta}")
        if heading_filter:
            print(f"Heading filter: {heading_filter}")
        print(f"Area: {area}, Alignment: {alignment}")
        print(f"{'='*60}")
        
        # Get the neural data for this alignment
        spikes = spikes_data[alignment]
        n_times = spikes.shape[2]
        
        # Extract training information
        training_info = self.extract_training_stats(trained_results)
        
        all_time_results = []
        
        for t in range(n_times):
            if t % 10 == 0:
                print(f"  Time {t}/{n_times}")
            
            if t not in training_info:
                continue
            
            spikes_t = spikes[:, :, t]
            
            # Use flexible data preparation to allow testing on different deltas
            X_test_all, beh_test_all, _ = self.prepare_decoder_data_flexible(
                spikes_t, behavior_data, valid_units,
                trial_mod=test_mod,
                trial_coh=test_coh,
                trial_del=test_delta,
                heading_filter=heading_filter
            )
            
            if X_test_all.shape[0] == 0:
                continue
            
            X_test = X_test_all
            beh_test = beh_test_all
            
            # Apply same preprocessing as training
            if self.partial_regress:
                X_test = self.preprocess_spikes(X_test, beh_test, target)
            
            # Get test labels
            y_test = self._get_test_labels(beh_test, target)
            
            # Test with each fold's trained coefficients
            fold_results = []
            for fold, fold_info in training_info[t].items():
                
                coefficients = fold_info['coefficients']
                unit_idx = fold_info['unit_idx']
                train_beh = fold_info['train_behavior']
                
                # Reconstruct predictions from coefficients
                y_proba, y_pred = self._reconstruct_predictions_from_coefficients(
                    X_test, coefficients, train_beh, unit_idx
                )
                
                # Compute DV
                DV = self._compute_dv(y_proba, y_pred, beh_test, target)
                
                # Compute metrics
                valid_mask = y_test >= 0
                if np.sum(valid_mask) > 0 and len(np.unique(y_test[valid_mask])) > 1:
                    metrics = {
                        'accuracy': np.mean(y_pred[valid_mask] == y_test[valid_mask]),
                        'auc': roc_auc_score(y_test[valid_mask], y_proba[valid_mask])
                    }
                else:
                    metrics = {'accuracy': np.nan, 'auc': np.nan}
                
                fold_results.append({
                    'fold': fold,
                    'y_proba': y_proba,
                    'y_pred': y_pred,
                    'y_test': y_test,
                    'DV': DV,
                    'metrics': metrics,
                    'n_test': len(y_test),
                    'behavior': beh_test
                })
            
            if fold_results:
                all_time_results.append({
                    'time': t,
                    'fold_results': fold_results
                })
        
        # Compile results
        results = {
            'time_results': all_time_results,
            'trained_from': {
                'area': area,
                'target': target,
                'alignment': alignment,
                'train_mod': train_mod,
                'train_coh': train_coh,
                'train_delta': train_delta
            },
            'test_condition': {
                'test_mod': test_mod,
                'test_coh': test_coh,
                'test_delta': test_delta,
                'heading_filter': heading_filter
            },
            'partial_regress': self.partial_regress
        }
        
        if save_results:
            self._save_test_results(results, area, target, alignment, 
                                   train_mod, train_coh, test_mod, test_coh, test_delta, heading_filter)
        
        return results
    
    def test_multiple_conditions(self, trained_results, spikes_data, behavior_data, 
                                valid_units=None, save_results=True):
        """
        Test on multiple standard conditions at once:
        - Same modality (sanity check)
        - Other single modalities
        - Zero heading [4]
        - Large headings [1, 7]
        - All available delta values
        
        Returns:
        --------
        dict : Results for all test conditions
        """
        area = trained_results['area']
        target = trained_results['target']
        train_mod = trained_results['train_mod']
        train_coh = trained_results['train_coh']
        train_delta = trained_results.get('train_delta', 0)
        
        print(f"\n{'='*60}")
        print(f"Testing {target} decoder on MULTIPLE conditions")
        print(f"Training: mod={train_mod}, coh={train_coh}, delta={train_delta}, area={area}")
        print(f"{'='*60}\n")
        
        # Get available delta values
        available_deltas = self.get_available_delta_values(behavior_data, exclude_delta=train_delta)
        print(f"Available delta values for testing: {available_deltas}\n")
        
        all_results = {}
        
        # Test on same modality (sanity check - should match original results)
        print(f"\n--- Test 1: Same modality (mod={train_mod}, delta={train_delta}) ---")
        all_results['same_modality'] = self.test_on_new_condition(
            trained_results, spikes_data, behavior_data,
            test_mod=train_mod, test_coh=train_coh, test_delta=train_delta,
            valid_units=valid_units, heading_filter=[2, 3, 5, 6],
            save_results=False
        )
        
        # Test on other modalities
        for test_mod in [1, 2, 3]:
            if test_mod == train_mod:
                continue
            
            mod_name = {1: 'combined', 2: 'vestibular', 3: 'visual'}[test_mod]
            print(f"\n--- Test: {mod_name} (mod={test_mod}) ---")
            all_results[f'mod_{test_mod}_{mod_name}'] = self.test_on_new_condition(
                trained_results, spikes_data, behavior_data,
                test_mod=test_mod, test_coh=train_coh, test_delta=train_delta,
                valid_units=valid_units, heading_filter=[2, 3, 5, 6],
                save_results=False
            )
        
        # Test on zero heading
        print(f"\n--- Test: Zero heading [4] (same modality) ---")
        all_results['zero_heading'] = self.test_on_new_condition(
            trained_results, spikes_data, behavior_data,
            test_mod=train_mod, test_coh=train_coh, test_delta=train_delta,
            valid_units=valid_units, heading_filter=[4],
            save_results=False
        )
        
        # Test on large headings
        print(f"\n--- Test: Large headings [1, 7] (same modality) ---")
        all_results['large_headings'] = self.test_on_new_condition(
            trained_results, spikes_data, behavior_data,
            test_mod=train_mod, test_coh=train_coh, test_delta=train_delta,
            valid_units=valid_units, heading_filter=[1, 7],
            save_results=False
        )
        
        # Test on all available delta values (same modality and headings)
        for delta_val in available_deltas:
            print(f"\n--- Test: Delta={delta_val} (mod={train_mod}, headings [2,3,5,6]) ---")
            all_results[f'delta_{delta_val}'] = self.test_on_new_condition(
                trained_results, spikes_data, behavior_data,
                test_mod=train_mod, test_coh=train_coh, test_delta=delta_val,
                valid_units=valid_units, heading_filter=[2, 3, 5, 6],
                save_results=False
            )
        
        # Test cross-modal with different deltas (if you want)
        for test_mod in [1, 2, 3]:
            if test_mod == train_mod:
                continue
            mod_name = {1: 'combined', 2: 'vestibular', 3: 'visual'}[test_mod]
            
            for delta_val in available_deltas:
                print(f"\n--- Test: {mod_name} (mod={test_mod}) + Delta={delta_val} ---")
                all_results[f'mod_{test_mod}_{mod_name}_delta_{delta_val}'] = self.test_on_new_condition(
                    trained_results, spikes_data, behavior_data,
                    test_mod=test_mod, test_coh=train_coh, test_delta=delta_val,
                    valid_units=valid_units, heading_filter=[2, 3, 5, 6],
                    save_results=False
                )
        
        # Compile summary
        summary_results = {
            'all_conditions': all_results,
            'trained_from': trained_results['trained_from'] if 'trained_from' in trained_results else {
                'area': area, 'target': target, 'train_mod': train_mod, 
                'train_coh': train_coh, 'train_delta': train_delta
            },
            'summary': self._create_summary_statistics(all_results)
        }
        
        if save_results:
            self._save_multiple_test_results(summary_results, area, target, 
                                            trained_results['alignment'],
                                            train_mod, train_coh)
        
        return summary_results
    
    def _create_summary_statistics(self, all_results):
        """Create summary statistics across all test conditions"""
        summary = {}
        
        for condition_name, result in all_results.items():
            if 'time_results' not in result:
                continue
            
            # Aggregate metrics across time and folds
            all_accs = []
            all_aucs = []
            
            for time_result in result['time_results']:
                for fold_result in time_result['fold_results']:
                    metrics = fold_result['metrics']
                    if not np.isnan(metrics['accuracy']):
                        all_accs.append(metrics['accuracy'])
                    if not np.isnan(metrics['auc']):
                        all_aucs.append(metrics['auc'])
            
            summary[condition_name] = {
                'mean_accuracy': np.mean(all_accs) if all_accs else np.nan,
                'std_accuracy': np.std(all_accs) if all_accs else np.nan,
                'mean_auc': np.mean(all_aucs) if all_aucs else np.nan,
                'std_auc': np.std(all_aucs) if all_aucs else np.nan,
                'n_timepoints': len(result['time_results'])
            }
        
        return summary
    
    def _save_test_results(self, results, area, target, alignment, 
                          train_mod, train_coh, test_mod, test_coh, test_delta, heading_filter):
        """Save test results"""
        heading_str = f"_h{''.join(map(str, heading_filter))}" if heading_filter else ""
        test_mod_str = f"Mod{test_mod}" if test_mod is not None else "ModAny"
        test_coh_str = f"Coh{test_coh}" if test_coh is not None else "CohAny"
        test_delta_str = f"Del{test_delta}" if test_delta is not None else "DelAny"
        
        filename = (f"{self.subject}_{self.date}_{area}_{target}_{alignment}_"
                   f"trainMod{train_mod}Coh{train_coh}_test{test_mod_str}{test_coh_str}{test_delta_str}"
                   f"{heading_str}_crosstest.npy")
        filepath = self.save_dir / filename
        np.save(filepath, results, allow_pickle=True)
        print(f"\nSaved: {filepath}")
    
    def _save_multiple_test_results(self, results, area, target, alignment, train_mod, train_coh):
        """Save results from multiple test conditions"""
        filename = (f"{self.subject}_{self.date}_{area}_{target}_{alignment}_"
                   f"trainMod{train_mod}Coh{train_coh}_ALLCONDITIONS_crosstest.npy")
        filepath = self.save_dir / filename
        np.save(filepath, results, allow_pickle=True)
        print(f"\n{'='*60}")
        print(f"Saved ALL conditions: {filepath}")
        print(f"{'='*60}")
