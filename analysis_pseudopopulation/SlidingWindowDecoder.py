import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, r2_score
import warnings
warnings.filterwarnings('ignore')


class SlidingWindowDecoder:
    
    def __init__(self, subject, date, partial_regress=True, 
                 run_permutation_test=False, n_permutations=100):
        self.subject = subject
        self.date = date
        self.partial_regress = partial_regress
        self.regress = 'partialregress' if partial_regress else 'noregress'
        self.save_dir = Path(rf'D:\Neural-Pipeline\results\analysis_pseudopopulation\decoders_{self.regress}')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.run_permutation_test = run_permutation_test
        self.n_permutations = n_permutations
        
        self.classifier_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(C=0.1, penalty='l1', solver='saga', 
                                              random_state=42, max_iter=1000, class_weight='balanced'))
        ])
        self.regression_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0, random_state=42))
        ])

    def prepare_decoder_data(self, spikes, behavior, valid_units, trial_mod=3, trial_coh=2, trial_del=0):
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

    def preprocess_spikes(self, spikes, behavior, decode_target):
        """Apply partial regression - should be done AFTER pooling pseudopopulation"""
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

    def get_training_headings(self, decode_target):
        if decode_target == 'stimulus':
            return [2, 3, 5, 6]
        else:
            return [2, 3, 5, 6]

    def create_cv_splits(self, behavior, decode_target, n_folds=5, seed=42):
        train_headings = self.get_training_headings(decode_target)
        heading_mask = np.isin(behavior['headingInd'], train_headings)
        n_trials = np.sum(heading_mask)
        
        if n_trials == 0:
            raise ValueError("No trials after filtering for training headings")
        
        actual_n_folds = min(n_folds, n_trials)
        if actual_n_folds < n_folds:
            print(f"  Warning: Reduced n_folds from {n_folds} to {actual_n_folds} (only {n_trials} trials)")
        
        if actual_n_folds < 2:
            raise ValueError(f"Not enough data for CV: only {n_trials} trials")
        
        kf = KFold(n_splits=actual_n_folds, shuffle=True, random_state=seed)
        cv_splits = list(kf.split(np.arange(n_trials)))
        
        return cv_splits, heading_mask, actual_n_folds

    def create_pseudotrials_from_indices(self, spikes, behavior, train_indices, decode_target, 
                                          n_pseudo_per_cond=None, min_trials_per_cond=2,
                                          seed=42, verbose=False):
        rng = np.random.RandomState(seed)
        
        train_spikes = spikes[train_indices]
        train_beh = {k: v[train_indices] for k, v in behavior.items()}
        n_units = train_spikes.shape[1]
        
        heading_map = {h: behavior['heading'][behavior['headingInd'] == h][0] 
                      for h in np.unique(behavior['headingInd'])}
        
        conditions = {}
        skipped = []
        
        for h in np.unique(train_beh['headingInd']):
            for c in np.unique(train_beh['choice']):
                for p in np.unique(train_beh['PDW']):
                    idx = np.where((train_beh['headingInd'] == h) & 
                                  (train_beh['choice'] == c) & 
                                  (train_beh['PDW'] == p))[0]
                    if len(idx) >= min_trials_per_cond:
                        conditions[(h, c, p)] = idx
                    elif len(idx) > 0:
                        skipped.append((h, c, p, len(idx)))
        
        if len(conditions) == 0:
            raise ValueError("No conditions with enough trials")
        
        n_pseudo = n_pseudo_per_cond or max(max(len(v) for v in conditions.values()), 10)
        
        if verbose:
            print(f"\n  Training data:")
            print(f"    Real trials: {len(train_indices)}")
            print(f"    Conditions (≥{min_trials_per_cond} trials): {len(conditions)}")
            if skipped:
                print(f"    Skipped ({len(skipped)} conds with <{min_trials_per_cond} trials)")
            for (h, c, p), idx in sorted(conditions.items()):
                print(f"      h={h}, c={c}, p={p}: {len(idx)} trials")
            print(f"    Pseudotrials per condition: {n_pseudo}")
            print(f"    Total pseudotrials: {n_pseudo * len(conditions)}")
        
        pseudo = {'spikes': [], 'headingInd': [], 'heading': [], 'choice': [], 'PDW': [], 'y': []}
        
        for (h, c, p), idx in conditions.items():
            for _ in range(n_pseudo):
                sampled = rng.choice(idx, size=n_units, replace=True)
                pseudo['spikes'].append([train_spikes[sampled[i], i] for i in range(n_units)])
                pseudo['headingInd'].append(h)
                pseudo['heading'].append(heading_map[h])
                pseudo['choice'].append(c)
                pseudo['PDW'].append(p)
                
                if decode_target == 'choice':
                    pseudo['y'].append(int(c) - 1)
                elif decode_target == 'PDW':
                    pseudo['y'].append(int(p))
                else:
                    pseudo['y'].append(0 if h in [2, 3] else 1)
        
        pseudo_spikes = np.array(pseudo['spikes'])
        pseudo_y = np.array(pseudo['y'])
        pseudo_beh = {k: np.array(pseudo[k]) for k in ['headingInd', 'heading', 'choice', 'PDW']}
        
        return pseudo_spikes, pseudo_y, pseudo_beh

    def get_real_test_data(self, spikes, behavior, test_indices, decode_target):
        test_spikes = spikes[test_indices]
        test_beh = {k: v[test_indices] for k, v in behavior.items()}
        
        if decode_target == 'choice':
            test_y = (test_beh['choice'] - 1).astype(int)
        elif decode_target == 'PDW':
            test_y = test_beh['PDW'].astype(int)
        else:
            test_y = np.where(np.isin(test_beh['headingInd'], [2, 3]), 0, 
                             np.where(np.isin(test_beh['headingInd'], [5, 6]), 1, -1))
        
        return test_spikes, test_y, test_beh

    def get_additional_test_data(self, spikes, behavior, decode_target, test_type):
        if test_type == 'zero':
            mask = behavior['headingInd'] == 4
        elif test_type == 'large':
            mask = np.isin(behavior['headingInd'], [1, 7])
        elif test_type.startswith('delta_'):
            delta_val = int(test_type.split('_')[1])
            mask = behavior['delta'] == delta_val
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
        
        if np.sum(mask) == 0:
            return np.array([]), np.array([]), {}
        
        test_spikes = spikes[mask]
        test_beh = {k: v[mask] for k, v in behavior.items()}
        
        if decode_target == 'choice':
            test_y = (test_beh['choice'] - 1).astype(int)
        elif decode_target == 'PDW':
            test_y = test_beh['PDW'].astype(int)
        else:
            if test_type == 'zero':
                test_y = np.full(np.sum(mask), -1)
            else:
                test_y = np.where(np.isin(test_beh['headingInd'], [1, 2, 3]), 0, 
                                np.where(np.isin(test_beh['headingInd'], [5, 6, 7]), 1, -1))
        
        return test_spikes, test_y, test_beh

    def get_available_delta_values(self, behavior, exclude_delta=0):
        if 'delta' not in behavior:
            return []
        
        unique_deltas = np.unique(behavior['delta'])
        valid_deltas = [d for d in unique_deltas 
                        if d is not None 
                        and not (isinstance(d, float) and np.isnan(d))
                        and d != exclude_delta]
        
        return sorted(valid_deltas)

    def _compute_dv(self, y_proba, y_pred, behavior, decode_target):
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

    def _evaluate(self, X_test, y_test, test_beh, decode_target):
        if len(y_test) == 0:
            return {'y_proba': np.array([]), 'y_pred': np.array([]), 'y_test': y_test, 
                    'DV': np.array([]), 'metrics': {}, 'n_test': 0}
        
        valid_mask = y_test >= 0
        
        if decode_target in ['choice', 'PDW', 'stimulus']:
            y_proba = self.classifier_pipeline.predict_proba(X_test)[:, 1]
            y_pred = self.classifier_pipeline.predict(X_test)
            
            if np.sum(valid_mask) > 0 and len(np.unique(y_test[valid_mask])) > 1:
                metrics = {
                    'accuracy': np.mean(y_pred[valid_mask] == y_test[valid_mask]),
                    'auc': roc_auc_score(y_test[valid_mask], y_proba[valid_mask])
                }
            else:
                metrics = {'accuracy': np.nan, 'auc': np.nan}
            
            DV = self._compute_dv(y_proba, y_pred, test_beh, decode_target)
            
            return {'y_proba': y_proba, 'y_pred': y_pred, 'y_test': y_test, 
                    'DV': DV, 'metrics': metrics, 'n_test': len(y_test)}

    def _permutation_test(self, X_train, y_train, X_test, y_test, test_beh, decode_target, seed=42):
        rng = np.random.RandomState(seed)
        
        true_eval = self._evaluate(X_test, y_test, test_beh, decode_target)
        true_metrics = true_eval['metrics']
        
        perm_metrics = []
        for _ in range(self.n_permutations):
            y_perm = rng.permutation(y_train)
            self.classifier_pipeline.fit(X_train, y_perm)
            perm_eval = self._evaluate(X_test, y_test, test_beh, decode_target)
            perm_metrics.append(perm_eval['metrics'])
        
        results = {'true_metrics': true_metrics, 'perm_metrics': perm_metrics}
        for metric in true_metrics:
            if not np.isnan(true_metrics[metric]):
                perm_vals = np.array([m[metric] for m in perm_metrics])
                results[f'p_{metric}'] = np.mean(perm_vals >= true_metrics[metric])
        
        return results

    def run_decoding_analysis_cv(self, spikes_data, behavior_data, time_axes, area, target, 
                                train_mod=3, train_coh=2, train_delta=0,
                                test_mod=None, test_coh=None, test_delta=None,
                                valid_units=None, n_folds=10, n_pseudo_per_cond=None, 
                                save_results=True):
        if target not in ['choice', 'stimulus', 'PDW']:
            raise ValueError(f"Invalid target: {target}")
        
        test_mod = test_mod if test_mod is not None else train_mod
        test_coh = test_coh if test_coh is not None else train_coh
        test_delta = test_delta if test_delta is not None else train_delta

        train_headings = self.get_training_headings(target)
        
        print(f"{'='*60}")
        print(f"{target.upper()} decoding in {area}")
        print(f"Train: mod={train_mod}, coh={train_coh}, delta={train_delta}")
        print(f"Training headings: {train_headings}")
        print(f"Test: CV held-out + zero [4] + large [1,7] + delta conditions")
        print(f"Partial regress: {self.partial_regress} (applied BEFORE pseudotrial creation)")
        print(f"{'='*60}")
        
        spikes_example = spikes_data['stimOn']
        if valid_units is not None:
            sel_idx = np.where(valid_units)[0] if valid_units.dtype == bool else valid_units
            final_units = np.zeros(spikes_example.shape[0], dtype=bool)
            final_units[sel_idx] = True
        else:
            sel_idx = np.arange(spikes_example.shape[0])
            final_units = None
        
        n_units = len(sel_idx)
        print(f"Units: {n_units}")
        
        spikes_t0 = spikes_example[:, :, 0]
        X_all_t0, beh_all, _ = self.prepare_decoder_data(spikes_t0, behavior_data, final_units, 
                                                         train_mod, train_coh, train_delta)
        
        if X_all_t0.shape[0] == 0:
            print("No data for training condition")
            return {}
        
        available_deltas = self.get_available_delta_values(beh_all, exclude_delta=train_delta)
        if available_deltas:
            print(f"Available delta values for testing: {available_deltas}")
        else:
            print("No additional delta values available for testing")
        
        try:
            cv_splits, heading_mask, actual_n_folds = self.create_cv_splits(beh_all, target, n_folds=n_folds, seed=42)
        except ValueError as e:
            print(f"Cannot create CV splits: {e}")
            return {}
        
        filtered_beh = {k: v[heading_mask] for k, v in beh_all.items()}
        n_total_trials = np.sum(heading_mask)
        n_train_per_fold = int(n_total_trials * (actual_n_folds - 1) / actual_n_folds)
        n_test_per_fold = n_total_trials - n_train_per_fold
        
        n_zero = np.sum(beh_all['headingInd'] == 4)
        n_large = np.sum(np.isin(beh_all['headingInd'], [1, 7]))
        
        print(f"\nData summary:")
        print(f"  Training trials (headings {train_headings}): {n_total_trials}")
        print(f"  CV folds: {actual_n_folds}")
        print(f"  Train/Test per fold: ~{n_train_per_fold}/{n_test_per_fold}")
        print(f"  Additional test - zero [4]: {n_zero} trials")
        print(f"  Additional test - large [1,7]: {n_large} trials")
        for delta_val in available_deltas:
            n_delta = np.sum(beh_all['delta'] == delta_val)
            print(f"  Additional test - delta={delta_val}: {n_delta} trials")
        
        all_results = {}
        
        for alignment in ['stimOn', 'saccOnset', 'postTargHold']:
            print(f"\n{'='*40}\n{alignment}\n{'='*40}")
            
            spikes = spikes_data[alignment]
            n_times = spikes.shape[2]
            time_results, all_coefs = [], []
            
            for t in range(n_times):
                if t % 10 == 0:
                    print(f"  Time {t}/{n_times}")
                
                spikes_t = spikes[:, :, t]
                
                X_all, beh_all_t, _ = self.prepare_decoder_data(spikes_t, behavior_data, final_units, 
                                                                train_mod, train_coh, train_delta)
                
                if X_all.shape[0] == 0:
                    continue
                
                # CRITICAL: Apply partial regression BEFORE creating pseudotrials
                # This ensures all real trials are regressed consistently
                if self.partial_regress:
                    X_all = self.preprocess_spikes(X_all, beh_all_t, target)
                
                X_filtered = X_all[heading_mask]
                beh_filtered = {k: v[heading_mask] for k, v in beh_all_t.items()}
                
                X_zero, y_zero, beh_zero = self.get_additional_test_data(X_all, beh_all_t, target, 'zero')
                X_large, y_large, beh_large = self.get_additional_test_data(X_all, beh_all_t, target, 'large')
                
                delta_tests = {}
                for delta_val in available_deltas:
                    X_delta, y_delta, beh_delta = self.get_additional_test_data(X_all, beh_all_t, target, f'delta_{delta_val}')
                    if len(y_delta) > 0:
                        delta_tests[delta_val] = (X_delta, y_delta, beh_delta)
                
                fold_coefs = []
                for fold, (train_idx, test_idx) in enumerate(cv_splits):
                    
                    verbose = (t == 0 and fold == 0 and alignment == 'stimOn')
                    
                    try:
                        # Create pseudotrials from ALREADY REGRESSED data
                        pseudo_spikes, pseudo_y, pseudo_beh = self.create_pseudotrials_from_indices(
                            X_filtered, beh_filtered, train_idx, target, 
                            n_pseudo_per_cond=n_pseudo_per_cond, seed=42 + fold, verbose=verbose)
                    except ValueError as e:
                        if verbose:
                            print(f"    Error: {e}")
                        continue
                    
                    test_spikes, test_y, test_beh = self.get_real_test_data(
                        X_filtered, beh_filtered, test_idx, target)
                    
                    if len(np.unique(pseudo_y)) < 2:
                        continue
                    
                    self.classifier_pipeline.fit(pseudo_spikes, pseudo_y)
                    
                    coef = self.classifier_pipeline.named_steps['classifier'].coef_[0]
                    full_coef = np.zeros(spikes.shape[0])
                    full_coef[sel_idx] = coef
                    
                    eval_cv = self._evaluate(test_spikes, test_y, test_beh, target)
                    eval_zero = self._evaluate(X_zero, y_zero, beh_zero, target) if len(y_zero) > 0 else None
                    eval_large = self._evaluate(X_large, y_large, beh_large, target) if len(y_large) > 0 else None
                    
                    eval_deltas = {}
                    for delta_val, (X_delta, y_delta, beh_delta) in delta_tests.items():
                        eval_deltas[delta_val] = self._evaluate(X_delta, y_delta, beh_delta, target)
                    
                    perm_result = None
                    if self.run_permutation_test and len(test_y) > 0:
                        perm_result = self._permutation_test(pseudo_spikes, pseudo_y, 
                                                            test_spikes, test_y, test_beh, target)
                    
                    result_dict = {
                        'time': t, 'fold': fold, 'unit_idx': sel_idx, 'coefficients': full_coef,
                        'n_train': len(pseudo_spikes),
                        'test_cv': {
                            'y_proba': eval_cv['y_proba'], 'y_pred': eval_cv['y_pred'],
                            'y_test': eval_cv['y_test'], 'DV': eval_cv['DV'],
                            'metrics': eval_cv['metrics'], 'n_test': eval_cv['n_test'],
                            'behavior': test_beh
                        },
                        'test_zero': {
                            'y_proba': eval_zero['y_proba'] if eval_zero else np.array([]),
                            'y_pred': eval_zero['y_pred'] if eval_zero else np.array([]),
                            'y_test': eval_zero['y_test'] if eval_zero else np.array([]),
                            'DV': eval_zero['DV'] if eval_zero else np.array([]),
                            'metrics': eval_zero['metrics'] if eval_zero else {},
                            'n_test': eval_zero['n_test'] if eval_zero else 0,
                            'behavior': beh_zero
                        },
                        'test_large': {
                            'y_proba': eval_large['y_proba'] if eval_large else np.array([]),
                            'y_pred': eval_large['y_pred'] if eval_large else np.array([]),
                            'y_test': eval_large['y_test'] if eval_large else np.array([]),
                            'DV': eval_large['DV'] if eval_large else np.array([]),
                            'metrics': eval_large['metrics'] if eval_large else {},
                            'n_test': eval_large['n_test'] if eval_large else 0,
                            'behavior': beh_large
                        },
                        'train_behavior': pseudo_beh,
                        'permutation_test': perm_result
                    }
                    
                    for delta_val, eval_delta in eval_deltas.items():
                        result_dict[f'test_delta_{delta_val}'] = {
                            'y_proba': eval_delta['y_proba'],
                            'y_pred': eval_delta['y_pred'],
                            'y_test': eval_delta['y_test'],
                            'DV': eval_delta['DV'],
                            'metrics': eval_delta['metrics'],
                            'n_test': eval_delta['n_test'],
                            'behavior': delta_tests[delta_val][2]
                        }
                    
                    time_results.append(result_dict)
                    fold_coefs.append(full_coef)
                
                if fold_coefs:
                    all_coefs.append(np.array(fold_coefs))
            
            coef_mean = np.mean(np.array(all_coefs), axis=1) if all_coefs else np.array([])
            coef_std = np.std(np.array(all_coefs), axis=1) if all_coefs else np.array([])
            
            all_results[alignment] = {
                'trial_results': time_results, 'coefficients_mean': coef_mean, 'coefficients_std': coef_std,
                'time_axes': time_axes, 'area': area, 'target': target, 'alignment': alignment,
                'train_mod': train_mod, 'train_coh': train_coh, 'train_delta': train_delta,
                'test_mod': test_mod, 'test_coh': test_coh, 'test_delta': test_delta,
                'train_headings': train_headings, 'available_deltas': available_deltas,
                'n_folds': actual_n_folds, 'n_pseudo_per_cond': n_pseudo_per_cond,
                'partial_regress': self.partial_regress
            }
            
            if save_results:
                self._save_results(all_results[alignment], area, target, alignment, 
                                  train_mod, train_coh, test_mod, test_coh)
        
        return all_results

    def _save_results(self, results, area, target, alignment, train_mod, train_coh, test_mod, test_coh):
        filename = f"{self.subject}_{self.date}_{area}_{target}_{alignment}_train_mod{train_mod}_coh{train_coh}_test_mod{test_mod}_coh{test_coh}_results.npy"
        filepath = self.save_dir / filename
        np.save(filepath, results, allow_pickle=True)
        print(f"Saved: {filepath}")